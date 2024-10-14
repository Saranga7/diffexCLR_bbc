import logging
import os
import time
from typing import List, Optional

import wandb
from wandb import Image
import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# from colat.generators import Generator
from colat.metrics import LossMetric
from diffae.experiment import LitModel


class Trainer:
    """Model trainer

    Args:
        model: model to train
        loss_fn: loss function
        optimizer: model optimizer
        generator: pretrained generator
        projector: pretrained projector
        device: device to train the model on
        batch_size: number of batch elements
        iterations: number of iterations
        scheduler: learning rate scheduler
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training
        mixed_precision: enable mixed precision training

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: LitModel,
        projector: torch.nn.Module,
        batch_size: int,
        # iterations: int,
        device: torch.device,
        eval_freq: int = 1000,
        eval_iters: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_max_norm: Optional[float] = None,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        mixed_precision: bool = False,
        train_projector: bool = True,
        feed_layers: Optional[List[int]] = None,
        epochs : int = 10,

        conf = None,
        mode : str = 'non_ema'
    ) -> None:

        # Logging
        self.logger = logging.getLogger()
        self.writer = writer

        # Saving
        self.save_path = save_path

        # Device
        self.device = device

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.projector = projector
        self.train_projector = train_projector
        self.feed_layers = feed_layers

        #  Eval
        self.eval_freq = eval_freq
        self.eval_iters = eval_iters

        # Scheduler
        self.scheduler = scheduler
        self.grad_clip_max_norm = grad_clip_max_norm

        # Batch & Iteration
        self.batch_size = batch_size
        # self.iterations = iterations
        self.start_epoch = 0
        self.epochs = epochs

        # Floating-point precision
        self.mixed_precision = (
            True if self.device.type == "cuda" and mixed_precision else False
        )
        self.scaler = GradScaler() if self.mixed_precision else None

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.train_acc_metric = LossMetric()
        self.train_loss_metric = LossMetric()

        self.val_acc_metric = LossMetric()
        self.val_loss_metric = LossMetric()

        # Best
        self.best_loss = -1

        #DiffAE
        self.conf = conf
        self.mode = mode

    def train(self) -> None:
        """Trains the model"""

        train_data = self.conf.make_dataset(split = 'train')
        test_data = self.conf.make_dataset(split = 'test')

        print(f"Training set size: {len(train_data)}")
        print(f"Testing set size: {len(test_data)}")

 
        train_dataloader = self.conf.make_loader(train_data, shuffle = True, drop_last = True)
        test_dataloader = self.conf.make_loader(test_data, shuffle = False, drop_last = True)


        self.logger.info("Beginning training")
        start_time = time.time()

        steps = 0
        for epoch in range(self.epochs):
            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(epoch, train_dataloader, steps)
            else:
                self._train_loop(epoch, train_dataloader, steps)

            self._val_loop(epoch, test_dataloader)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time)

        
        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(
            os.path.join(self.save_path, "final_model.pt"), epoch
        )

    def _encode_and_concatenate_classifier(self, imgs):
        w = self.generator.encode(imgs, mode = self.mode)
        if self.mode == 'ema':
            concat_classifier = self.generator.ema_model.classifier_component
        else:
            concat_classifier = self.generator.model.classifier_component
        z_sem = concat_classifier(imgs, w)
        return z_sem



    def _train_loop(self, epoch: int, dataloader, steps: int) -> None:
        """
        Regular train loop

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Set to eval
        self.generator.eval()

        if self.train_projector:
            self.projector.train()
        else:
            self.projector.eval()

        iterations = len(dataloader) * epoch

        for batch in pbar:
            imgs = batch['img'].to(self.device)
            # grid = (make_grid(imgs) + 1) / 2
            # wandb.log({'Original images': [Image(grid)]})
            
            # print("Encode and concatenate classifier")
            z_sem = self._encode_and_concatenate_classifier(imgs)
            z_sem_orig = z_sem
            # print("Encoding noise map of the original image")
            # xT = self.generator.encode_stochastic(imgs, z_sem, T = 50, mode = self.mode)

            # Original features
            with torch.no_grad():
                # print(z_sem_orig.shape)
                orig_feats = self.projector(z_sem_orig)
            
  
            # Apply Directions
            self.optimizer.zero_grad()
            z_sem = self.model(z_sem)
            # print("\n\n\n")
            # print(z_sem.shape)

            # Forward
            features = []
            for j in range(z_sem.shape[0] // self.batch_size):
                # Prepare batch
                start, end = j * self.batch_size, (j + 1) * self.batch_size
                z_sem_batch = z_sem[start:end, ...]

                
                 # Get features
                # feats = self.generator.get_features(z_batch)

                # Generate image with shifted z_sems and encode again
                # print("Generating")
                # shifted_generated_images = self.generator.render(xT, z_sem_batch, T = 20, mode = self.mode)
                # grid = (make_grid(shifted_generated_images) + 1) / 2
                # wandb.log({f'Shifted_{j}': [Image(grid)]})
             
        
                # feats = self._encode_and_concatenate_classifier(shifted_generated_images)
                feats = self.projector(z_sem_batch)

                # print(feats.shape)
                

                # Take feature divergence
                feats = feats - orig_feats
                feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))
                features.append(feats)

            features = torch.cat(features, dim=0)

            # Loss
            acc, loss = self.loss_fn(features)
            loss.backward()

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z_sem.shape[0])
            self.train_loss_metric.update(loss.item(), z_sem.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
            )

            self.writer.add_scalar("Loss/train", loss.item(), iterations)
            self.writer.add_scalar("Acc/train", acc.item(), iterations)

            
            wandb.log({'Loss/train': loss.item()})
            wandb.log({'Acc/train': acc.item()})

   
            iterations += 1

        pbar.close()

    def _train_loop_amp(self, epoch: int, dataloader) -> None:
        """
        Train loop with Automatic Mixed Precision

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Set to eval
        self.generator.eval()

        if self.train_projector:
            self.projector.train()
        else:
            self.projector.eval()

        # Loop
        iterations = len(dataloader) * epoch
        for batch in pbar:
            imgs = batch['img'].to(self.device)
            z_sem = self._encode_and_concatenate_classifier(imgs)
            z_sem_orig = z_sem
            # xT = self.generator.encode_stochastic(imgs, z_sem, T = 50, mode = self.mode)

            # Forward + backward
            self.optimizer.zero_grad()

            # Use amp in forward pass
            with autocast():
                # Original features
                with torch.no_grad():
                    # orig_feats = self.generator.get_features(z)
                    orig_feats = self.projector(z_sem_orig)

                # Apply Directions
                z_sem = self.model(z_sem)

                # Forward
                features = []
                for j in range(z_sem.shape[0] // self.batch_size):
                    # Prepare batch
                    start, end = j * self.batch_size, (j + 1) * self.batch_size
                    z_sem_batch = z_sem[start:end, ...]

                    # Get features
                    # shifted_generated_images = self.generator.render(xT, z_sem_batch, T = 20, mode = self.mode)
                    # # print("Generating")
                    # feats = self._encode_and_concatenate_classifier(shifted_generated_images)
                    feats = self.projector(z_sem_batch)

                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Update optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z_sem.shape[0])
            self.train_loss_metric.update(loss.item(), z_sem.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
            )

            self.writer.add_scalar("Loss/train", loss.item(), iterations)
            self.writer.add_scalar("Acc/train", acc.item(), iterations)

            wandb.log({'Loss/train': loss.item()})
            wandb.log({'Acc/train': acc.item()})

            iterations += 1

        pbar.close()

    def _val_loop(self, epoch: int, dataloader) -> None:
        """
        Standard validation loop

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # Set to eval
        self.model.eval()
        self.generator.eval()
        self.projector.eval()

        # Loop
        iterations = len(dataloader) * epoch
        for batch in pbar:
            with torch.no_grad():
                imgs = batch['img'].to(self.device)
                z_sem = self._encode_and_concatenate_classifier(imgs)
                z_sem_orig = z_sem
                # xT = self.generator.encode_stochastic(imgs, z_sem, T = 50, mode = self.mode)

                # Original features
                # orig_feats = self.generator.get_features(z)
                orig_feats = self.projector(z_sem_orig)

                # Apply Directions
                z_sem = self.model(z_sem)

                # Forward
                features = []
                for j in range(z_sem.shape[0] // self.batch_size):
                    # Prepare batch
                    start, end = j * self.batch_size, (j + 1) * self.batch_size
                    z_sem_batch = z_sem[start:end, ...]

                    # Get features
                    # feats = self.generator.get_features(z[start:end, ...])
                    # shifted_generated_images = self.generator.render(xT, z_sem_batch, T = 20, mode = self.mode)
                    # # print("Generating")
                    # feats = self._encode_and_concatenate_classifier(shifted_generated_images)
                    feats = self.projector(z_sem_batch)
     

                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)
                self.val_acc_metric.update(acc.item(), z_sem.shape[0])
                self.val_loss_metric.update(loss.item(), z_sem.shape[0])

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(
                    f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
                )

                self.writer.add_scalar("Loss/val", loss.item(), iterations)
                self.writer.add_scalar("Acc/val", acc.item(), iterations)

                wandb.log({'Loss/val': loss.item()})
                wandb.log({'Acc/val': acc.item()})

                iterations += 1

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        # if self.writer is not None:
        #     self._write_to_tb(epoch)

        # Save model
        if self.save_path is not None:
            self._save_model(os.path.join(self.save_path, "most_recent.pt"), epoch)

        eval_loss = self.val_loss_metric.compute()
        if self.best_loss == -1 or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self._save_model(os.path.join(self.save_path, "best_model.pt"), epoch)

        # Clear metrics
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()
        self.val_loss_metric.reset()
        self.val_acc_metric.reset()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train acc: {self.train_acc_metric.compute():.3f} "
        s += f"| Train loss: {self.train_loss_metric.compute():.3f} "
        s += f"| Val acc: {self.val_acc_metric.compute():.3f} "
        s += f"| Val loss: {self.val_loss_metric.compute():.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"

        return s

    # def _write_to_tb(self, iteration):
    #     self.writer.add_scalar(
    #         "Loss/train", self.train_loss_metric.compute(), iteration
    #     )
    #     self.writer.add_scalar("Acc/train", self.train_acc_metric.compute(), iteration)
    #     self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), iteration)
    #     self.writer.add_scalar("Acc/val", self.val_acc_metric.compute(), iteration)

    def _save_model(self, path, epoch):
        obj = {
            "epoch": epoch + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "projector": self.projector.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict() if self.mixed_precision else None,
        }
        torch.save(obj, os.path.join(self.save_path, path))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = checkpoint["epoch"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.mixed_precision and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scheduler"])

        # if self.start_epoch > self.iterations:
        #     raise ValueError("Starting iteration is larger than total iterations")

        self.logger.info(
            f"Checkpoint loaded, resuming from epoch {self.start_epoch}"
        )
