import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import wandb
# saranga: use your wandb key here
wandb.login(key = "280a63fbe206439a036945bcecd7d1f619763c7d")


from colat.evaluator import Evaluator
from colat.projectors import Projector
from colat.trainer import Trainer
from colat.visualizer import Visualizer
from colat.attFinder import AttFinder
from colat.animator import Animator

from diffae.experiment import LitModel
from diffae.templates import bbc_autoenc


def train(cfg: DictConfig) -> None:
    """Trains model from config

    Args:
        cfg: Hydra config

    """
    # Device
    device = get_device(cfg)

    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss, k=cfg.k).to(device)
    projector: Projector = instantiate(cfg.projector).to(device)

    # DiffAE encoder as generator
    conf = bbc_autoenc()
    conf.batch_size = cfg.hparams.batch_size
    conf.include_classifier = False
    # conf.name = 'ffhq128_autoenc_w_classifier2'

    generator = LitModel(conf)
    
    diffae_ckpt_path = cfg.diffae_ckpt_path
    state = torch.load(diffae_ckpt_path, map_location='cpu')
    generator.load_state_dict(state['state_dict'])
    generator.to(device)


    mode = 'ema' if cfg.use_ema else 'non_ema'
    # generator = generator.ema_model if cfg.use_ema else generator.model
    generator.eval()
    

    # Optimizer and scheduler
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer,
        list(model.parameters()) + list(projector.parameters())
        if cfg.train_projector
        else model.parameters(),
    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)

    # Paths
    save_path = os.getcwd() if cfg.save else None
    checkpoint_path = (
        hydra.utils.to_absolute_path(cfg.checkpoint)
        if cfg.checkpoint is not None
        else None
    )

    # Tensorboard
    if cfg.tensorboard:
        # Note: global step is in epochs here
        writer = SummaryWriter(os.getcwd())
        # Indicate to TensorBoard that the text is pre-formatted
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)

        # saranga: use your own project name and entity
        wandb.init(project = "DiffaeCLR_Bio__w_Classifier", entity = "saranga7")
        # wandb.init(project="DiffaeCLR", config=cfg)
        # wandb.config.update(cfg)  # Add the configuration details to wandb
        # Log the entire configuration in a neat format
        wandb.log({"config": OmegaConf.to_yaml(cfg)})
    else:
        writer = None

    # Trainer init
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        generator=generator,
        projector=projector,
        batch_size=cfg.hparams.batch_size,
        # iterations=cfg.hparams.iterations,
        device=device,
        eval_freq=cfg.eval_freq,
        eval_iters=cfg.eval_iters,
        scheduler=scheduler,
        grad_clip_max_norm=cfg.hparams.grad_clip_max_norm,
        writer=writer,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        mixed_precision=cfg.mixed_precision,
        train_projector=cfg.train_projector,
        feed_layers=cfg.feed_layers,
        epochs = cfg.hparams.epochs,

        conf = conf,
        mode = mode
    )

    # print("Things are working fine")
    # print(cfg.use_ema)
    # exit()

    # Launch training process
    trainer.train()


def evaluate(cfg: DictConfig) -> None:
    """Evaluates model from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    # Model
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    projector.load_state_dict(checkpoint["projector"])

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        generator=generator,
        projector=projector,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        feed_layers=cfg.feed_layers,
    )
    evaluator.evaluate()


def generate(cfg: DictConfig) -> None:
    """Generates images from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    # Model
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    # generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    # DiffAE encoder as generator
    conf = bbc_autoenc()
    conf.batch_size = cfg.hparams.batch_size
    conf.include_classifier = False
    # conf.name = 'ffhq128_autoenc_w_classifier2'

    generator = LitModel(conf)
    diffae_ckpt_path = cfg.diffae_ckpt_path
    state = torch.load(diffae_ckpt_path, map_location='cpu')
    generator.load_state_dict(state['state_dict'])


    generator.to(device)


    mode = 'ema' if cfg.use_ema else 'non_ema'
    # generator = generator.ema_model if cfg.use_ema else generator.model
    generator.eval()


    projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    projector.load_state_dict(checkpoint["projector"])

    visualizer = Visualizer(
        model=model,
        generator=generator,
        projector=projector,
        device=device,
        n_samples=cfg.n_samples,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,

        conf = conf,
        mode = mode
    )
    visualizer.visualize()


def att_find(cfg: DictConfig) -> None:
    """Runs the attribute finder

    Args:
        cfg: Hydra config
    """

    device = get_device(cfg)

    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    conf = bbc_autoenc()
    conf.batch_size = cfg.hparams.batch_size
    conf.include_classifier = False
    # conf.name = 'ffhq128_autoenc_w_classifier2'
    conf.data_name = 'bbc021_simple_selective'

    generator = LitModel(conf)
    diffae_ckpt_path = cfg.diffae_ckpt_path
    state = torch.load(diffae_ckpt_path, map_location='cpu')
    generator.load_state_dict(state['state_dict'])
    generator.to(device)


    mode = 'ema' if cfg.use_ema else 'non_ema'
    # generator = generator.ema_model if cfg.use_ema else generator.model
    generator.eval()

    projector : torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    projector.load_state_dict(checkpoint["projector"])

    att_finder = AttFinder(
        model=model,
        generator=generator,
        projector=projector,
        device=device,
        n_samples=cfg.n_samples,
        n_dirs=cfg.k,
        alpha=cfg.alpha,
        image_size=cfg.image_size,
        threshold=cfg.threshold,
        M=cfg.M,
        conf = conf,
        mode = mode,
        desired_class = str(cfg.desired_class)
    )

    att_finder.run_att_find()


def animate(cfg: DictConfig) -> None:
    """Generates images from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    # Model
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    # generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    # DiffAE encoder as generator
    conf = bbc_autoenc()
    conf.batch_size = cfg.hparams.batch_size
    conf.include_classifier = False
    conf.data_name = 'bbc021_simple_selective'

    generator = LitModel(conf)
    diffae_ckpt_path = cfg.diffae_ckpt_path
    state = torch.load(diffae_ckpt_path, map_location='cpu')
    generator.load_state_dict(state['state_dict'])


    generator.to(device)


    mode = 'ema' if cfg.use_ema else 'non_ema'
    # generator = generator.ema_model if cfg.use_ema else generator.model
    generator.eval()


    projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    projector.load_state_dict(checkpoint["projector"])

    animator = Animator(
        model=model,
        generator=generator,
        projector=projector,
        device=device,
        n_samples=cfg.n_samples,
        direction=cfg.direction,
        max_alpha=cfg.max_alpha,
        alphas_step_size = cfg.alphas_step_size,
        iterative=cfg.iterative,
        image_size=cfg.image_size,
        conf = conf,
        mode = mode,
        desired_class = str(cfg.desired_class)
    )
    animator.animate()


def get_device(cfg: DictConfig) -> torch.device:
    """Initializes the device from config

    Args:
        cfg: Hydra config

    Returns:
        device on which the model will be trained or evaluated

    """
    if cfg.auto_cpu_if_no_gpu:
        device = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    return device



