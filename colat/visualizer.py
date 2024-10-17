import logging
import math
import random
from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import tqdm
from PIL import Image, ImageDraw, ImageFont
from omegaconf.listconfig import ListConfig

# from colat.generators import Generator
from diffae.experiment import LitModel

sign = lambda x: math.copysign(1, x)


class Visualizer:
    """Model evaluator

    Args:
        model: model to be evaluated
        generator: pretrained generator
        projector: pretrained projector
        device: device on which to evaluate model
        n_samples: number of samples
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generator: LitModel,
        projector: torch.nn.Module,
        device: torch.device,
        n_samples: Union[int, str],
        n_dirs: Union[int, List[int]],
        alphas: List[int],
        iterative: bool = True,
        feed_layers: Optional[List[int]] = None,
        image_size: Optional[Union[int, List[int]]] = None,
        conf = None,
        mode: str = 'non_ema'
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Diffae conf
        self.conf = conf
        self.mode = mode

        # Device
        self.device = device

        # Model
        self.model = model
        self.generator = generator
        self.projector = projector

        # Set to eval
        self.generator.eval()
        self.projector.eval()
        self.model.eval()

        assert type(n_samples) is int

        # N Samples
        # self.samples = self.generator.sample_latent(n_samples)
        # self.samples = self.samples.to(self.device)

        test_data = self.conf.make_dataset(split = 'test')
        # Get the total number of samples in the dataset
        dataset_length = len(test_data)
        # Ensure n_samples does not exceed the dataset size
        n_samples = min(n_samples, dataset_length)
        # Generate n_samples random indices
        random_indices = random.sample(range(dataset_length), n_samples)
        # Sample the images using the generated indices
        sampled_images = [test_data[idx]['img'] for idx in random_indices]
        # Convert the list of sampled images to a tensor (if necessary)
        self.sampled_images = torch.stack(sampled_images).to(self.device)
        # print(self.sampled_images.shape)

        # exit()


        #  Sub-sample Dirs
        if n_dirs == -1:
            self.dirs = list(range(self.model.k))
        elif isinstance(n_dirs, int):
            self.dirs = np.random.choice(self.model.k, n_dirs, replace=False)
        else:
            assert isinstance(n_dirs, ListConfig)
            self.dirs = n_dirs

        # Alpha
        alphas = sorted(alphas)
        i = 0
        while alphas[i] < 0:
            i += 1
        self.neg_alphas = alphas[:i]

        if alphas[i] == 0:
            i += 1
        self.pos_alphas = alphas[i:]

        # Iterative
        self.iterative = iterative

        # Image Size
        if image_size:
            self.image_transform = T.Resize(image_size)
        else:
            self.image_transform = torch.nn.Identity()

        # Feed Layers
        self.feed_layers = feed_layers

    def visualize(self) -> float:
        """Generates images from the trained model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """
        # Progress bar
        pbar = tqdm.tqdm(total=self.sampled_images.shape[0], leave=False)
        pbar.set_description("Generating... ")

        # Set to eval
        self.generator.eval()
        self.generator.model.eval()
        self.generator.ema_model.eval()
        self.projector.eval()
        self.model.eval()

        #  Helper function to edit latent codes
        def _edit(z, alpha, ks):
            #  check if only one latent code is given
            assert z.shape[0] == 1 or z.shape[0] == len(
                ks
            ), """Only able to apply all directions to single latent code or
                apply each direction to single code"""
            self.model.alpha = alpha

            # Apply Directions
            zs = []
            for i, k in enumerate(ks):
                _i = i if z.shape[0] > 1 else 0
                zs.append(self.model.forward_single(z[_i : _i + 1, ...], k=k))
            zs = torch.cat(zs, dim=0)
            return zs
        
        def _encode_and_concatenate_classifier(imgs):
            w = self.generator.encode(imgs, mode = self.mode)
            if self.mode == 'ema':
                concat_classifier = self.generator.ema_model.classifier_component
            else:
                concat_classifier = self.generator.model.classifier_component
            z_sem = concat_classifier(imgs, w)
            return z_sem
        
        def _classify(imgs, mode = self.mode):
            if mode == 'ema':
                model = self.generator.ema_model
            else:
                model = self.generator.model

            return torch.softmax(model.classifier_component.classifier(imgs), dim=1)


        def _generate(z_sem, xT, T = 20):
            images = self.generator.render(xT, z_sem, T = T, mode = self.mode)
            return images
        
        # plt.style.use('seaborn-poster')
        plt.rcParams.update({'font.size': 25})

        #Loop
        with torch.no_grad():
            for i in range(self.sampled_images.shape[0]):

                # fig_width = (len(self.neg_alphas) + len(self.pos_alphas) + 1) * 3  # 3 inches per column
                # fig_height = len(self.dirs) * 2  # 2 inches per row
                # fig, ax = plt.subplots(len(self.dirs), len(self.neg_alphas) + len(self.pos_alphas) + 1, figsize=(fig_width, fig_height))
                # plt.subplots_adjust(hspace=0.5) 

                # Adjust subplot dimensions and spacing
                fig_width = (len(self.neg_alphas) + len(self.pos_alphas) + 1) * 5  # Increase width per subplot
                fig_height = len(self.dirs) * 5  # Increase height per subplot

                # Create subplots with larger size and adjust spacing
                fig, ax = plt.subplots(len(self.dirs), len(self.neg_alphas) + len(self.pos_alphas) + 1, figsize=(fig_width, fig_height))

                # # Adjust spacing: less space between columns, normal space between rows
                # fig.subplots_adjust(wspace=0.2, hspace=0.4)  # Adjust wspace for narrower column gaps, hspace for row spacing

                # Take a single sample
                orj_img = self.sampled_images[i : i + 1, ...]
                # print(orj_img.shape)
                z_sem = _encode_and_concatenate_classifier(orj_img)
                # print(z_sem.shape)
 
                print("Encoding noise map of the original image")
                xT = self.generator.encode_stochastic(orj_img, z_sem, T = 50, mode = self.mode)
                classifier_output = _classify(orj_img)
                formatted_output = [f"{x:.2f}" for x in classifier_output.cpu().detach().numpy().flatten()]

                orj_img = (orj_img + 1)/2
                # orj_img = orj_img.detach().cpu()

                for idx in range(len(self.dirs)):
                    ax[idx][len(self.neg_alphas)].imshow(orj_img[0].permute(1, 2, 0).cpu())
                    ax[idx][len(self.neg_alphas)].set_title(f"Original\n{formatted_output}", weight = "bold")
                    # ax[idx][len(self.neg_alphas)].set_title(f"{formatted_output}" , fontsize=37, weight = "bold")
                    ax[idx][len(self.neg_alphas)].axis('off')


                # images = []
                # classifier_outputs = []

                #  Start with z and alpha = 0
                z_sem_orig = z_sem
                prev_alpha = 0
                for idx, alpha in enumerate(self.neg_alphas):
                    #  if iterative use last z and d(alpha)
                    _z = z_sem if self.iterative else z_sem_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha
                    
                    z_sem = _edit(_z, _alpha, ks=self.dirs)

                    xT_repeated = xT.repeat(z_sem.shape[0], 1, 1, 1)
                    # print("Generating images...")
                    generated_images = _generate(z_sem, xT_repeated)
                    
                    c_output = _classify(generated_images)
                 
                    formatted_output = []
                    for c_ops in c_output:
                        formatted_output.append([f"{x:.2f}" for x in c_ops.cpu().detach().numpy()])
                 
                    # classifier_outputs.append(c_output)
                    # images.append(generated_images.detach().cpu())

                    for k in range(len(self.dirs)):
                        ax[k][idx].imshow(generated_images[k].permute(1, 2, 0).cpu())
                        ax[k][idx].set_title(f"α={alpha:.1f}\n{formatted_output[k]}", weight = "bold")
                        # ax[k][idx].set_title(f"{formatted_output[k]}", fontsize=37, weight = "bold")
                        ax[k][idx].axis('off')

                    prev_alpha = alpha



                # Reset z and alpha
                z_sem = z_sem_orig
                prev_alpha = 0
                for idx, alpha in enumerate(self.pos_alphas):
                    #  if iterative use last z and d(alpha)
                    _z = z_sem if self.iterative else z_sem_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z_sem = _edit(_z, _alpha, ks=self.dirs)
                    generated_images = _generate(z_sem, xT_repeated)
                    c_output = _classify(generated_images)
                    formatted_output = []
                    for c_ops in c_output:
                        formatted_output.append([f"{x:.2f}" for x in c_ops.cpu().detach().numpy()])
                    # classifier_outputs.append(c_output)
                    # images.append(generated_images.detach().cpu())

                    for k in range(len(self.dirs)):
                        ax[k][idx + 1 + len(self.neg_alphas)].imshow(generated_images[k].permute(1, 2, 0).cpu())
                        ax[k][idx + 1 + len(self.neg_alphas)].set_title(f"α={alpha:.1f}\n{formatted_output[k]}", weight = "bold")
                        # ax[k][idx + 1 + len(self.neg_alphas)].set_title(f"{formatted_output[k]}", fontsize=37, weight = "bold")
                        ax[k][idx + 1 + len(self.neg_alphas)].axis('off')

                    prev_alpha = alpha

                for idx, k in enumerate(self.dirs):
                    fig.text(0.05, 1 - (idx + 0.5) / len(self.dirs), f"k = {k}", fontsize=42, va='center', ha='right', weight = "bold")

                
                # Adjust spacing between subplots
                fig.subplots_adjust(wspace=0.2, hspace=0.5) 
                plt.tight_layout(rect=[0.1, 0, 1, 1])  # Leave some space for the labels on the left
                plt.savefig(f'sample_{i}.png')
                pbar.update()
 

        pbar.close()
