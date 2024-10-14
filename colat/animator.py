import os
import matplotlib.animation as animation

import logging
import math
import random
from typing import List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T
import tqdm
from PIL import Image, ImageDraw, ImageFont
from omegaconf.listconfig import ListConfig

# from colat.generators import Generator
from diffae.experiment import LitModel

class Animator:
    def __init__(self, 
                 model: torch.nn.Module, 
                 generator: LitModel, 
                 projector: torch.nn.Module, 
                 device: torch.device, 
                 n_samples: Union[int, str], 
                 direction: int, 
                 max_alpha: float = 5, 
                 iterative: bool = True, 
                 image_size: Optional[Union[int, List[int]]] = None, 
                 alphas_step_size: float = 0.5,
                 conf=None, 
                 mode: str = 'non_ema',
                 desired_class: str = 'female') -> None:
        
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

        self.desired_class = desired_class

        # N Samples
        # self.samples = self.generator.sample_latent(n_samples)
        # self.samples = self.samples.to(self.device)

        if self.desired_class == 'DMSO':
            opposite_class = "latrunculin_B_high_conc"
            self.target_class_index = 0

        elif self.desired_class == "latrunculin_B_high_conc":
            opposite_class = "DMSO"
            self.target_class_index = 1

        self.test_data = self.conf.make_dataset(desired_class = opposite_class)
        self.dataset_length = len(self.test_data)
        print(f"Dataset length: {self.dataset_length}")
        # Generate n_samples random indices
        random.seed(777) 
        random_indices = random.sample(range(self.dataset_length), n_samples)
        # Sample the images using the generated indices
        sampled_images = [self.test_data[idx]['img'] for idx in random_indices]
        # Convert the list of sampled images to a tensor (if necessary)
        self.sampled_images = torch.stack(sampled_images).to(self.device)



        # #  Sub-sample Dirs
        # if n_dirs == -1:
        #     self.dirs = list(range(self.model.k))
        # elif isinstance(n_dirs, int):
        #     self.dirs = np.random.choice(self.model.k, n_dirs, replace=False)
        # else:
        #     assert isinstance(n_dirs, ListConfig)
        #     self.dirs = n_dirs

        # Alpha
        # alphas = sorted(alphas)
        # i = 0
        # while alphas[i] < 0:
        #     i += 1
        # self.neg_alphas = alphas[:i]

        # if alphas[i] == 0:
        #     i += 1
        # self.pos_alphas = alphas[i:]

        # Iterative
        self.iterative = iterative

        # Image Size
        if image_size:
            self.image_transform = T.Resize(image_size)
        else:
            self.image_transform = torch.nn.Identity()

        # create a range of alphas
        self.alphas = [round(alphas_step_size * i, 2) for i in range(1, int(abs(max_alpha) / alphas_step_size) + 1)]
        self.alphas = [-alpha if max_alpha < 0 else alpha for alpha in self.alphas]

        print("\nAlphas", self.alphas)
        print()

        self.direction = direction

        

    def animate(self) -> None:
        """Generates images from the trained model, saves individual images, and creates an animation."""
        pbar = tqdm.tqdm(total=self.sampled_images.shape[0], leave=False)
        pbar.set_description("Animating... ")

        plt.rcParams.update({'font.size': 25})

        def _edit(z, alpha, k):
            self.model.alpha = alpha

            z_sem_modified = self.model.forward_single(z, k = k)
            return z_sem_modified
            
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

            return torch.softmax(model.classifier_component.classifier(imgs), dim = -1)[: , self.target_class_index]


        def _generate(z_sem, xT, T = 20):
            images = self.generator.render(xT, z_sem, T = T, mode = self.mode)
            return images
        
        # Loop over samples
        for i in range(self.sampled_images.shape[0]):
            print("Animating sample", i)
            orj_img = self.sampled_images[i : i + 1, ...]
            z_sem = _encode_and_concatenate_classifier(orj_img)
            xT = self.generator.encode_stochastic(orj_img, z_sem, T = 50, mode = self.mode)
            c_output = _classify(orj_img)
            # formatted_output = [f"{x:.2f}" for x in c_output[0].cpu().detach().numpy()]

            fig, ax = plt.subplots(1, figsize=(6, 6))  # Create a figure for animation
            ims = []  # List to store images for animation

            orj_img = (orj_img + 1)/2
            text = ax.text(0.01, 1, f'Alpha: {0.0}\n{self.desired_class} prob: {c_output.item():.2f}', 
                               transform=ax.transAxes, color='white', fontsize=12, 
                               bbox=dict(facecolor='black', alpha=0.7))
            ims.append([ax.imshow(orj_img[0].permute(1, 2, 0).cpu(), animated=True), text])  # Add original image to the list

            for alpha in self.alphas:
                print("Shifting with alpha", alpha)
                # Apply direction edits and generate image
                z_sem_edited = _edit(z_sem, alpha, k = self.direction)
                xT_repeated = xT.repeat(z_sem_edited.shape[0], 1, 1, 1)
                generated_images = _generate(z_sem_edited, xT_repeated)

                # Classify generated images
                c_output = _classify(generated_images)
                # formatted_output = [f"{x:.2f}" for x in c_output[0].cpu().detach().numpy()]

                # Display the generated image in the plot for animation
                im = ax.imshow(generated_images[0].permute(1, 2, 0).cpu(), animated=True)
                
                # Add alpha and classifier output text to the image
                text = ax.text(0.01, 1, f'Alpha: {alpha:.2f}\n{self.desired_class} prob: {c_output.item():.2f}', 
                               transform=ax.transAxes, color='white', fontsize=12, 
                               bbox=dict(facecolor='black', alpha=0.7))

                ims.append([im, text])  # Add each frame and text to the list

                ax.axis('off')  # Hide axes for a cleaner visualization

                # Save individual image
                # img_filename = f"sample_{i}_alpha_{alpha:.2f}.png"
                # plt.imsave(img_filename, generated_images[0].permute(1, 2, 0).cpu().numpy())
            
            # Create animation object using matplotlib's animation functionality
            ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)

            # Save the animation as a gif or mp4
            ani.save(f'sample_{i}_animation.gif', writer='imagemagick')  # Save as GIF
            # ani.save(f'sample_{i}_animation.mp4', writer='ffmpeg')  # Uncomment to save as mp4

            plt.close(fig)  # Close the figure to avoid displaying static images

            pbar.update()

        pbar.close()


        
