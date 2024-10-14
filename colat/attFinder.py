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
import time

from diffae.experiment import LitModel

sign = lambda x: math.copysign(1, x)


class AttFinder:
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
        n_dirs: int,
        alpha: int,
        # iterative: bool = True,
        image_size: Optional[Union[int, List[int]]] = None,
        threshold: float = 0.1,
        M: int = 5, # No. of coordinates to find
        conf = None,
        mode: str = 'non_ema',
        desired_class: str = 'female'
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

        self.threshold = threshold
        self.n_dirs = n_dirs
        self.M = M
        self.alpha = alpha
        self.desired_class = desired_class

        # Set to eval
        self.generator.eval()
        self.projector.eval()
        self.model.eval()

        assert type(n_samples) is int
        self.n_samples = n_samples


        # We need to sample images from the opposite class
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
        random_indices = random.sample(range(self.dataset_length), n_samples)
        # Sample the images using the generated indices
        sampled_images = [self.test_data[idx]['img'] for idx in random_indices]
        # Convert the list of sampled images to a tensor (if necessary)
        self.sampled_images = torch.stack(sampled_images).to(self.device)
             
     



    def run_att_find(self) -> float:
        """
        Runs the AttFind algorithm

        Returns:
            Selected coordinates and directions as lists
        """
        # print("\n\n If it reached here, it should run! Congrats")
        # exit()
        # Progress bar
        pbar = tqdm.tqdm(total = self.M, leave=False)
        pbar.set_description("Running AttFind... ")

        start = time.time()

        # Set to eval
        self.generator.eval()
        self.generator.model.eval()
        self.generator.ema_model.eval()
        self.projector.eval()
        self.model.eval()


        #  Helper function to edit latent codes
        def _edit(z, alpha, k, direction):
  
            self.model.alpha = direction * alpha
            z_sem_modified = self.model.forward_single(z, k = k)
            return z_sem_modified
        
        # Helper function to calculate the change in probability
        def _calculate_prob_change(original_image, modified_image, target_class_index, mode = self.mode):
            if mode == 'ema':
                model = self.generator.ema_model
            else:
                model = self.generator.model

            original_logit = model.classifier_component.classifier(original_image)
            original_prob = torch.softmax(original_logit, dim = -1)[: , target_class_index]
            modified_logit = model.classifier_component.classifier(modified_image)
            modified_prob = torch.softmax(modified_logit, dim = -1)[: , target_class_index]
            
            return modified_prob - original_prob
        
        # Helper function to get the classifier's prediction probability for the target class
        def _get_classifier_probability(image, target_class_index, mode=self.mode):
            """
            Get the classifier's prediction probability for the target class.
            """
            model = self.generator.ema_model if mode == 'ema' else self.generator.model
            logits = model.classifier_component.classifier(image)
            probs = torch.softmax(logits, dim = -1)[: , target_class_index]  
            return probs


        # Helper function to convert image to z_sem
        def _encode_and_concatenate_classifier(imgs):
            w = self.generator.encode(imgs, mode = self.mode)
            if self.mode == 'ema':
                concat_classifier = self.generator.ema_model.classifier_component
            else:
                concat_classifier = self.generator.model.classifier_component
            z_sem = concat_classifier(imgs, w)
            return z_sem
        
 
        # Helper function to generate images
        def _generate(z_sem, xT, T = 20):
            images = self.generator.render(xT, z_sem, T = T, mode = self.mode)
            return images
        
        

        def _visualize_images(originals, mods, orig_probs, mod_probs, best_coordinate, best_direction, num):
            fig, axs = plt.subplots(num, 2, figsize=(10, 25))  # 5 rows for the top 5 images, 2 columns (original and modified)

            # Convert tensors to numpy arrays for plotting
            def tensor_to_np(tensor_image):
                img = tensor_image.squeeze().detach().cpu()
                return img.permute(1, 2, 0).numpy()  # CHW to HWC format

            if best_direction == 1:
                dir = "Positive"
            elif best_direction == -1:
                dir = "Negative"

            # Iterate through the top 5 images and plot them
            for i in range(num):
                # Convert tensor images for display
                orig_np = tensor_to_np(originals[i])
                mod_np = tensor_to_np(mods[i])

                # Plot original image
                axs[i, 0].imshow(orig_np)
                axs[i, 0].set_title(f"Original\nProb: {orig_probs[i]:.2f}")
                axs[i, 0].axis('off')

                # Plot modified (shifted) image
                axs[i, 1].imshow(mod_np)
                axs[i, 1].set_title(f"Shift\nProb: {mod_probs[i]:.2f}")
                axs[i, 1].axis('off')

            # Set the overall title for the figure with the best coordinate and direction
            fig.suptitle(f"Coordinate: {best_coordinate} along {dir} direction", fontsize=16)

            # Adjust layout to make space for the title
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save the figure
            plt.savefig(f"{k}_coord_{best_coordinate}.png")
            plt.close()
        

        
 
        #Loop
        selected_coordinates = [] # Best coordinates will be saved here
        selected_directions = [] # Best directions (either postivie (+1), or negative (-1)) will be saved here
        batch_size = 256

        
        used_coordinates = set()
        remaining_images = self.sampled_images.clone()

        k = 0 # To keep a track of the number of iterations
        with torch.no_grad():

            while len(selected_coordinates) < self.M:
                
                max_effect = 0
                best_coordinate = None
                best_direction = None

                best_sample_idxs = []  # Store indices of the top 5 most affected images
                # top5_effects = []  # Store the top 5 effects
                
                print("\nNumber of remaining images: ", remaining_images.shape[0])
                # Process all images in batches to speed up the loop
                for i in tqdm.tqdm(range(0, remaining_images.shape[0], batch_size), leave=False):
                    image_batch = remaining_images[i:i + batch_size]

                    # Encode once for the batch
                    z_sem_batch = _encode_and_concatenate_classifier(image_batch)
                    xT_batch = self.generator.encode_stochastic(image_batch, z_sem_batch, T = 50, mode = self.mode)


                    for z_sem_index in tqdm.tqdm(range(self.n_dirs)):

                        if z_sem_index in used_coordinates:  # Skip this coordinate if it has already been chosen
                            continue
                        
                        pos_z_sem_batch = _edit(z_sem_batch, self.alpha, k = z_sem_index, direction = 1)
                        neg_z_sem_batch = _edit(z_sem_batch, self.alpha, k = z_sem_index, direction = -1)

                    # Generate images in batch
                        pos_images = _generate(pos_z_sem_batch, xT_batch)
                        neg_images = _generate(neg_z_sem_batch, xT_batch)

                        # Calculate changes in logits for the target class for the entire batch
                        pos_effects = _calculate_prob_change(image_batch, pos_images, 
                                                              target_class_index = self.target_class_index)
                        neg_effects = _calculate_prob_change(image_batch, neg_images, target_class_index = 
                                                              self.target_class_index)

                        # Average effects across the batch
                        avg_pos_effect = pos_effects.mean().item()
                        avg_neg_effect = neg_effects.mean().item()


                        # Select best coordinate and direction
                        if avg_pos_effect > max_effect:
                            # print("Positive effect is greater")
                            max_effect = avg_pos_effect
                            best_coordinate = z_sem_index
                            best_direction = 1
                            best_sample_idxs = torch.argsort(pos_effects, descending=True)[:5].tolist()  # Get top 5 indices for positive shift

                        if avg_neg_effect > max_effect:
                            # print("Negative effect is greater")
                            max_effect = avg_neg_effect
                            best_coordinate = z_sem_index
                            best_direction = -1
                            best_sample_idxs = torch.argsort(neg_effects, descending=True)[:5].tolist()  # Get top 5 indices for negative shift

                    
                    print(f"\n\nMax_effect in {k}: {max_effect}")
                    # Append the best coordinate and direction to the selected lists
                    if max_effect > self.threshold:
                        selected_coordinates.append(best_coordinate)
                        selected_directions.append(best_direction)
                        used_coordinates.add(best_coordinate)

                        print(f"\n\nFound best coordinate {best_coordinate} and direction {best_direction}")
                        print("Visualizing the top 5 most affected images...")


                         # Generate and visualize the image with the largest effect
                        original_images = remaining_images[best_sample_idxs]  # The most affected image
                        z_sem = _encode_and_concatenate_classifier(original_images)
                        xT = self.generator.encode_stochastic(original_images, z_sem, T = 50, mode = self.mode)

                        # Generate positive and negative shifts
                        modified_z_sem = _edit(z_sem, self.alpha, k = best_coordinate, direction = best_direction)
                        mod_imgs = _generate(modified_z_sem, xT)

                        # Get classifier probabilities for each image
                        orig_prob = _get_classifier_probability(original_images, 
                                                                target_class_index = self.target_class_index)
                        mod_prob = _get_classifier_probability(mod_imgs, 
                                                               target_class_index = self.target_class_index)
                        
                        original_images = (original_images + 1)/2
                        # Visualize and save the images using subplots
                        _visualize_images(original_images, mod_imgs, 
                                          orig_probs = orig_prob, mod_probs = mod_prob, 
                                          best_coordinate = best_coordinate, best_direction = best_direction, num = len(best_sample_idxs))
                        # Update the remaining images by removing explained ones in batch
                        new_remaining_images = []

                        
                        for i in tqdm.tqdm(range(0, remaining_images.shape[0], batch_size), leave=False):
                            img_batch = remaining_images[i:i + batch_size]
                            z_sem_batch = _encode_and_concatenate_classifier(img_batch)
                            xT_batch = self.generator.encode_stochastic(img_batch, z_sem_batch, T = 50, mode = self.mode)

                            modified_z_sem_batch = _edit(z_sem_batch, self.alpha, k = best_coordinate, direction = best_direction)
                            modified_images = _generate(modified_z_sem_batch, xT_batch)

                            # Check the logit change to decide if the images should remain
                            keep_images = _calculate_prob_change(img_batch, modified_images, target_class_index=0) < max_effect
                            print(keep_images)
                            new_remaining_images.append(img_batch[keep_images])

                        remaining_images = torch.cat(new_remaining_images, dim=0) if new_remaining_images else torch.empty((0, *remaining_images.shape[1:]))
                        
                self.threshold -= 0.02 # Decrease the threshold after each iteration
                self.threshold = min(self.threshold, max_effect) # Ensure the threshold is not greater than the max effect
                pbar.update()
                k += 1
                    
            pbar.close()
        print(f"Best coordinates: {selected_coordinates}")
        print(f"Corresponding directions: {selected_directions}")
        end = time.time()
        print(f"Time taken: {((end - start) / 3600):.2f}h")
                
            





