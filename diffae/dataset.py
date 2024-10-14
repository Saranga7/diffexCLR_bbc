import os
import glob
from pathlib import Path

import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class BBCDataset(Dataset):
    def __init__(self, path,
                img_size = 128, 
                split = None,
                as_tensor: bool = True,
                do_augment: bool = True,
                do_normalize: bool = True):
    
        self.path = path
        # Store image paths
        if split:
            data_path = glob.glob(os.path.join(self.path, split, "*/*"))
        else:
            data_path = glob.glob(os.path.join(self.path, "*/*/*"))
        
        self.data = [path for path in data_path]
   
        # Image transformation
        transform = [
            transforms.Resize(img_size),
        ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                )
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        assert index < len(self.data)
        img_path = self.data[index]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'index': index}
    



class Selective_BBC_Dataset(Dataset):
    def __init__(self, path, img_size=128, split = None, **kwargs):
        self.path = path
        self.desired_class = kwargs.get("desired_class", "DMSO") # Store desired class

        # Store image paths
        if split:
            data_path = glob.glob(os.path.join(self.path, split, "*/*"))
        else:
            data_path = glob.glob(os.path.join(self.path, "*/*/*"))
     

        # Filter data based on the desired class (if specified)
        self.data = []
        for img_path in data_path:
            class_name = img_path.split(os.sep)[-2]
            if self.desired_class is None or class_name == self.desired_class:
                self.data.append([img_path, class_name])

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, _class_name = self.data[index]
        _img = self.transform(Image.open(img_path))
        return {"img": _img, "index": index, "class": _class_name}
    


if __name__ == "__main__":
    data_path = "/projects/deepdevpath/Anis/diffusion-comparison-experiments/datasets/bbc021_simple"

    dataset = Selective_BBC_Dataset(data_path, desired_class = "latrunculin_B_high_conc")
    # print(len(dataset))
    i = random.randint(0, len(dataset))
    print(dataset[i]['img'].shape)
    print(dataset[i]['index'])
    print(dataset[i]['class'])


# DMSO
# latrunculin_B_high_conc