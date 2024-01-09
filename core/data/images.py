import torch.nn as nn
from torchvision.datasets.vision import VisionDataset
import os
import torch
import pdb
import torchvision
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class Images(VisionDataset):
    def __init__(self, root, size, transform=None, target_transform=None):
        super(Images, self).__init__(root, transform=transform, target_transform=target_transform)
        self.image_path_list = root
        self.size = size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


    def __getitem__(self, item):
        img = Image.open(self.image_path_list[item]).convert("RGB")
        img = self.transform(img)
        img = img.reshape(-1)
        return img

    def __len__(self) -> int:
        return len(self.image_path_list)