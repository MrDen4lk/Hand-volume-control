from utils import generate_heatmaps

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
class HandPointDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, split='train', transform=None, sigma=2, downsample=4):
        self.img_dir = os.path.join(img_dir, split)
        self.lbl_dir = os.path.join(lbl_dir, split)
        self.sigma = sigma
        self.downsample = downsample
        self.transform = transform

        self.image_files = sorted(os.listdir(self.img_dir))
        self.label_files = sorted(os.listdir(self.lbl_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        base_name, _ = os.path.splitext(img_name)
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")

        with open(lbl_path, "r") as f:
            parts = list(map(float, f.read().strip().split()))[5:]

        keypoints = torch.tensor(parts, dtype=torch.float32).view(-1, 3)

        if self.transform:
            image = self.transform(image)

        heatmaps = generate_heatmaps(keypoints, 224, 224, sigma=self.sigma, downsample=self.downsample)

        return image, heatmaps