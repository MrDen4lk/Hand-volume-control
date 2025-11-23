import os
import warnings
warnings.filterwarnings("ignore")

import wandb
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset
from model import HandPoseUNet
import train_utils

def main():
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
    train_img_dir = os.path.join(BASE_DIR, "hand-keypoints/images")
    train_lbl_dir = os.path.join(BASE_DIR, "hand-keypoints/labels")

    val_img_dir = os.path.join(BASE_DIR, "hand-keypoints/images")
    val_lbl_dir = os.path.join(BASE_DIR, "hand-keypoints/labels")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    train_dataset = dataset.HandPointDataset(train_img_dir, train_lbl_dir, split="train", transform=train_transform, downsample=4)
    val_dataset = dataset.HandPointDataset(val_img_dir, val_lbl_dir, split="val", transform=val_transform, downsample=4)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    # image, heatmaps = next(iter(train_loader))
    # utils.visualize_heatmap(image, heatmaps)

    config = {
        "project": "heatmap_keypoints_detector",
        "experiment": "convnext_small&unet",
        "epochs": 10,
        "lr": 3e-4,
        "weight_decay" : 1e-2,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "scheduler": "CosineAnnealingWarmRestarts",
        "model": "convnext_small",
        "num_keypoints": 21,
        "device": 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu'),
    }

    model = HandPoseUNet(out_channels=config["num_keypoints"])
    model = model.to(config["device"], non_blocking=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,
        T_mult=2,
        eta_min=1e-6
    )

    # ОБУЧЕНИЕ

    wandb.init(project=config["project"], config=config)
    train_utils.train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        n_epoch=config["epochs"]
    )

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()