from model import EfficientNetHeatmap
import dataset
import utils
import train_utils

import os
import tqdm.auto as tqdm
import wandb

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

def main():
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # текущая рабочая директория
    train_img_dir = os.path.join(BASE_DIR, "hand-keypoints/images")
    train_lbl_dir = os.path.join(BASE_DIR, "hand-keypoints/labels")

    val_img_dir = os.path.join(BASE_DIR, "hand-keypoints/images")
    val_lbl_dir = os.path.join(BASE_DIR, "hand-keypoints/labels")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = dataset.HandPointDataset(train_img_dir, train_lbl_dir, split="train", transform=transform)
    val_dataset = dataset.HandPointDataset(val_img_dir, val_lbl_dir, split="val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=5)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=5)

    #image, heatmaps = next(iter(val_loader))
    #utils.visualize_heatmap(image, heatmaps)

    config = {
        "project": "heatmap_keypoints_detector",
        "experiment": "efficientnet-b0_finetune",
        "epochs": 10,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "scheduler": "CosineAnnealingLR",
        "model": "efficientnet-bo + decoder",
        "num_keypoints": 21,
        "device": 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu'),
    }

    model = EfficientNetHeatmap(out_channels=config["num_keypoints"])
    model = model.to(config["device"], non_blocking=True)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"] * len(train_loader))

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

    # # Проверьте распределение данных
    # print(f"Train size: {len(train_dataset)}")
    # print(f"Val size: {len(val_dataset)}")
    #
    # # Проверьте один пример из каждого датасета
    # train_img, train_heatmap = train_dataset[0]
    # val_img, val_heatmap = val_dataset[0]
    #
    # print(f"Train image range: [{train_img.min():.3f}, {train_img.max():.3f}]")
    # print(f"Train heatmap range: [{train_heatmap.min():.3f}, {train_heatmap.max():.3f}]")
    # print(f"Val image range: [{val_img.min():.3f}, {val_img.max():.3f}]")
    # print(f"Val heatmap range: [{val_heatmap.min():.3f}, {val_heatmap.max():.3f}]")

if __name__ == "__main__":
    main()