import torch
import tqdm.auto as tqdm
import wandb
import numpy as np

def NCC(logit, target): # метрика качества heatmap
    logit = logit - logit.mean(dim=[2, 3], keepdim=True)
    target = target - target.mean(dim=[2, 3], keepdim=True)
    numerator = (logit * target).sum(dim=[2, 3])
    denominator = torch.sqrt((logit ** 2).sum(dim=[2, 3]) * (target ** 2).sum(dim=[2, 3]) + 1e-6)
    score = numerator / denominator  # [B, K]
    return score.mean()

def train_epoch(model, optimizer, criterion, device, train_loader, scheduler=None):
    loss_log, ncc_log = [], []

    model.train()
    for batch_num, (batch_image, batch_heatmaps) in enumerate(tqdm.tqdm(train_loader, desc="Training Epoch")):
        batch_image = batch_image.to(device, non_blocking=True)
        batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(batch_image)

        loss = criterion(logits, batch_heatmaps)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_log.append(loss.item())
        ncc_value = NCC(logits, batch_heatmaps)
        ncc_log.append(ncc_value)

        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_ncc": ncc_value,
            "train/batch_num": batch_num
        })

    avg_loss = np.mean(loss_log)
    avg_ncc = np.mean(ncc_log)

    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/epoch_ncc": avg_ncc
    })

    return avg_loss, avg_ncc

def val_epoch(model, criterion, device, val_loader):
    loss_log, ncc_log = [], []

    model.eval()
    for batch_num, (batch_image, batch_heatmaps) in enumerate(tqdm.tqdm(val_loader, desc="Validation Epoch")):
        batch_image = batch_image.to(device, non_blocking=True)
        batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(batch_image)

        loss = criterion(logits, batch_heatmaps)
        loss_log.append(loss.item())
        ncc_value = NCC(logits, batch_heatmaps)
        ncc_log.append(ncc_value)

        wandb.log({
                "val/batch_loss": loss.item(),
                "val/batch_ncc": ncc_value,
                "val/val_batch_num": batch_num
            })

    avg_loss = np.mean(loss_log)
    avg_ncc = np.mean(ncc_log)

    wandb.log({
        "val/epoch_loss": avg_loss,
        "val/epoch_ncc": avg_ncc
    })

    return avg_loss, avg_ncc

def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, device, n_epoch):
    wandb.watch(model, log="all", log_freq=10)

    for epoch in tqdm.trange(n_epoch, desc="Training Progress"):

        train_loss, train_ncc = train_epoch(model, optimizer, criterion, device, train_loader, scheduler)
        val_loss, val_ncc = val_epoch(model, criterion, device, val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ncc": train_ncc,
            "val_loss": val_loss,
            "val_ncc": val_ncc
        })

        try:
            # Трассируем модель
            example_input = torch.randn(1, 3, 224, 224).to(device, non_blocking=True)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(f"../models/hand_keypoints_traced_{epoch}.pt")
            print("✅ Модель сохранена")
        except Exception as e:
            print(f"❌ Ошибка при tracing: {e}")