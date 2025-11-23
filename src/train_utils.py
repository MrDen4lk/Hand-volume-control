import torch
import tqdm.auto as tqdm
import wandb
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler


def NCC(logit, target):
    logit = logit - logit.mean(dim=[2, 3], keepdim=True)
    target = target - target.mean(dim=[2, 3], keepdim=True)
    numerator = (logit * target).sum(dim=[2, 3])
    denominator = torch.sqrt((logit ** 2).sum(dim=[2, 3]) * (target ** 2).sum(dim=[2, 3]) + 1e-6)
    score = numerator / denominator
    return score.mean().item()


def train_epoch(model, optimizer, criterion, device, train_loader, scaler, scheduler=None):
    loss_log, ncc_log = [], []

    model.train()
    for batch_num, (batch_image, batch_heatmaps) in enumerate(tqdm.tqdm(train_loader, desc="Training Epoch")):
        batch_image = batch_image.to(device, non_blocking=True)
        batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP
        with autocast(enabled=(device == 'cuda')):
            logits = model(batch_image)
            loss = criterion(logits, batch_heatmaps)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Логирование
        with torch.no_grad():
            loss_value = loss.item()
            ncc_value = NCC(logits.detach(), batch_heatmaps.detach())

            loss_log.append(loss_value)
            ncc_log.append(ncc_value)

            wandb.log({
                "train/batch_loss": loss_value,
                "train/batch_ncc": ncc_value,
                "train/batch_num": batch_num
            })

    if scheduler is not None:
        scheduler.step()

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
            with autocast(enabled=(device == 'cuda')):
                logits = model(batch_image)
                loss = criterion(logits, batch_heatmaps)

            loss_log.append(loss.item())
            ncc_value = NCC(logits, batch_heatmaps)
            ncc_log.append(ncc_value)

    avg_loss = np.mean(loss_log)
    avg_ncc = np.mean(ncc_log)

    wandb.log({
        "val/epoch_loss": avg_loss,
        "val/epoch_ncc": avg_ncc
    })

    return avg_loss, avg_ncc


def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, device, n_epoch):
    wandb.watch(model, log="all", log_freq=10)

    # AMP
    scaler = GradScaler(enabled=(device == 'cuda'))

    # Создаем папку для моделей, если нет
    os.makedirs("../models", exist_ok=True)
    best_ncc = -1.0

    for epoch in tqdm.trange(n_epoch, desc="Training Progress"):
        train_loss, train_ncc = train_epoch(model, optimizer, criterion, device, train_loader, scaler, scheduler)
        val_loss, val_ncc = val_epoch(model, criterion, device, val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ncc": train_ncc,
            "val_loss": val_loss,
            "val_ncc": val_ncc
        })

        try:
            model.eval()

            dummy_input = torch.randn(1, 3, 224, 224, device=device)

            onnx_path = f"../models/hand_keypoints_epoch_{epoch}.onnx"

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"✅ Модель сохранена в ONNX: {onnx_path}")

        except Exception as e:
            print(f"❌ Ошибка при экспорте в ONNX: {e}")