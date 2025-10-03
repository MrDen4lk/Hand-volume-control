from torchvision.models.convnext import convnext_small, ConvNeXt_Small_Weights
from torch import nn
import torch


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EfficientNetHeatmap(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        encoder_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.encoder = encoder_model.features

        self.decoder = nn.Sequential(
            UpBlock(768, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x