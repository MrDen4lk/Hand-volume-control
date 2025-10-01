import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn

class EfficientNetHeatmap(nn.Module):
    def __init__(self, out_channels):
        super(EfficientNetHeatmap, self).__init__()

        self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
        encoder_out_channels = 1280

        self.decoder = nn.Sequential(
            # Upsample block 1: 7x7 -> 14x14
            nn.Conv2d(encoder_out_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Upsample block 2: 14x14 -> 28x28
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Upsample block 3: 28x28 -> 56x56
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Final convolution to get the right number of channels
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x) # [B, 1280, 7, 7]
        x = self.decoder(x) # [B, out_channels, H, W]
        return x