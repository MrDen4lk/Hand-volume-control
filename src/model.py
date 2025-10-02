from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torch import nn

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class EfficientNetHeatmap(nn.Module):
    def __init__(self, out_channels):
        super(EfficientNetHeatmap, self).__init__()

        encoder_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.encoder = encoder_model.features

        self.decoder = nn.Sequential(
            UpBlock(1024, 512),  # 7x7 -> 14x14
            UpBlock(512, 256),  # 14x14 -> 28x28
            UpBlock(256, 128),  # 28x28 -> 56x56
            UpBlock(128, 64),  # 56x56 -> 112x112
            UpBlock(64, 32),  # 112x112 -> 224x224
            nn.Conv2d(32, out_channels, kernel_size=1) # финальный слой
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x