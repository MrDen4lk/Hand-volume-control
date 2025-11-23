import torch
import torch.nn as nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class HandPoseUNet(nn.Module):
    def __init__(self, out_channels=21):
        super().__init__()

        # 2. ЗАГРУЖАЕМ BASE MODEL
        weights = ConvNeXt_Base_Weights.DEFAULT
        base_model = convnext_base(weights=weights)

        # Имена слоев остаются те же, но каналы внутри них изменились
        return_nodes = {
            'features.1': 'layer1',  # 128 каналов
            'features.3': 'layer2',  # 256 каналов
            'features.5': 'layer3',  # 512 каналов
            'features.7': 'layer4',  # 1024 канала
        }

        self.encoder = create_feature_extractor(base_model, return_nodes=return_nodes)

        # 3. ОБНОВЛЯЕМ ДЕКОДЕР ПОД НОВЫЕ РАЗМЕРЫ

        # Up 1: 1024 (l4) -> 512 (l3)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Вход: 1024 + 512 = 1536. Выход: 512
        self.conv1 = DoubleConv(1024 + 512, 512)

        # Up 2: 512 -> 256 (l2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Вход: 512 + 256 = 768. Выход: 256
        self.conv2 = DoubleConv(512 + 256, 256)

        # Up 3: 256 -> 128 (l1)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Вход: 256 + 128 = 384. Выход: 128
        self.conv3 = DoubleConv(256 + 128, 128)

        # Final Head (Stride 4 output)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        l1 = features['layer1']
        l2 = features['layer2']
        l3 = features['layer3']
        l4 = features['layer4']

        x = self.up1(l4)
        x = torch.cat([x, l3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, l2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, l1], dim=1)
        x = self.conv3(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x