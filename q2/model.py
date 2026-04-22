import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class UNet(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()

        # ✅ Updated pretrained usage
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)

        # 🔥 CRITICAL FIX (shape mismatch solved here)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x