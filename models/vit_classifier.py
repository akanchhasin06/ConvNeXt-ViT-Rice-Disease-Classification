import torch
import torch.nn as nn

from .vit_encoder import ViTEncoder


class ViTClassifier(nn.Module):

    def __init__(
        self,
        num_classes=10,
        pretrained=True,
        freeze_backbone=True
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

        self.classifier = nn.Sequential(

            nn.Linear(192, 512),

            nn.GELU(),

            nn.Dropout(0.4),

            nn.Linear(512, 256),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)

        )

    def forward(self, x):

        x = self.encoder(x)

        cls_token = x[:, 0]

        x = self.classifier(cls_token)

        return x