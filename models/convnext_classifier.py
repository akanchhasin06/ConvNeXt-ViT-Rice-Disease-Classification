import torch
import torch.nn as nn

from .convnext_encoder import ConvNeXtEncoder


class ConvNeXtClassifier(nn.Module):

    def __init__(
        self,
        num_classes=10,
        pretrained=True,
        freeze_backbone=True
    ):
        super().__init__()

        self.encoder = ConvNeXtEncoder(
            pretrained=pretrained,
            freeze_stages=freeze_backbone
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(

        nn.Linear(768, 512),

        nn.GELU(),

        nn.Dropout(0.4),

        nn.Linear(512, 256),

        nn.GELU(),

        nn.Dropout(0.3),

        nn.Linear(256, num_classes)
    )

    def forward(self, x):

        x = self.encoder(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.classifier(x)

        return x