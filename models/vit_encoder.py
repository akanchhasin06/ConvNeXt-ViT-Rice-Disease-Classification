import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):

    def __init__(
        self,
        model_name="vit_tiny_patch16_224",
        pretrained=True,
        freeze_backbone=False
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):

        return self.backbone.forward_features(x)