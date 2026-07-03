import torch
import torch.nn as nn
import timm


class ConvNeXtEncoder(nn.Module):

    def __init__(
        self,
        model_name="convnext_tiny",
        pretrained=True,
        freeze_stages=False
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        if freeze_stages:

            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):

        features = self.backbone.forward_features(x)

        return features