import torch
import torch.nn as nn

from .convnext_encoder import ConvNeXtEncoder
from .vit_encoder import ViTEncoder
from .cross_attention import CrossAttention
from .fusion import FeatureFusion


class HybridModel(nn.Module):

    def __init__(
        self,
        num_classes=10,
        pretrained=True,
        freeze_backbone=False
    ):
        super().__init__()

        # -------------------------------
        # Encoders
        # -------------------------------

        self.convnext = ConvNeXtEncoder(
            pretrained=pretrained,
            freeze_stages=freeze_backbone
        )

        self.vit = ViTEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

        # -------------------------------
        # ConvNeXt Feature Projection
        # -------------------------------

        self.conv_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(768,512),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(512,256)

        )

        # -------------------------------
        # ViT Feature Projection
        # -------------------------------

        self.vit_projection = nn.Sequential(

            nn.Linear(192,512),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(512,256)

        )

        # -------------------------------
        # Cross Attention
        # -------------------------------

        self.cross_attention = CrossAttention(
            embed_dim=256,
            num_heads=8
        )

        # -------------------------------
        # Feature Fusion
        # -------------------------------

        self.fusion = FeatureFusion(
            embed_dim=256
        )

        # -------------------------------
        # Classification Head
        # -------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(256,128),

            nn.GELU(),

            nn.Dropout(0.4),

            nn.Linear(128,num_classes)

        )

    def forward(self,x):

        # -------------------------------
        # ConvNeXt
        # -------------------------------

        conv = self.convnext(x)

        conv = self.conv_pool(conv)

        conv = self.conv_projection(conv)

        # -------------------------------
        # ViT
        # -------------------------------

        vit = self.vit(x)

        vit = vit[:,0]

        vit = self.vit_projection(vit)

        # -------------------------------
        # Cross Attention
        # -------------------------------

        conv_token = conv.unsqueeze(1)

        vit_token = vit.unsqueeze(1)

        conv_attended,_ = self.cross_attention(

            conv_token,

            vit_token,

            vit_token

        )

        vit_attended,_ = self.cross_attention(

            vit_token,

            conv_token,

            conv_token

        )

        conv_attended = conv_attended.squeeze(1)

        vit_attended = vit_attended.squeeze(1)

        # -------------------------------
        # Fusion
        # -------------------------------

        fused = self.fusion(

            conv_attended,

            vit_attended

        )

        # -------------------------------
        # Classification
        # -------------------------------

        out = self.classifier(fused)

        return out