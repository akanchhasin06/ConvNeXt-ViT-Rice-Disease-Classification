import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        hidden_dim=512,
        dropout=0.3
    ):
        super().__init__()

        self.fusion = nn.Sequential(

            nn.Linear(embed_dim * 2, hidden_dim),

            nn.BatchNorm1d(hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embed_dim),

            nn.BatchNorm1d(embed_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )

    def forward(self, conv_features, vit_features):

        fused = torch.cat(
            [conv_features, vit_features],
            dim=1
        )

        fused = self.fusion(fused)

        return fused