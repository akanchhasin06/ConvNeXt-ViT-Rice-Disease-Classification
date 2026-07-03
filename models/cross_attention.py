import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim)

        self.value = nn.Linear(embed_dim, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(

            nn.Linear(embed_dim, embed_dim * 4),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(embed_dim * 4, embed_dim)

        )

    def forward(
        self,
        query,
        key,
        value
    ):

        q = self.query(query)

        k = self.key(key)

        v = self.value(value)

        attn_output, attn_weights = self.attention(
            q,
            k,
            v
        )

        x = self.norm1(
            query + attn_output
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x, attn_weights