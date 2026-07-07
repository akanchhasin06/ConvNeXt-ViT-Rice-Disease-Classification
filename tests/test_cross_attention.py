import torch

from models.cross_attention import CrossAttention

model = CrossAttention()

query = torch.randn(2,1,256)

key = torch.randn(2,1,256)

value = torch.randn(2,1,256)

output, weights = model(
    query,
    key,
    value
)

print(output.shape)

print(weights.shape)