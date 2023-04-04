import torch.nn as nn

from Encoder import Encoder
from MultiheadAttention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):

  def __init__(self, dim, num_heads, dim_hidden, drop_rate):
    super().__init__()
    self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
    self.self_attention = MultiHeadAttention(dim, num_heads, drop_rate)
    self.drop = nn.Dropout(drop_rate)
    self.feedforward = Encoder(dim, dim_hidden, drop_rate)

  def forward(self, source, mask_source=None):
    # TODO: Implement
    normed = self.layer_norm(source)  # Apply layer norm on source
    attended, attention = self.self_attention(normed, normed, normed, mask_source)  # Apply self-attention on normed (be sure to use mask_source).
    attended = self.drop(attended) + source  # Re-write attended by applying dropout and adding a residual connection to source.
    return self.feedforward(attended), attention