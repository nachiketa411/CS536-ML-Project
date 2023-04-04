import torch.nn as nn

from TransformerEncoderLayer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):

  def __init__(self, dim, num_heads, dim_hidden, drop_rate, num_layers):
    super().__init__()
    self.layers = nn.ModuleList([TransformerEncoderLayer(dim, num_heads, dim_hidden, drop_rate) for _ in range(num_layers)])
    self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

  def forward(self, source, mask_source=None):
    out = source
    self_attentions = []
    for layer in self.layers:
      out, attention = layer(out, mask_source)
      self_attentions.append(attention)
    return self.layer_norm(out), self_attentions