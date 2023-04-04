import math

import torch
import torch.nn as nn


# SinusoidalPositioner
class PositionEmbedding(nn.Module):

  def __init__(self, dim, drop_rate=0.1, length_max=5000):
    super().__init__()
    frequency = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.) / dim))  # Using different frequency for each dim
    positions = torch.arange(0, length_max).unsqueeze(1)
    wave = torch.zeros(length_max, dim)
    wave[:, 0::2] = torch.sin(frequency * positions)
    wave[:, 1::2] = torch.cos(frequency * positions)
    self.register_buffer('wave', wave.unsqueeze(0))  # (1, length_max, dim)
    self.dropout = nn.Dropout(drop_rate)
    self.dim = dim
    self.length_max = length_max

  def forward(self, x, step=-1):
    assert x.size(-2) <= self.length_max

    if step < 0:  # Take the corresponding leftmost embeddings.
      position_encoding = self.wave[:, :x.size(-2), :]
    else:  # Take the embedding at the step.
      position_encoding = self.wave[:, step, :]

    x = x * math.sqrt(self.dim)
    return self.dropout(x + position_encoding)