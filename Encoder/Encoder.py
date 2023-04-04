import torch.nn as nn


# PositionwiseFeedForward
class Encoder(nn.Module):

  def __init__(self, dim, dim_hidden, drop_rate=0.1):
    super().__init__()
    self.w1 = nn.Linear(dim, dim_hidden)
    self.w2 = nn.Linear(dim_hidden, dim)
    self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
    self.drop1 = nn.Dropout(drop_rate)
    self.relu = nn.ReLU()
    self.drop2 = nn.Dropout(drop_rate)

  def forward(self, x):
    inter = self.drop1(self.relu(self.w1(self.layer_norm(x))))
    output = self.drop2(self.w2(inter))
    return output + x