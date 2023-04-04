import torch.nn as nn

from Encoder.Encoder import Encoder
from MultiheadAttention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):

  def __init__(self, dim, num_heads, dim_hidden, drop_rate):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
    self.self_attention = MultiHeadAttention(dim, num_heads, drop_rate)
    self.drop = nn.Dropout(drop_rate)
    self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
    self.context_attention = MultiHeadAttention(dim, num_heads, drop_rate)
    self.feedforward = Encoder(dim, dim_hidden, drop_rate)

  def forward(self, target, memory, mask_target, mask_source, layer_cache=None):
    # TODO: Implement
    normed = self.layer_norm1(target)  # Apply layer_norm1 to target.
    query, self_attention = self.self_attention(normed, normed, normed, mask_target, layer_cache, memory_attention = False)  # Apply self-attention on normed (be sure to use mask_target, layer_cache, and set memory_attention correctly).

    query = self.drop(query) + target  # Re-write query by applying dropout and adding a residual connection to target.
    attended, cross_attention = self.context_attention(self.layer_norm2(query),
                                                       memory, memory, mask_source,
                                                       layer_cache,
                                                       memory_attention = True)  # Apply cross-attention using **layer_norm2(query)** as query and **memory** as key/value (be sure to use mask_source, layer_cache, and set memory_attention correctly).
    attended = self.drop(attended) + query  # Re-write attended by applying dropout and adding a residual connection to query.
    return self.feedforward(attended), self_attention, cross_attention