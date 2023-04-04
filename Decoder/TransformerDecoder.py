import torch.nn as nn

from Decoder.TransformerDecoderLayer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):

  def __init__(self, dim, num_heads, dim_hidden, drop_rate, num_layers):
    super().__init__()
    self.layers = nn.ModuleList([TransformerDecoderLayer(dim, num_heads, dim_hidden, drop_rate) for _ in range(num_layers)])
    self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
    self.state = {}  # Decoder state

  def forward(self, target, memory, mask_target=None, mask_source=None, step=-1):
    # Must have separate caches for different devices, for data parallel.
    cache_name = 'cache_{}'.format(memory.get_device())

    if step == 0:  # First step in decoding, need to initialize cache.
      self._init_cache(cache_name)

    out = target
    self_attentions = []
    cross_attentions = []
    for index_layer, layer in enumerate(self.layers):
      layer_cache = None if step < 0 else self.state[cache_name]['layer_{}'.format(index_layer)]
      out, self_attention, cross_attention = layer(out, memory, mask_target, mask_source, layer_cache)
      self_attentions.append(self_attention)
      cross_attentions.append(cross_attention)
    return self.layer_norm(out), self_attentions, cross_attentions

  def _init_cache(self, cache_name):
    self.state[cache_name] = {}
    for index_layer, layer in enumerate(self.layers):
      self.state[cache_name]['layer_{}'.format(index_layer)] = {'memory_key': None, 'memory_value': None, 'self_key': None, 'self_value': None}

  def remap_cache(self, indices):
    for cache_name in self.state:
      for layer_name in self.state[cache_name]:
        for name, tensor in self.state[cache_name][layer_name].items():
            self.state[cache_name][layer_name][name] = tensor.index_select(0, indices)  # Batch dim 0