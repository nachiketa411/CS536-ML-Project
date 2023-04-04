import math

import torch
import torch.nn as nn
import seaborn


def scaled_dot(query, key, mask_key=None):
    score = torch.matmul(query, key.transpose(-2, -1))
    score /= math.sqrt(query.size(-1))
    if mask_key is not None:
        score = score.masked_fill(mask_key, -1e18)  # Represents negative infinity
    return score


def attend(query, key, value, mask_key=None, dropout=None):
    score = scaled_dot(query, key, mask_key=mask_key)  # Use scaled_dot, be sure to mask key
    attention = nn.Softmax(dim=1)(score)
    if dropout is not None:
        attention = nn.Dropout(p=dropout)(attention)
    answer = torch.matmul(attention, value)  # Convexly combine value embeddings using attention, this should be just a matrix-matrix multiplication.
    return answer, attention


def draw_attention(attention, x, y, ax=None):  # attention: matrix with probabilities as elements
    seaborn.heatmap(attention, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False, ax=ax)


def split_heads(batch, num_heads):
    (batch_size, length, dim) = batch.size()  # These are the expected batch dimensions.
    assert dim % num_heads == 0  # Assert that dimension is divisible by the number of heads.
    dim_head = dim // num_heads
    splitted = batch.view(batch_size, -1, num_heads, dim_head).transpose(1, 2)
    return splitted


def merge_heads(batch):
    (batch_size, num_heads, length, dim_head) = batch.size()  # These are the expected batch dimensions.
    # New memory allocation (reshape), can't avoid.
    merged = batch.transpose(1, 2).reshape(batch_size, -1, num_heads * dim_head)
    return merged  # (batch_size, length, dim)


def get_mask_future(num_positions):
    return torch.triu(torch.ones(1, num_positions, num_positions), diagonal=1) == 1

