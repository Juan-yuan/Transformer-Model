"""
Encoder layer:
  token + positional embedding
  -> (repeated N times in a full encoder, default N=6)
       multi-head self-attention -> residual + LayerNorm
       -> feed-forward -> residual + LayerNorm
  -> decoder / next stage
"""

import sys
from pathlib import Path

import torch

# Prefer this folder's config.py over other projects named config (e.g. Smart-Voyage).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import EncoderLayer, FeedForward, MultiHeadedAttention
from transformer_input_embedding import test_input_embedding


def test_encoder_layer():
    x = test_input_embedding()
    multi_head = MultiHeadedAttention(config.d_model, config.num_heads)
    feed_forward = FeedForward(config.d_model, config.d_ff)
    encoder_layer = EncoderLayer(config.d_model, multi_head, feed_forward)

    # No mask for this shape check (pass a causal mask when testing decoder-style blocking).
    result = encoder_layer(x, mask=None)
    print(f'encoder layer output shape: {result.shape}')  # [batch, seq, d_model]
    return result


if __name__ == '__main__':
    test_encoder_layer()
