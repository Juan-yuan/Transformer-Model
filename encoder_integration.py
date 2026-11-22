"""
Full encoder demo:
  1. Default stack is N encoder layers (paper default N=6; uses config.num_layers).
  2. Each layer has two sublayers: multi-head self-attention and feed-forward.
  3. Each sublayer uses residual connection + LayerNorm.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import Encoder, EncoderLayer, FeedForward, MultiHeadedAttention
from transformer_input_embedding import test_input_embedding


def test_encoder():
    x = test_input_embedding()
    multi_head = MultiHeadedAttention(config.d_model, config.num_heads)
    feed_forward = FeedForward(config.d_model, config.d_ff)
    encoder_layer = EncoderLayer(config.d_model, multi_head, feed_forward)
    encoder = Encoder(encoder_layer, N=config.num_layers)

    # No mask for this shape check.
    result = encoder(x, mask=None)
    print(f'encoder output shape: {result.shape}')  # [batch, seq, d_model]
    return result


if __name__ == '__main__':
    test_encoder()
