"""
Full decoder demo:
  stack N decoder layers (paper default N=6; demo uses config.num_layers)
  then apply a final LayerNorm for stable outputs
"""

import copy
import sys
from pathlib import Path

# Prefer this folder's config.py over other projects named config (e.g. Smart-Voyage).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import Decoder, DecoderLayer, FeedForward, MultiHeadedAttention
from encoder_integration import test_encoder
from transformer_input_embedding import test_input_embedding


def test_decoder():
    x = test_input_embedding()
    encoder_output = test_encoder()

    multi_attn = MultiHeadedAttention(config.d_model, config.num_heads)
    self_attn = copy.deepcopy(multi_attn)
    src_attn = copy.deepcopy(multi_attn)
    feed_forward = FeedForward(config.d_model, config.d_ff)
    decoder_layer = DecoderLayer(config.d_model, self_attn, src_attn, feed_forward)
    decoder = Decoder(decoder_layer, N=config.num_layers)

    # No masks for this shape check.
    result = decoder(x, encoder_output, source_mask=None, target_mask=None)
    print(f'decoder output shape: {result.shape}')  # [batch, seq, d_model]
    return result


if __name__ == '__main__':
    test_decoder()
