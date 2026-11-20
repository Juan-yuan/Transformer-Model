"""
Encoder sub-layer demos:
  1) Multi-Head Attention + residual (Add) + LayerNorm
  2) Feed Forward + residual (Add) + LayerNorm
"""

import sys
from pathlib import Path

# Prefer this folder's config.py over other projects named config (e.g. Smart-Voyage).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import FeedForward, MultiHeadedAttention, SublayerConnection
from transformer_input_embedding import test_input_embedding


def test_encoder_sublayer(option='multihead'):
    x = test_input_embedding()
    sublayer_conn = SublayerConnection(config.d_model, config.dropout_p)

    if option == 'multihead':
        mha = MultiHeadedAttention(config.embed_dim, config.num_heads, config.dropout_p)
        result = sublayer_conn(x, lambda t: mha(t, t, t))
        print(f'residual+norm (multi-head): {result.shape}')
    elif option == 'feedforward':
        ffn = FeedForward(config.d_model, config.d_ff, config.dropout_p)
        result = sublayer_conn(x, ffn)
        print(f'residual+norm (feed-forward): {result.shape}')
    else:
        raise ValueError(f'Invalid option: {option}')

    return result


if __name__ == '__main__':
    test_encoder_sublayer(option='multihead')
    test_encoder_sublayer(option='feedforward')
