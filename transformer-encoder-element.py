"""
Encoder building blocks demo:
component -> sub-layer pieces used inside an encoder layer
  - multi-head attention + residual + norm
  - feed-forward + residual + norm

Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
"""

import sys
from pathlib import Path

import torch

# Prefer this folder's config.py over other projects named config (e.g. Smart-Voyage).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import (
    FeedForward,
    LayerNorm,
    MultiHeadedAttention,
    attention,
    create_causal_mask,
)
from transformer_input_embedding import test_input_embedding


def test_attention():
    x = test_input_embedding()
    query = key = value = x

    result, p_attn = attention(query, key, value)
    print(f'no-mask result: {result.shape}')   # [batch, seq, d_model]
    print(f'no-mask p_attn: {p_attn.shape}')   # [batch, seq, seq]

    mask = create_causal_mask((config.batch_size, config.seq_len, config.seq_len))
    result, p_attn = attention(query, key, value, mask)
    print(f'causal-mask result: {result.shape}')
    print(f'causal-mask p_attn: {p_attn.shape}')
    return result, p_attn


def test_multihead_attention():
    x = test_input_embedding()
    query = key = value = x
    mh_attention = MultiHeadedAttention(config.d_model, config.num_heads, config.dropout_p)
    mask = create_causal_mask((config.batch_size, config.seq_len, config.seq_len))
    result = mh_attention(query, key, value, mask)
    print(f'multi-head attention result: {result.shape}')  # [batch, seq, d_model]
    return result


def test_feedforward():
    x = test_multihead_attention()
    feedforward = FeedForward(config.d_model, config.d_ff, config.dropout_p)
    result = feedforward(x)
    print(f'feed-forward result: {result.shape}')
    return result


def test_layernorm():
    x = test_multihead_attention()
    layernorm = LayerNorm(num_features=config.d_model)
    result = layernorm(x)
    print(f'layer-norm result: {result.shape}')

    # Optional: compare with torch.nn.LayerNorm
    # layernorm = torch.nn.LayerNorm(config.d_model)
    # res = layernorm(x)
    # print(f'official layer-norm result: {res.shape}')
    # is_close = torch.allclose(result, res, 1e-4)
    # print(f'values close: {is_close}')
    return result


if __name__ == '__main__':
    test_attention()
    test_multihead_attention()
    test_feedforward()
    test_layernorm()
