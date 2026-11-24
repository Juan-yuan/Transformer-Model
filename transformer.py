"""
Full Transformer demo:
  source embed -> encoder
  target embed -> decoder (with encoder memory)
  -> generator over the vocabulary
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    FeedForward,
    Generator,
    MultiHeadedAttention,
    PositionEmbedding,
    TokenEmbedding,
)


def test_transformer():
    # Encoder stack
    source_embed = TokenEmbedding(config.vocab_size, config.embed_dim)
    source_position = PositionEmbedding(config.max_len, config.embed_dim)
    self_attn = MultiHeadedAttention(config.d_model, config.num_heads)
    feed_forward = FeedForward(config.d_model, config.d_ff)
    encoder_layer = EncoderLayer(config.d_model, self_attn, feed_forward)
    encoder = Encoder(encoder_layer, N=config.num_layers)

    # Decoder stack (deep-copy shared building blocks)
    target_embed = copy.deepcopy(source_embed)
    target_position = copy.deepcopy(source_position)
    self_attn1 = copy.deepcopy(self_attn)
    source_attn1 = copy.deepcopy(self_attn)
    ff1 = copy.deepcopy(feed_forward)
    decoder_layer = DecoderLayer(config.d_model, self_attn1, source_attn1, ff1)
    decoder = Decoder(decoder_layer, N=config.num_layers)

    # Output head
    output_classifier = Generator(config.d_model, config.vocab_size)

    transformer = EncoderDecoder(
        nn.Sequential(source_embed, source_position),
        encoder,
        nn.Sequential(target_embed, target_position),
        decoder,
        output_classifier,
    )

    source_x = torch.randint(1, config.vocab_size, (config.batch_size, config.seq_len))
    target_y = torch.randint(1, config.vocab_size, (config.batch_size, config.seq_len))
    # No masks for this shape check (all-zero masks would blank out attention).
    result = transformer(source_x, target_y, source_mask=None, target_mask=None)
    print(f'transformer output shape: {result.shape}')  # [batch, seq, vocab_size]
    return result


if __name__ == '__main__':
    test_transformer()
