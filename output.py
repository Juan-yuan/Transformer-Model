"""
Output layer demo:
  map decoder hidden states to next-token log-probabilities over the vocabulary
"""

import sys
from pathlib import Path

# Prefer this folder's config.py over other projects named config (e.g. Smart-Voyage).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import Generator
from decoder import test_decoder


def test_generator():
    decoder_output = test_decoder()
    output_classifier = Generator(config.d_model, config.vocab_size)
    output = output_classifier(decoder_output)
    print(f'generator output shape: {output.shape}')  # [batch, seq, vocab_size]
    return output


if __name__ == '__main__':
    test_generator()
