import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import TokenEmbedding, PositionEmbedding


def test_token_embedding():
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_len))
    embed_token = TokenEmbedding(config.vocab_size, config.d_model)
    return embed_token(x)


def test_input_embedding():
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_len))
    embed_token = TokenEmbedding(config.vocab_size, config.d_model)
    embed_position = PositionEmbedding(config.max_len, config.d_model)
    # Equivalent: nn.Sequential(embed_token, embed_position)(x)
    return embed_position(embed_token(x))


if __name__ == '__main__':
    token_embeddings = test_token_embedding()
    print(f'partial token embeddings:\n{token_embeddings[:, :, :5]}')
    print(f'token_embeddings.shape: {token_embeddings.shape}')

    input_embeddings = test_input_embedding()
    print(f'partial input embeddings:\n{input_embeddings[:, :, :5]}')
    print(f'input_embeddings.shape: {input_embeddings.shape}')
