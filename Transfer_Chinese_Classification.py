"""
Chinese text classification with a pretrained BERT backbone and a custom head.

Data files (same schema): data/train.csv, data/validation.csv, data/test.csv
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# Keep Hugging Face caches inside the project (avoids sandbox/home permission issues).
_HF_CACHE = _ROOT / '.hf_cache'
os.environ.setdefault('HF_HOME', str(_HF_CACHE))
os.environ.setdefault('HF_DATASETS_CACHE', str(_HF_CACHE / 'datasets'))
os.environ.setdefault('TRANSFORMERS_CACHE', str(_HF_CACHE / 'transformers'))


from config import (
    CLASSIF_DATA_DIR,
    BertClassifier,
    create_text_dataloader,
    get_torch_device,
    load_bert_tokenizer,
    train_model,
)

device = get_torch_device()


def load_datasets():
    from datasets import load_dataset

    train = load_dataset('csv', data_files=str(CLASSIF_DATA_DIR / 'train.csv'), split='train')
    valid = load_dataset('csv', data_files=str(CLASSIF_DATA_DIR / 'validation.csv'), split='train')
    test = load_dataset('csv', data_files=str(CLASSIF_DATA_DIR / 'test.csv'), split='train')
    print(f'train: {train}')
    print(f'validation: {valid}')
    print(f'test: {test}')
    return train, test, valid


def test_bert_classifier():
    tokenizer = load_bert_tokenizer()
    dataloader = create_text_dataloader(CLASSIF_DATA_DIR / 'train.csv', tokenizer=tokenizer)
    model = BertClassifier().to(device)

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        output = model(input_ids, attention_mask)
        print(f'classifier output shape: {output.shape}')  # [batch, num_classes]
        print(f'labels shape: {labels.shape}')
        break
    return model


if __name__ == '__main__':
    print(f'device: {device}')
    load_datasets()
    test_bert_classifier()
    train_model(freeze_bert=True, device=device)
