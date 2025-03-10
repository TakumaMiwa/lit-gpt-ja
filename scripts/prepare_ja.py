import json
import glob
import os
from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from lit_gpt import Tokenizer
import lit_gpt.packed_dataset as packed_dataset

sample_ids = [
        "izumi-lab/wikinews-ja-20230728",
        "izumi-lab/wikinews-en-20230728",
        "izumi-lab/wikipedia-ja-20230720",
        "izumi-lab/wikipedia-en-20230720",
        'izumi-lab/open-text-books',
        'if001/oscar_2023_filtered',
        'if001/aozorabunko-clean-sin'
]


def format_number(num):
    if abs(num) >= 10**12:  # Trillion
        return "{:.2f}T".format(num / 10**12)
    elif abs(num) >= 10**9:  # Billion
        return "{:.2f}B".format(num / 10**9)
    elif abs(num) >= 10**6:  # Million
        return "{:.2f}M".format(num / 10**6)
    else:
        return str(num)

def prepare_for_dataset(dataset_ids, tokenizer_path, destination_path, chunk_size):

    destination_path.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path)

    total_token_cnt = 0
    for dataset_id in dataset_ids:
        token_cnt = 0
        print(f"Processing {dataset_ids}")
        prefix = dataset_id.split('/')[-1]
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )
        ds = load_dataset(dataset_id)
        ds = ds['train']

        if 'aozora' in dataset_id:
            for v in ds['text']:
                text_ids = tokenizer.encode(v)
                token_cnt += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        else:
            cnt = 0 
            for v in ds:
                text_ids = tokenizer.encode(v['text'])
                token_cnt += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        builder.write_reminder()
        print('tokens ', format_number(token_cnt))
        total_token_cnt += token_cnt
    print('total tokens', format_number(total_token_cnt))

def prepare(
    dataset_ids = sample_ids,
    tokenizer_path: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    destination_path: Path = Path("data/ja_data"),
    chunk_size: int = 2049 * 1024,  # 2048 block size + 1 for causal (from LLama), 1024 blocks
    sample: bool = False
) -> None: 
    prepare_for_dataset(
        dataset_ids=dataset_ids,
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=chunk_size,
    )

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)