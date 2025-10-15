"""Utilities for handling JSONL training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from ala.config import FolderPaths


def merge_json_to_jsonl(output_file: Path, args) -> None:
    folders = FolderPaths(args)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        for file in sorted(folders.json_data_folder.glob('*.json'), key=lambda path: path.name):
            with file.open('r') as json_file:
                data = json.load(json_file)
                print('loaded json file {}'.format(data))
                for qa_pair in data:
                    question = qa_pair['question']
                    answer = qa_pair['answer']
                    validate = qa_pair.get('validate', 1)
                    qa_pair_text = (
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                        "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>"
                        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>"
                    )
                    json_line = {
                        "text": qa_pair_text,
                        "question": question,
                        "answer": answer,
                        "validate": validate,
                    }
                    json.dump(json_line, f)
                    f.write('\n')


def jsonl_split(input_jsonl_file: Path, args) -> None:
    folders = FolderPaths(args)

    if not input_jsonl_file.exists():
        print("The input JSONL file does not exist. Skipping train-test split.")
        return

    with input_jsonl_file.open('r') as file:
        full_jsonl = [json.loads(line) for line in file]

    if not full_jsonl:
        print("The input JSONL file is empty. Skipping train-test split.")
        return

    full_df = pd.DataFrame(full_jsonl)
    if full_df.empty:
        print("The input JSONL file is empty after loading. Skipping train-test split.")
        return

    full_df = full_df[full_df['validate'] == 1]
    if full_df.empty:
        print("No valid entries to split. Skipping train-test split.")
        return

    full_df = full_df.drop(columns=['validate'])

    train, val = train_test_split(full_df, test_size=args.val_set_size, random_state=42)

    train_path = folders.finetune_data_folder / 'train.jsonl'
    val_path = folders.finetune_data_folder / 'valid.jsonl'
    train.to_json(train_path, orient='records', lines=True)
    val.to_json(val_path, orient='records', lines=True)
