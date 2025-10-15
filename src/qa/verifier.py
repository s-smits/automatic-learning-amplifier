"""Verification utilities for generated question-answer pairs."""

from __future__ import annotations

import inflect
import json

from mlx_lm.utils import generate, load

from config import BaseModelPaths
from utils import extract_json


def verify_outputs(jsonl_data, text_chunk, args) -> bool:
    base_paths = BaseModelPaths()
    prompt = f"""
<[begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a genius fact verification assistant that helps analyzing the text and outputs \"validate\" JSON keys.<|eot_id|><|start_header_id|>user<|end_header_id|>
<prompt>Analyze the text corpus and the {args.question_amount} {inflect.engine().number_to_words(args.question_amount)} questions and answers based on the text corpus with the following structure:
{{
    "text": [
        {{
            "validate": 0 or 1
        }},
        ...
        {{
            "validate": 0 or 1
        }}
    ]
}}
For each question and answer combination, only respond with key {{validate: 1}} if the QA combination is correct. If not correct, only respond with {{validate: 0}}.</prompt>
<text_corpus>{text_chunk}</text_corpus>
<questions_answers>{json.dumps(jsonl_data, indent=2)}</questions_answers><|eot_id|>
<|start_header_id|>assistant<|end_header_id|><json_format_only_per_qa_validate_keys>
"""
    model, tokenizer = load(base_paths.get_model_path('mlx_4bit'))
    output = generate(model, tokenizer, prompt, temp=0.4, max_tokens=256, verbose=True)
    verification_response = extract_json(output)

    if not verification_response:
        print("Skipping verification update due to invalid response.")
        return False

    for i, qa_pair in enumerate(jsonl_data):
        if i >= len(verification_response):
            print(f"No validation data for QA pair index {i}")
            continue
        validate_value = verification_response[i].get('validate')
        if isinstance(validate_value, int):
            qa_pair['validate'] = validate_value
        else:
            print(f"Invalid validation flag for QA pair index {i}")
    return True
