"""Model comparison utilities."""

from __future__ import annotations

from typing import Callable, Tuple

import mlx.core as mx
from mlx_lm.utils import generate, load

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - Streamlit unavailable in tests
    st = None

from config import BaseModelPaths, FolderPaths


def create_eval_prompt_local(question: str) -> str:
    return (
        "<[begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant."
        "<|eot_id|><|start_header_id|>user<|end_header_id|><prompt>"
        f"{question}</prompt><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


def create_eval_prompt_anthropic(question: str) -> str:
    return f"<system>You are a helpful assistant.</system><prompt>{question}</prompt>"


def load_finetuned_model(args):
    folders = FolderPaths(args)
    model_path = folders.ft_folder
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Finetuned model file not found at {model_path}")
    return load(str(model_path))


def load_initial_model(args):
    base_paths = BaseModelPaths()
    model_key = "fp16" if args.fp16 else "mlx_4bit"
    model_path = base_paths.get_model_path(model_key)
    if not model_path:
        raise FileNotFoundError(f"Initial model path for {model_key} is not configured")
    return load(model_path)


def perform_inference(model, tokenizer, sample, create_eval_prompt: Callable[[str], str]):
    if isinstance(sample, dict) and 'question' in sample:
        question = sample['question']
        eval_prompt = create_eval_prompt(question)
        response = generate(model, tokenizer, eval_prompt, temp=0.2, max_tokens=1024, verbose=True)
        print(response)
        return response
    print("Sample must be a dictionary containing a 'question' key.")
    return None


def compare_initial(args, sample):
    if args.gguf:
        if st:
            st.warning("GGUF comparison is not supported in this version.")
        return sample, None, None

    finetuned_model, finetuned_tokenizer = load_finetuned_model(args)
    initial_model, initial_tokenizer = load_initial_model(args)

    cache_limit = mx.metal.set_cache_limit(0)
    try:
        initial_response = perform_inference(initial_model, initial_tokenizer, sample, create_eval_prompt_local)
        finetuned_response = perform_inference(finetuned_model, finetuned_tokenizer, sample, create_eval_prompt_local)
    finally:
        del initial_model, initial_tokenizer
        del finetuned_model, finetuned_tokenizer
        mx.metal.set_cache_limit(cache_limit)

    return sample, initial_response, finetuned_response


def compare_anthropic(args, sample):
    import os

    import anthropic
    from dotenv import load_dotenv

    load_dotenv()

    if args.gguf:
        if st:
            st.warning("Anthropic comparison is not supported for GGUF models.")
        return sample, None, None

    finetuned_model, tokenizer = load_finetuned_model(args)

    def perform_inference_anthropic(sample):
        if isinstance(sample, dict) and 'question' in sample:
            question = sample['question']
            prompt = create_eval_prompt_anthropic(question)

            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2,
            )
            print(response.content)
            response_content = response.content[0].text
            print(response_content)
            return response_content
        print("Sample must be a dictionary containing a 'question' key.")
        return None

    comparison_sample = sample
    compared_model_response = perform_inference_anthropic(sample)
    finetuned_model_response = perform_inference(finetuned_model, tokenizer, sample, create_eval_prompt_anthropic)

    return comparison_sample, compared_model_response, finetuned_model_response
