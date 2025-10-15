"""Argument parsing and application setup helpers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

try:
    import streamlit as st
except ModuleNotFoundError:  # Streamlit is optional outside the UI context
    st = None

from .paths import BaseModelPaths, FolderPaths


SUMMARY_CHOICES = (
    "math",
    "science",
    "history",
    "geography",
    "english",
    "art",
    "music",
    "education",
    "computer science",
    "drama",
)


def setup_logging() -> None:
    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(logs_dir / "log.txt"), level=logging.INFO)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, Sequence[str]]:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for each file")
    parser.add_argument("--word_limit", type=int, default=1000, help="Word limit for each file")
    parser.add_argument("--question_amount", type=int, default=5, help="Number of questions and answers to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the finetuning process")
    parser.add_argument("--images", action="store_true", help="Generate captions for images in documents")
    parser.add_argument("--focus", type=str, choices=["processes", "knowledge", "formulas"], help="Focus the synthetic data on a specific goal")
    parser.add_argument("--local", action="store_true", default=True, help="Use a local model for inference")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter for inference")
    parser.add_argument("--claude", action="store_true", help="Use Claude for inference")
    parser.add_argument("--optimize", action="store_true", help="Activate optimization mode")
    parser.add_argument("--verify", action="store_true", help="Verify generated questions and answers")
    parser.add_argument("--val_set_size", type=float, default=0.05, help="Validation split size for training data")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--ft_type", type=str, default="qlora", choices=["lora", "qlora"], help="Fine-tuning technique to use")
    parser.add_argument("--lora_layers", type=int, default=16, help="Number of LoRA layers to enable")
    parser.add_argument("--q4", action="store_true", default=False, help="Run inference in Q4")
    parser.add_argument("--fp16", action="store_true", default=False, help="Run inference in FP16")
    parser.add_argument("--gguf", action="store_true", default=False, help="Convert and infer the model with GGUF")
    parser.add_argument("--compare", action="store_true", default=True, help="Compare fine-tuned and base model outputs")
    parser.add_argument("--deploy", action="store_true", default=False, help="Deploy the fine-tuned model")
    parser.add_argument("--starting_index", type=int, default=0, help="Starting index for processed files")
    parser.add_argument("--ending_index", type=int, default=-1, help="Ending index for processed files (-1 for all)")
    parser.add_argument("--overlap", type=float, default=0.1, help="Chunk overlap ratio between 0 and 1")

    summary_group = parser.add_argument_group("Summary Arguments")
    summary_group.add_argument("--add_summary", choices=SUMMARY_CHOICES, help="Add subject-focused summaries")
    summary_group.add_argument("--summary_batch_size", type=int, default=2, help="Number of chunks per summary")

    args, _ = parser.parse_known_args(args=argv)

    if args.local and (args.openrouter or args.claude):
        raise ValueError("--local cannot be combined with --openrouter or --claude")
    if args.openrouter and args.claude:
        raise ValueError("--openrouter and --claude are mutually exclusive")

    if args.ending_index == -1:
        args.ending_index = None

    if args.ending_index is not None and args.ending_index <= args.starting_index:
        raise ValueError("--ending_index must be greater than --starting_index or -1 for all files")

    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be between 0 (inclusive) and 1 (exclusive)")

    if args.add_summary and args.summary_batch_size < 1:
        raise ValueError("--summary_batch_size must be at least 1 when --add_summary is used")

    return args, SUMMARY_CHOICES


def initialize_setup(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, Sequence[str]]:
    setup_logging()
    args, summary_choices = parse_arguments(argv=argv)
    return args, summary_choices
