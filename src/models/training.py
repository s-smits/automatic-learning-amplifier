"""Model fine-tuning orchestration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import streamlit as st

from config import BaseModelPaths, FolderPaths


def train_model(args) -> None:
    base_paths = BaseModelPaths()
    folders = FolderPaths(args)

    train_file = folders.finetune_data_folder / "train.jsonl"
    if not train_file.exists():
        st.error("Training file not found. Please generate training data before starting fine-tuning.")
        return

    rows = sum(1 for _ in train_file.open("r"))
    if rows == 0:
        st.error("train.jsonl is empty. Aborting training.")
        return

    print(f"Number of rows in train.jsonl: {rows}")
    lr = float(args.learning_rate)
    print(f"Learning rate: {lr}")
    epochs = int(args.epochs)
    print(f"Epochs: {epochs}")
    total_iters = max(rows * epochs, 1)

    model_key = "fp16" if args.ft_type == "lora" else "mlx_4bit"
    model_path = base_paths.get_model_path(model_key)
    if not model_path:
        st.error(f"Base model path for '{model_key}' is not configured.")
        return

    adapter_path = folders.ft_folder

    def run_command(command: List[str], spinner_text: str) -> None:
        print("Running command:", " ".join(command))
        try:
            with st.spinner(spinner_text):
                subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            st.error(f"Command failed with exit code {exc.returncode}: {' '.join(command)}")
            raise

    train_command = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--train",
        "--model",
        model_path,
        "--data",
        str(folders.finetune_data_folder),
        "--batch-size",
        "4" if args.ft_type == "lora" else "1",
        "--lora-layers",
        str(args.lora_layers),
        "--iters",
        str(total_iters),
        "--save-every",
        str(rows),
        "--learning-rate",
        str(lr),
        "--adapter-path",
        str(adapter_path),
    ]

    run_command(train_command, f"Finetuning {model_path} with {args.ft_type.upper()}...")

    fuse_command = [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        model_path,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(adapter_path),
    ]

    run_command(fuse_command, "Fusing finetuned weights with the adapter...")

    if args.ft_type == "qlora" and args.gguf:
        st.info("GGUF conversion is not automated in this build. Please convert the fused model manually if needed.")
