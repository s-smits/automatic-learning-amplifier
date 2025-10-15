"""Filesystem and model path helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


class BaseModelPaths:
    """Central registry for base model identifiers used throughout the app."""

    def __init__(self) -> None:
        self._base_models = {
            "fp16": "abacusai/Llama-3-Smaug-8B",
            "mlx_4bit": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "gguf": "bartowski/Llama-3-Smaug-8B-GGUF/LLama-3-Smaug-8B-Q5_K_M.gguf",
            "mlx_4bit_long_context": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "mlx_image": "mlx-community/llava-1.5-7b-4bit",
        }

    def get_model_path(self, key: str) -> Optional[str]:
        return self._base_models.get(key)


class FolderPaths:
    """Resolve and manage runtime directories for the pipeline."""

    def __init__(self, args) -> None:
        self.base_dir = Path(__file__).resolve().parents[2]
        data_dir = self.base_dir / "data"

        self.documents_folder = data_dir / "documents"
        self.prepared_data_folder = data_dir / "data_prepared"
        self.image_folder = data_dir / "images"
        self.json_data_folder = data_dir / "qa_json"
        self.finetune_data_folder = data_dir / "data_ft"
        self.summaries_folder = data_dir / "summaries"
        self.ft_folder = self._resolve_ft_folder(args.ft_type)
        self.create_folders()

    def _resolve_ft_folder(self, ft_type: str) -> Path:
        models_dir = Path(__file__).resolve().parents[2] / "models"
        if ft_type not in {"lora", "qlora"}:
            raise ValueError(f"Unsupported ft_type '{ft_type}'. Expected 'lora' or 'qlora'.")
        return models_dir / ft_type

    def create_folders(self) -> None:
        folders = [
            self.documents_folder,
            self.prepared_data_folder,
            self.image_folder,
            self.json_data_folder,
            self.finetune_data_folder,
            self.summaries_folder,
            self.ft_folder,
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    def clear_folders(self) -> None:
        folders = [
            self.prepared_data_folder,
            self.image_folder,
            self.json_data_folder,
            self.finetune_data_folder,
            self.ft_folder,
            self.summaries_folder,
        ]
        for folder in folders:
            if folder.exists() and any(folder.iterdir()):
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

        if self.documents_folder.exists():
            for path in self.documents_folder.rglob("*"):
                if path.is_file():
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
