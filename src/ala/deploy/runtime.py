"""Deployment helper for serving finetuned models via mlx-ui."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

from ala.config import BaseModelPaths, FolderPaths


def deploy_models(args) -> None:
    logging.basicConfig(level=logging.INFO)
    base_paths = BaseModelPaths()
    folder_paths = FolderPaths(args)

    initial_memory = mx.metal.set_cache_limit(0)
    repo_path = Path.cwd() / "mlx-ui"
    repo_url = "https://github.com/da-z/mlx-ui.git"

    def clone_and_setup_mlx_ui() -> None:
        if repo_path.exists() and any(repo_path.iterdir()):
            logging.info("mlx-ui directory already exists and is not empty.")
            return

        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
        requirements_path = repo_path / "requirements.txt"
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)
        logging.info("MLX-UI cloned and dependencies installed.")

    def create_mymodels_txt() -> None:
        models_txt_path = repo_path / "mymodels.txt"
        with models_txt_path.open("w") as f:
            f.write(f"{folder_paths.ft_folder} | Improved Finetuned Model (Q4)\n")
            if args.fp16:
                f.write(f"{base_paths.get_model_path('fp16')} | Initial Model (FP16)\n")
            elif args.q4:
                f.write(f"{base_paths.get_model_path('mlx_4bit')} | Initial Model (Q4)\n")
        logging.info("Models configuration written to mymodels.txt.")

    def run_mlx_ui() -> None:
        app_path = repo_path / "app.py"
        if not app_path.exists():
            logging.error("Failed to find app.py at %s.", app_path)
            return
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--",
                "--models",
                str(repo_path / "mymodels.txt"),
            ],
            check=True,
        )
        logging.info("Streamlit application is running.")

    if not folder_paths.ft_folder.exists():
        logging.error("Finetuned model folder not found. Please train and fuse a model before deployment.")
        mx.metal.set_cache_limit(initial_memory)
        return

    try:
        clone_and_setup_mlx_ui()
        create_mymodels_txt()
        run_mlx_ui()
    finally:
        mx.metal.set_cache_limit(initial_memory)
