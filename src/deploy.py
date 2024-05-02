import subprocess
import os
import mlx.core as mx
from setup_and_parse import BaseModelPaths, FolderPaths
import sys
import logging

def deploy_models(args):
    logging.basicConfig(level=logging.INFO)
    base_paths = BaseModelPaths()
    folder_paths = FolderPaths(args) 
    
    initial_memory = mx.metal.set_cache_limit(0)
    repo_path = os.path.join(os.getcwd(), "mlx-ui")
    repo_url = "https://github.com/da-z/mlx-ui.git"

    def clone_and_setup_mlx_ui():
        if os.path.exists(repo_path) and os.listdir(repo_path):
            logging.info(f"Directory {repo_path} already exists and is not empty.")
            return
        
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
        requirements_path = os.path.join(repo_path, "requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
        logging.info("MLX-UI cloned and dependencies installed.")

    def create_mymodels_txt():
        models_txt_path = os.path.join(repo_path, "mymodels.txt")
        with open(models_txt_path, "w") as f:
            f.write(f"{folder_paths.ft_folder} | Improved Finetuned Model (Q4)\n")
            if args.fp16:
                f.write(f"{base_paths.get_model_path('fp16')} | Initial Model (FP16)\n")
            elif args.q4:
                f.write(f"{base_paths.get_model_path('mlx_4bit')} | Initial Model (Q4)\n")
        logging.info("Models configuration written to mymodels.txt.")

    def run_mlx_ui():
        app_path = os.path.join(repo_path, "app.py")
        if not os.path.exists(app_path):
            logging.error(f"Failed to find app.py at {app_path}.")
            return
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--", "--models", os.path.join(repo_path, "mymodels.txt")], check=True)
        logging.info("Streamlit application is running.")

    clone_and_setup_mlx_ui()
    create_mymodels_txt()
    run_mlx_ui()
    mx.metal.set_cache_limit(initial_memory)