import subprocess
import sys
import os
from setup_and_parse import BaseModelPaths, FolderPaths
from stqdm import stqdm
import streamlit as st

def train_model(args):
    base_paths = BaseModelPaths()
    folders = FolderPaths(args)
    
    train_file = os.path.join(folders.finetune_data_folder, "train.jsonl")
    if os.path.exists(train_file):
        with open(train_file, 'r') as file:
            rows = sum(1 for _ in file)
            print(f"Number of rows in train.jsonl: {rows}")
    else:
        print("An error occurred while processing the train.jsonl file.")
    lr = float(args.learning_rate)
    print(f"Learning rate: {lr}")
    epochs = int(args.epochs)
    print(f"Epochs: {epochs}")
    
    if args.ft_type == "lora":
        print(f"Finetuning {base_paths.get_model_path('fp16')} using LoRA...")
        command = [
            sys.executable, 
            "-m", "mlx_lm.lora", 
            "--train", 
            "--model", base_paths.get_model_path('fp16'),
            "--data", str(folders.finetune_data_folder),
            "--batch-size", "4",
            "--lora-layers", str(args.lora_layers), 
            "--iters", str(rows*epochs), 
            "--save-every", str(rows), 
            "--learning-rate", str(lr), 
            "--adapter-path", str(folders.ft_folder)
        ]
        subprocess.run(command, check=True)
        print("Fusing the model with the adapter...")
        command = [
            sys.executable,
            "-m", "mlx_lm.fuse", 
            "--model", base_paths.get_model_path('fp16'),
            "--adapter-path", folders.lora_folder, 
            "--save-path", folders.lora_folder
        ]
        subprocess.run(command, check=True)
    elif args.ft_type == "qlora":
        print(f"Finetuning {base_paths.get_model_path('mlx_4bit')} using QLoRA...")
        command = [
            sys.executable, 
            "-m", "mlx_lm.lora", 
            "--train", 
            "--model", base_paths.get_model_path('mlx_4bit'),
            "--data", folders.finetune_data_folder,
            "--batch-size", "1",
            "--lora-layers", str(args.lora_layers), 
            "--iters", str(rows*epochs), 
            "--save-every", str(rows), 
            "--learning-rate", str(lr), 
            "--adapter-path", str(folders.ft_folder)
        ]
        with st.spinner(f"Finetuning {base_paths.get_model_path('mlx_4bit')} with QLoRA..."):
            subprocess.run(command, check=True)
        command = [
            sys.executable, 
            "-m", "mlx_lm.fuse", 
            "--model", base_paths.get_model_path('mlx_4bit'), 
            "--adapter-path", folders.ft_folder, 
            "--save-path", folders.ft_folder
        ]
        with st.spinner("Fusing finetuned model with adapter..."):
            subprocess.run(command, check=True)
        
        if args.gguf:
            print("Converting the model to gguf format...")
            if not os.path.exists("llama.cpp"):
                print("LLama.cpp not found. Installing LLama.cpp and ensuring it is in the correct folder.")
                command = [
                    "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
                ]
                subprocess.run(command, check=True)
                os.chdir("llama.cpp")
                model_type = "lora" if args.ft_type == "lora" else "qlora"
                command = [
                    sys.executable,
                    "convert-hf-to-gguf.py", os.path.join(folders.get_model_path(model_type)),
                    "--outtype", "f16"
                ]
                subprocess.run(command, check=True)
                print("Quantizing the model to Q5_K_M format for almost no PPL loss compared to FP16...")
                command = [
                    "./quantize", os.path.join(folders.get_model_path(model_type), "ggml-model-f16.gguf"),
                    os.path.join(folders.get_model_path(model_type), "ggml-model-q5_K_M.bin"),
                    "Q5_K_M"
                ]
            subprocess.run(command, check=True)
            os.chdir("..")
