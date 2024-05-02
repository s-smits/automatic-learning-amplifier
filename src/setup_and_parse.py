# setup_and_parse.py (refactored)

import argparse
import os
import logging
import streamlit as st
import shutil

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/log.txt', level=logging.INFO)

class BaseModelPaths:
    def __init__(self):
        self.base_models = {
            'fp16': 'abacusai/Llama-3-Smaug-8B',
            'mlx_4bit': 'mlx-community/Meta-Llama-3-8B-Instruct-4bit',
            'gguf': 'bartowski/Llama-3-Smaug-8B-GGUF/LLama-3-Smaug-8B-Q5_K_M.gguf',
            'mlx_4bit_long_context': 'mlx-community/Phi-3-mini-128k-instruct-4bit',
            'mlx_image': 'mlx-community/llava-1.5-7b-4bit'
        }

    def get_model_path(self, key):
        return self.base_models.get(key, None)

class FolderPaths:
    def __init__(self, args):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from the src directory
        self.documents_folder = os.path.join(self.base_dir, "data", "documents")
        self.prepared_data_folder = os.path.join(self.base_dir, "data", "data_prepared")
        self.image_folder = os.path.join(self.base_dir, "data", "images")
        self.json_data_folder = os.path.join(self.base_dir, "data", "qa_json")
        self.finetune_data_folder = os.path.join(self.base_dir, "data", "data_ft")
        if args.ft_type == "lora":
            self.ft_folder = os.path.join(self.base_dir, "models", "lora")
        elif args.ft_type == "qlora":
            self.ft_folder = os.path.join(self.base_dir, "models", "qlora")
        self.create_folders()

    def create_folders(self):
        folders = [
            self.prepared_data_folder, self.image_folder,
            self.json_data_folder, self.finetune_data_folder, self.ft_folder
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def clear_folders(self):
        folders = [
            self.prepared_data_folder, self.image_folder,
            self.json_data_folder, self.finetune_data_folder, self.ft_folder
        ]
        for folder in folders:
            # Check if the folder exists and is not empty
            if os.path.exists(folder) and os.listdir(folder):
                try:
                    shutil.rmtree(folder)
                except FileNotFoundError:
                    # If the file is already deleted, ignore the error
                    pass
                os.makedirs(folder)

        # Clear files inside the documents folder, but don't delete the folder itself
        if os.path.exists(self.documents_folder) and os.listdir(self.documents_folder):
            for root, dirs, files in os.walk(self.documents_folder):
                for file in files:
                    try:
                        os.unlink(os.path.join(root, file))
                    except FileNotFoundError:
                        # If the file is already deleted, ignore the error
                        pass

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General arguments
    parser.add_argument('--retries', type=int, default=3, help="Number of retries for each file")
    parser.add_argument('--word_limit', type=int, default=1000, help="Word limit for each file")
    parser.add_argument('--question_amount', type=int, default=5, help="Amount of synthetic questions and answers to generate")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for the finetuning process")
    parser.add_argument('--images', action='store_true', help="Generate captions of the images in the documents (can be very slow depending on the images provided)")
    parser.add_argument('--focus', type=str, choices=['processes', 'knowledge', 'formulas'], help="Focus on specific goals for LLM to retrieve synthetic data")
    parser.add_argument('--local', action='store_true', default=True, help="Use local model for inference")
    parser.add_argument('--openrouter', action='store_true', help="Use openrouter for inference")
    parser.add_argument('--claude', action='store_true', help="Use claude for inference")
    parser.add_argument('--optimize', action="store_true", help="Activate optimization mode.")
    parser.add_argument('--verify', action="store_true", help="Verify the questions and answers.")
    parser.add_argument('--val_set_size', type=float, default=0.05, help="Test size for splitting the data into training and validation sets")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs for the finetuning process")
    parser.add_argument('--ft_type', type=str, default="qlora", choices=["lora", "qlora"], help="Finetune the model with LoRA or QLoRA")
    parser.add_argument('--lora_layers', type=int, default=16, help="Number of LoRA layers to use")
    parser.add_argument('--q4', action='store_true', default=False, help="Run inference in Q4")
    parser.add_argument('--fp16', action='store_true', default=False, help="Run inference in FP16")
    parser.add_argument('--gguf', action='store_true', default=False, help="Converting and infering the model with GGUF")
    parser.add_argument('--compare', action='store_true', default=True, help="Compare the performance of the finetuned model with the non-finetuned model")
    parser.add_argument('--deploy', action='store_true', default=False, help="Deploy the finetuned model")
    parser.add_argument('--starting_index', type=int, default=0, help="Starting index for the data files")
    parser.add_argument('--ending_index', type=int, default=-1, help="Ending index for the data files")
    parser.add_argument('--overlap', type=float, default=0.1, help="Overlap between the batches for summarizing documents.")
    # Summary arguments
    summary_group = parser.add_argument_group('Summary Arguments')
    summary_choices = ['math', 'science', 'history', 'geography', 'english', 'art', 'music', 'education', 'computer science', 'drama']
    summary_group.add_argument("--add_summary", choices=summary_choices, help="Add a short summary to each prompt from every 10 document files.")
    summary_group.add_argument("--summary_batch_size", type=int, default=5, help="Batch size for summarizing documents.")

    args = parser.parse_args()

    # Check for mutually exclusive arguments
    if args.local and (args.openrouter or args.claude):
        parser.error("--local cannot be used with --openrouter or --claude")
    if args.openrouter and args.claude:
        parser.error("--openrouter and --claude are mutually exclusive")

    # Adjust summary batch size
    if args.add_summary:
        if args.summary_batch_size > args.ending_index - args.starting_index:
            st.error(f"--summary_batch_size cannot be higher than the number of files to be processed, setting --summary_batch_size to {args.ending_index - args.starting_index}")
            args.summary_batch_size = args.ending_index - args.starting_index

    return args, summary_choices

def initialize_setup():
    setup_logging()
    args = parse_arguments()
    return args