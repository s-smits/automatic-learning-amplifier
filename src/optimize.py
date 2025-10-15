import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from mlx_lm.utils import generate, load

from config import BaseModelPaths
from utils import extract_json

def optimize(prompt, text_chunks):
    base_paths = BaseModelPaths()
    json_file_path = Path('optimization_data.json')
    graph_folder = Path('graphs')
    graph_folder.mkdir(parents=True, exist_ok=True)
    retries = 3  # Define the number of retries as needed

    def load_or_initialize_json(filepath):
        if Path(filepath).exists():
            with Path(filepath).open('r') as file:
                return json.load(file)
        else:
            return []

    def save_json(data, filepath):
        with Path(filepath).open('w') as file:
            json.dump(data, file, indent=4)

    def visualize_data_distribution(json_data_list, graph_folder):
        temperatures = [entry["temperature"] for entry in json_data_list if not entry["retry"]]
        max_tokens = [entry["max_tokens"] for entry in json_data_list if not entry["retry"]]
        word_counts = [entry["words_amount"] for entry in json_data_list if not entry["retry"]]
        retries_counts = [entry["attempt"] for entry in json_data_list if entry["retry"]]

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.hist(word_counts, bins=20, color='blue', alpha=0.7)
        plt.title('Word Counts Distribution')
        plt.xlabel('Word count')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.scatter(retries_counts, word_counts[:len(retries_counts)], alpha=0.7)
        plt.title('Retry Attempts vs. Word Counts')
        plt.xlabel('Retry Attempts')
        plt.ylabel('Word Count')

        plt.tight_layout()
        plt.savefig(graph_folder / 'data_distribution.png')
        plt.show()

    json_data_list = load_or_initialize_json(json_file_path)

    model, tokenizer = load(base_paths.get_model_path('mlx_4bit'))
    
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    max_tokens_array = [512, 768, 1024]
    for j in temperatures:
        for h in max_tokens_array:
            attempt = 0
            while attempt < retries:
                for text_chunk in text_chunks:
                    combined_prompt = f'{prompt}<text_corpus>{text_chunk}</text_corpus><|eot_id|><|start_header_id|>assistant<|end_header_id|><json_output>'
                    response_content = generate(model, tokenizer, combined_prompt, temp=j, max_tokens=h, verbose=True)
                    jsonl_data = extract_json(response_content)
                    if jsonl_data:
                        print('JSON outputs:', jsonl_data)
                        words = count_words_in_answers(jsonl_data)
                        logging.info(f"Processed with temperature {j} and max_tokens {h} with {words} words")
                        entry = {
                            "temperature": j,
                            "max_tokens": h,
                            "words_amount": words,
                            "retry": False,
                        }
                        json_data_list.append(entry)
                        save_json(json_data_list, json_file_path)
                        break
                    else:
                        attempt += 1
                        logging.info(f"Retry {attempt}/{retries} with temperature {j} and max_tokens {h}")
                        entry = {
                            "temperature": j,
                            "max_tokens": h,
                            "retry": True,
                            "attempt": attempt,
                            "retries": retries,
                        }
                        json_data_list.append(entry)
                        save_json(json_data_list, json_file_path)

    visualize_data_distribution(json_data_list, graph_folder)

def count_words_in_answers(jsonl_data):
    """ Count words in the 'answer' field of each item in jsonl_data. """
    return sum(len(item["answer"].split()) for item in jsonl_data if "answer" in item)