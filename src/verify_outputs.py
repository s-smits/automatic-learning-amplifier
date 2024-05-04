from mlx_lm.utils import load, generate
from setup_and_parse import BaseModelPaths
import inflect
from json_extractor import extract_json
import json

def verify_outputs(jsonl_data, text_chunk, args):
    base_paths = BaseModelPaths()
    prompt = f"""
    <[begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a genius fact verification assistant that helps analyzing the text and outputs "validate" JSON keys.<|eot_id|><|start_header_id|>user<|end_header_id|>
    <prompt>Analyze the text corpus and the {args.question_amount} {inflect.engine().number_to_words(str(args.question_amount))} questions and answers based on the text corpus with the following structure:
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
    
    if verification_response is not None:
        for i, qa_pair in enumerate(jsonl_data):
            if i < len(verification_response):
                qa_pair['validate'] = verification_response[i]['validate']
            else:
                print(f"No validation data for QA pair index {i}")
        return True
    else:
        print("Skipping verification update due to invalid response.")
        return False