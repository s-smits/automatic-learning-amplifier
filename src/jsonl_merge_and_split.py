# jsonl_merge_and_split.py
import json
import os
from setup_and_parse import FolderPaths


def merge_json_to_jsonl(output_file, args):
    folders = FolderPaths(args)
    # Open the output file once, outside the loop, in append mode
    with open(output_file, 'w') as f:  # Changed to 'w' to overwrite and ensure clean start
        for file in os.listdir(folders.json_data_folder):
            if file.endswith('.json'):
                with open(os.path.join(folders.json_data_folder, file), 'r') as json_file:
                    data = json.load(json_file)
                    print('loaded json file {}'.format(data))
                    for qa_pair in data:
                        question = qa_pair['question']
                        answer = qa_pair['answer']
                        validate = qa_pair.get('validate', 1)  # Default to 1 if 'validate' key is missing
                        qa_pair_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>"
                        json_line = {"text": qa_pair_text, "question": question, "answer": answer, "validate": validate}
                        json.dump(json_line, f)
                        f.write('\n')

def jsonl_split(input_jsonl_file, args):
    import pandas as pd
    import json
    import os
    from sklearn.model_selection import train_test_split
    folders = FolderPaths(args)  # Create an instance of FolderPaths
    
    # Read the JSONL file into a list of dictionaries
    with open(input_jsonl_file, 'r') as file:
        full_jsonl = [json.loads(line) for line in file]

    # Convert list of dictionaries to DataFrame
    full_df = pd.DataFrame(full_jsonl)

    # Check if DataFrame is empty
    if full_df.empty:
        print("The input JSONL file is empty. Skipping train-test split.")
        return

    # Filter out rows where validate is 0
    full_df = full_df[full_df['validate'] == 1]

    # Check if DataFrame is empty after filtering
    if full_df.empty:
        print("No valid entries to split. Skipping train-test split.")
        return

    # Drop the 'validate' column as it's not needed in the training and validation files
    full_df.drop(columns=['validate'], inplace=True)

    # Split DataFrame into training and validation sets
    train, val = train_test_split(full_df, test_size=args.val_set_size, random_state=42)

    # Save both files in finetune_data_folder
    train.to_json(os.path.join(folders.finetune_data_folder, 'train.jsonl'), orient='records', lines=True)
    val.to_json(os.path.join(folders.finetune_data_folder, 'valid.jsonl'), orient='records', lines=True)