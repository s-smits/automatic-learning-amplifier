# retrieve_synthetic.py
from setup_and_parse import BaseModelPaths, FolderPaths
from summarize_documents import summarize_documents

import requests
import json
from json_extractor import extract_json
from mlx_lm.utils import load, generate
import os
import mlx.core as mx
from stqdm import stqdm
from verify_outputs import verify_outputs
import streamlit as st

def get_focus_choice(focus_arg):
    '''This function returns a specific focus description based on the provided focus argument.'''
    focus_descriptions = {
        'processes': ", focus on specific processes",
        'knowledge': ", focus on specific knowledge not known widely",
        'formulas': ", focus on obtaining formulas in Latex format"
    }
    return focus_descriptions.get(focus_arg, "Invalid focus argument")

def infere_local(prompt_template, file_contents, args):
    base_paths = BaseModelPaths()
    folders = FolderPaths(args)
    original_limit = mx.metal.set_cache_limit(0)  # Capture the original cache limit to restore later    
    # Load model with progress bar
    # Initialize session state for showing the message
    if 'show_message' not in st.session_state:
        st.session_state.show_message = True
    with st.spinner("Loading model..."):
        model, tokenizer = load(base_paths.get_model_path('mlx_4bit'))  # Load once if possible
    
    # FUNCTION 1
    summary = None
    if args.add_summary:
        print("Generating summaries...")
        for i in range(0, len(file_contents), args.summary_batch_size):
            batch_contents = file_contents[i:i+args.summary_batch_size]
            print(f'Summarizing files {i+1} to {i+args.summary_batch_size}...')
            summary = summarize_documents(model, tokenizer, batch_contents, args)
            if summary:
                output_filename = f'file{i+1}_{i+args.summary_batch_size}.summary.txt'
                output_path = os.path.join(folders.finetune_data_folder, output_filename)
                with open(output_path, 'w') as txt_file:
                    txt_file.write(summary)
                print('Summary added to', output_path)
            else:
                print('Failed to generate summary for files:', batch_contents)

    tqdm_bar = stqdm(file_contents[args.starting_index:args.ending_index], desc=f"Generating {args.question_amount} questions and answer pairs for {len(file_contents[args.starting_index:args.ending_index])} chunks")

    display_placeholder = st.empty()

    for i, file_content in enumerate(tqdm_bar):
        print('Processing file content:', file_content)
        
        prompt = f'{prompt_template} {"<extra_context>" + summary + "</extra_context>" if args.add_summary else ""}\n<text_corpus>{file_content}</text_corpus><|eot_id|><|start_header_id|>assistant<|end_header_id|><json_output>'
        print('prompt:', prompt)
        output = generate(model, tokenizer, prompt, temp=0.2, max_tokens=1024, verbose=True)
        jsonl_data = extract_json(output)

        if jsonl_data:
            output_filename = f'chunk_{i+args.starting_index+1}_{args.word_limit}.json'
            output_path = os.path.join(folders.json_data_folder, output_filename)
            with open(output_path, 'w') as json_file:
                json.dump(jsonl_data, json_file)
            print('Added JSON to', output_path)
            
            display_placeholder.empty()
            with display_placeholder.container():
                st.subheader(f'Chunk {i+args.starting_index+1} Questions and Answers')
                for index, item in enumerate(jsonl_data):
                    # Display 'question' and 'answer' fields using st.container for better control over styling
                    with st.container():
                        # Assign unique keys to each text_area to avoid DuplicateWidgetID error
                        answer_key = f"answer_{i+args.starting_index+1}_{index+1}"
                        st.text(f"Question {index+1}: {item['question']}")
                        st.text_area(f"Answer {index+1}", value=item['answer'], height=80, disabled=True, key=answer_key)

            # FUNCTION 3 
            if args.verify:
                if verify_outputs(args, jsonl_data, file_content):
                    print("Comparison successful, updating JSON file.")
                    with open(output_path, 'w') as json_file:
                        json.dump(jsonl_data, json_file)
                    print(f"Updated JSON data written to {output_path}")
                else:
                    print("Comparison failed, not updating JSON file.")
        else:
            print(f'Failed to extract JSON for file: {i+args.starting_index+1}')
    
    display_placeholder.empty()    
    del model, tokenizer # Cleanup after all files are processed
    mx.metal.set_cache_limit(original_limit)  # Restore the original cache limit
        
def infere_claude(prompt, file):
    import anthropic

    with open(os.path.join(file), 'r') as f:
        file_content = f.read()
    
    prompt = f"""{prompt} {file_content}
    """
    
    client = anthropic.Anthropic(
        api_key = os.getenv("ANTHROPIC_API_KEY"),
    )
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1536,
        # temperature=0.6, old choice
        temperature=0.3,
    )
        
    response_content = response.content
    print(response_content)

    # Extract the JSON string from the ContentBlock object
    content_block = response_content[0]
    json_string = content_block.text

    # Use the extract_json function to extract the JSON component
    data_formatted = extract_json(json_string)

    def write_questions(data_formatted, file):
        output_file = os.path.splitext(file)[0] + ".json"
        output_path = os.path.join("qa_json", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(data_formatted)
            print(f"Questions and answers written to {output_file}")

    write_questions(data_formatted, file)

def infere_openrouter(prompt, file):
    import os
    from dotenv import load_dotenv
    load_dotenv()

    with open(os.path.join(file), 'r') as f:
        file_content = f.read()

    prompt = f"""{prompt} {file_content}"""
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": None,
            "X-Title": None,
        },
        data=json.dumps({
            "model": "mistralai/mixtral-8x7b-instruct:nitro",
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    response_content = response.json()
    print(response_content)
    return response_content