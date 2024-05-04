import os
import inflect
from jsonl_merge_and_split import jsonl_split, merge_json_to_jsonl
import json
from json_extractor import extract_json
from mlx_lm.utils import load, generate
import os
import mlx.core as mx
from stqdm import stqdm
from verify_outputs import verify_outputs
import streamlit as st
from setup_and_parse import BaseModelPaths, FolderPaths

def get_focus_choice(focus_arg):
    '''This function returns a specific focus description based on the provided focus argument.'''
    focus_descriptions = {
        'processes': ", focus on specific processes",
        'knowledge': ", focus on specific knowledge not known widely",
        'formulas': ", focus on obtaining formulas in Latex format"
    }
    return focus_descriptions.get(focus_arg, "Invalid focus argument")

def generate_questions(args, folders, text_chunks, summaries):
    amount_of_questions = args.question_amount
    stringed_number = inflect.engine().number_to_words(str(amount_of_questions))
    focus_choice = get_focus_choice(args.focus) if args.focus else ''
    
    if args.local:
        base_paths = BaseModelPaths()
        folders = FolderPaths(args)
        original_limit = mx.metal.set_cache_limit(0)  # Capture the original cache limit to restore later    
        # Load model with progress bar
        # Initialize session state for showing the message
        if 'show_message' not in st.session_state:
            st.session_state.show_message = True
        
        with st.spinner("Loading model..."):
            model, tokenizer = load(base_paths.get_model_path('mlx_4bit'))  # Load once if possible

        tqdm_bar = stqdm(text_chunks[args.starting_index:args.ending_index], desc=f"Generating {args.question_amount} questions and answer pairs for {len(text_chunks[args.starting_index:args.ending_index])} chunks")
        display_placeholder = st.empty()

        for i, text_chunk in enumerate(tqdm_bar):
            if args.add_summary:
                summary_index = (args.starting_index + i) // args.summary_batch_size + 1
                summary_filename = f'summary_batch_{summary_index}.txt'
                summary_path = os.path.join(folders.summaries_folder, summary_filename)
                with open(summary_path, 'r') as file:
                    corresponding_summary = file.read()
                print('corresponding_summary:', corresponding_summary)

            # Convert corresponding_summary to string from array
            
            # Prepare the prompt for each content item
            prompt = f'''<[begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a brilliant curious assistant who generates structured JSON outputs.
<|eot_id|><|start_header_id|>user<|end_header_id|>
<prompt>Analyze the text corpus and extract the text to a JSON file with {amount_of_questions} {stringed_number} extensive, complex, diversified questions and answers {focus_choice} with the following structure:
{{
    "text": [
        {{
            "question": "Extensive question.",
            "answer": "Extensive answer: Sentence 1. Sentence 2. Sentence 3. If needed, add more sentences."
        }},
        ...
        {{
            "question": "... ",
            "answer": "... "
        }}
    ]
}}
'''
            if args.add_summary:
                prompt += f'''<context>Here is a summary to give extra context: {corresponding_summary}</context>\n'''

            prompt += f'''Don't generate code or another JSON after it, only output one 'text' JSON structure with {amount_of_questions} {stringed_number} questions and answers.
            <text_corpus>{text_chunk}</text_corpus>
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
            
            max_tokens = args.question_amount*128
                        
            print('Prompt:', prompt)
            
            output = generate(model, tokenizer, prompt, temp=0.2, max_tokens=max_tokens, verbose=True)
            jsonl_data = extract_json(output)

            if jsonl_data:
                output_filename = f'chunk_{i+args.starting_index}_{args.word_limit}.json'
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
                            answer_key = f"answer_{i+args.starting_index}_{index+1}"
                            st.text(f"Question {index+1}: {item['question']}")
                            st.text_area(f"Answer {index+1}", value=item['answer'], height=80, disabled=True, key=answer_key)

                # FUNCTION 3 
                if args.verify:
                    if verify_outputs(jsonl_data, text_chunks, args):
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
    
        output_jsonl_file = os.path.join(folders.finetune_data_folder, 'output.jsonl')
        merge_json_to_jsonl(output_jsonl_file, args)
        jsonl_split(output_jsonl_file, args)


# elif args.claude:
#     infere_claude(prompt_template, text_chunks, summary_text, args)
# elif args.openrouter:
#     infere_openrouter(prompt_template, text_chunks, summary_text, args)
# else:
#     raise ValueError("Please specify either --local, --openrouter or --claude")
