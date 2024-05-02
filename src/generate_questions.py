import os
import inflect
from retrieve_synthetic import infere_local, infere_claude, infere_openrouter, get_focus_choice
from jsonl_merge_and_split import jsonl_split, merge_json_to_jsonl

def generate_questions(args, folders, file_contents):
    amount_of_questions = args.question_amount
    stringed_number = inflect.engine().number_to_words(str(amount_of_questions))
    focus_choice = get_focus_choice(args.focus) if args.focus else ''
    prompt_template = f'''<[begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a brilliant curious assistant who generates structured JSON outputs.<|eot_id|><|start_header_id|>user<|end_header_id|>
    <prompt>Analyze the text corpus and extract the text to a JSON file with {amount_of_questions} {stringed_number} extensive, complex, diversified questions and answers{focus_choice} with the following structure:
    {{
        "text": [
            {{@
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
    Don't generate code or another JSON after it, only output one 'text' JSON structure with {amount_of_questions} {stringed_number} questions and answers.</prompt>
    '''
    if args.local:
        infere_local(prompt_template, file_contents, args)
    elif args.claude:
        infere_claude(prompt_template, file_contents, args)
    elif args.openrouter:
        infere_openrouter(prompt_template, file_contents, args)
    else:
        raise ValueError("Please specify either --local, --openrouter or --claude")
    
    output_jsonl_file = os.path.join(folders.finetune_data_folder, 'output.jsonl')
    merge_json_to_jsonl(output_jsonl_file, args)
    jsonl_split(output_jsonl_file, args)