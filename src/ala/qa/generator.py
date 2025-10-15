"""Question generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import inflect
import mlx.core as mx
import streamlit as st
from mlx_lm.utils import generate, load
from stqdm import stqdm

from ala.config import BaseModelPaths
from ala.utils import extract_json, jsonl_split, merge_json_to_jsonl

from .verifier import verify_outputs


def get_focus_choice(focus_arg: str | None) -> str:
    focus_descriptions = {
        'processes': ", focus on specific processes",
        'knowledge': ", focus on specific knowledge not known widely",
        'formulas': ", focus on obtaining formulas in Latex format",
    }
    return focus_descriptions.get(focus_arg, "")


def generate_questions(args, folders, text_chunks, summaries):
    amount_of_questions = args.question_amount
    number_engine = inflect.engine()
    stringed_number = number_engine.number_to_words(amount_of_questions)
    focus_choice = get_focus_choice(args.focus)

    if not args.local:
        raise ValueError("Only local generation is supported in this build.")

    base_paths = BaseModelPaths()
    original_limit = mx.metal.set_cache_limit(0)
    target_chunks = text_chunks[args.starting_index:args.ending_index]

    if 'show_message' not in st.session_state:
        st.session_state.show_message = True

    with st.spinner("Loading model..."):
        model, tokenizer = load(base_paths.get_model_path('mlx_4bit'))

    generated_files: list[Path] = []
    tqdm_bar = stqdm(
        target_chunks,
        desc=f"Generating {args.question_amount} questions and answer pairs for {len(target_chunks)} chunks",
    )
    display_placeholder = st.empty()

    try:
        for i, text_chunk in enumerate(tqdm_bar):
            corresponding_summary = ""
            if args.add_summary:
                summary_index = (args.starting_index + i) // args.summary_batch_size + 1
                summary_filename = f'summary_batch_{summary_index}.txt'
                summary_path = folders.summaries_folder / summary_filename
                try:
                    corresponding_summary = summary_path.read_text()
                    print('corresponding_summary:', corresponding_summary)
                except FileNotFoundError:
                    st.warning(
                        f"Summary file {summary_filename} not found. Skipping summary context for this chunk."
                    )
                    corresponding_summary = ""

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
            if args.add_summary and corresponding_summary:
                prompt += f'''<context>Here is a summary to give extra context: {corresponding_summary}</context>\n'''

            prompt += f'''Don't generate code or another JSON after it, only output one 'text' JSON structure with {amount_of_questions} {stringed_number} questions and answers.
                <text_corpus>{text_chunk}</text_corpus>
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                '''

            max_tokens = args.question_amount * 128

            print('Prompt:', prompt)

            output = generate(model, tokenizer, prompt, temp=0.2, max_tokens=max_tokens, verbose=True)
            jsonl_data = extract_json(output)

            if jsonl_data:
                output_filename = f'chunk_{i + args.starting_index}_{args.word_limit}.json'
                output_path = folders.json_data_folder / output_filename
                output_path.write_text(json.dumps(jsonl_data))
                print('Added JSON to', output_path)
                generated_files.append(output_path)

                display_placeholder.empty()
                with display_placeholder.container():
                    st.subheader(f'Chunk {i + args.starting_index + 1} Questions and Answers')
                    for index, item in enumerate(jsonl_data):
                        with st.container():
                            answer_key = f"answer_{i + args.starting_index}_{index + 1}"
                            st.text(f"Question {index + 1}: {item['question']}")
                            st.text_area(
                                f"Answer {index + 1}",
                                value=item['answer'],
                                height=80,
                                disabled=True,
                                key=answer_key,
                            )

                if args.verify:
                    if verify_outputs(jsonl_data, text_chunk, args):
                        print("Comparison successful, updating JSON file.")
                        output_path.write_text(json.dumps(jsonl_data))
                        print(f"Updated JSON data written to {output_path}")
                    else:
                        print("Comparison failed, not updating JSON file.")
            else:
                print(f'Failed to extract JSON for file: {i + args.starting_index + 1}')
    finally:
        display_placeholder.empty()
        del model, tokenizer
        mx.metal.set_cache_limit(original_limit)

    if generated_files:
        output_jsonl_file = folders.finetune_data_folder / 'output.jsonl'
        merge_json_to_jsonl(output_jsonl_file, args)
        jsonl_split(output_jsonl_file, args)
    else:
        st.error("No question/answer files were generated. Please review the logs for details.")
