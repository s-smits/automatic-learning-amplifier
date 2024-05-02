
# add_summary.py
from mlx_lm.utils import generate
from setup_and_parse import BaseModelPaths
import mlx.core as mx
from mlx_lm.utils import load
import os

def add_summary(model, tokenizer, file_contents, args):
    
    subject_prompts = {
        'math': 'Summarize the key mathematical concepts, equations, problem-solving strategies, and applications covered in this content.',
        'science': 'Provide a concise overview of the scientific topics, theories, experiments, data, and conclusions presented in this material.',
        'history': 'Give a brief summary of the historical events, figures, timelines, causes, consequences, and significance discussed in this text.',
        'geography': 'Summarize the main geographical features, locations, populations, climates, resources, and human interactions described in this document.',
        'english': 'Concisely summarize the central themes, characters, plot, structure, style, and literary devices used in this English language content.',
        'art': 'Provide a brief overview of the artistic styles, techniques, media, artists, works, and interpretations covered in this material.',
        'music': 'Summarize the key musical elements, genres, composers, instruments, performances, and cultural context discussed in this content.',
        'education': 'Give a concise summary of the educational topics, learning objectives, pedagogical approaches, assessments, and resources presented in this document.',
        'computer science': 'Summarize the main computer science concepts, algorithms, programming languages, tools, applications, and challenges covered in this text.',
        'drama': 'Provide a brief overview of the dramatic works, playwrights, characters, plots, themes, staging, and critical interpretations discussed in this material.'
    }
    
    if args.add_summary in subject_prompts:
        prompt_template_phi3 = f"<|system|>\nYou are a concise summarization assistant. You answer in 10 sentences maximum.<|end|>\n<|user|>\n<prompt>{subject_prompts[args.add_summary]}</prompt><file_content>{file_contents}</file_content><|end|>\n<|assistant|>"
        print(prompt_template_phi3)
        max_tokens = 256
        summary = generate(model, tokenizer, prompt_template_phi3, max_tokens=max_tokens)
        print(summary)
        return summary
    else:
        return ''

def summarize_documents(file_contents, args, folders):
    base_paths = BaseModelPaths()
    original_limit = mx.metal.set_cache_limit(0)  # Capture the original cache limit to restore later

    print("Generating summaries...")
    model, tokenizer = load(base_paths.get_model_path('mlx_4bit_long_context'))
    summaries = []
    
    for batch_contents in file_contents:  # Process each batch passed from retrieve_synthetic
        print('Summarizing batch of files...')
        
        summary = add_summary(model, tokenizer, batch_contents, args)
        if summary:
            summaries.append(summary)
            output_filename = f'batch_summary_{len(summaries)}.txt'
            output_path = os.path.join(folders.finetune_data_folder, output_filename)
            with open(output_path, 'w') as txt_file:
                txt_file.write(summary)
            print('Summary added to', output_path)
        else:
            print('Failed to generate summary for batch:', batch_contents)
    
    del model, tokenizer
    mx.metal.set_cache_limit(original_limit)  # Restore the original cache limit
    
    return "\n".join(summaries)