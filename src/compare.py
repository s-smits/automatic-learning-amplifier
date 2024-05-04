import os
from mlx_lm.utils import load, generate
from setup_and_parse import FolderPaths
import mlx.core as mx
from ggml_inference import ggml_inference

# Define helper functions
def create_eval_prompt_local(question):
    return f'''<[begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|><prompt>{question}</prompt><|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

def create_eval_prompt_anthropic(question):
    return f'''<system>You are a helpful assistant.</system><prompt>{question}</prompt>'''

def load_finetuned_model(args):
    folders = FolderPaths(args)
    if args.ft_type == "lora":
        model_path = folders.ft_folder
    elif args.ft_type == "qlora":
        model_path = folders.ft_folder
    print('Loaded model for finetuned model: ', model_path)
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Finetuned model file not found at {model_path}")
    return load(model_path)

def perform_inference(model, tokenizer, sample, create_eval_prompt, description):
    if isinstance(sample, dict) and 'question' in sample:
        question = sample['question']
        eval_prompt = create_eval_prompt(question)
        response = generate(model, tokenizer, eval_prompt, temp=0.2, max_tokens=1024, verbose=True)
        print(response)
        return response
    else:
        print("Sample must be a dictionary containing a 'question' key.")
        return None

def compare_initial(args, sample):
    folders = FolderPaths(args)
    
    # Load finetuned model
    finetuned_model, tokenizer = load_finetuned_model(args)
    
    # GGUF model inference
    if args.gguf:
        initial_model_response = ggml_inference(finetuned_model, tokenizer, sample, "initial GGUF model")
        finetuned_model_file = os.path.join(folders.ft_folder, "ggml-model-q5_K_M.bin")
        finetuned_model_response = ggml_inference(finetuned_model, tokenizer, sample, finetuned_model_file, "finetuned GGUF model")
    else:
        # MLX inference
        initial_memory = mx.metal.set_cache_limit(0)
        initial_model_response = perform_inference(finetuned_model, tokenizer, sample, create_eval_prompt_local, "initial model")
        finetuned_model_response = perform_inference(finetuned_model, tokenizer, sample, create_eval_prompt_local, "finetuned model")
        mx.metal.set_cache_limit(initial_memory)
    
    return sample, initial_model_response, finetuned_model_response    

def compare_anthropic(args, sample):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    
    # Load finetuned model
    finetuned_model, tokenizer = load_finetuned_model(args)
    
    def perform_inference_anthropic(sample):
        if isinstance(sample, dict) and 'question' in sample:
            question = sample['question']
            prompt = create_eval_prompt_anthropic(question)
            
            client = anthropic.Anthropic(
                api_key = os.getenv("ANTHROPIC_API_KEY"),
            )
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.2,
            )
            print(response.content)
            response_content = response.content[0].text
            print(response_content)
            return response_content
        else:
            print("Sample must be a dictionary containing a 'question' key.")
            return None

    # Perform inference
    comparison_sample = sample
    compared_model_response = perform_inference_anthropic(sample)
    finetuned_model_response = perform_inference(finetuned_model, tokenizer, sample, create_eval_prompt_anthropic, "Finetuned Model")

    return comparison_sample, compared_model_response, finetuned_model_response