import os
from mlx_lm.utils import load, generate
from setup_and_parse import BaseModelPaths, FolderPaths
import mlx.core as mx
from ggml_inference import ggml_inference

def compare_models(args, sample):
    folders = FolderPaths(args)
    
    # Define helper functions
    def create_eval_prompt(question):
        return f'''<[begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|><prompt>{question}</prompt><|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

    def load_model(args, model_type):
        base_paths = BaseModelPaths()
        folders = FolderPaths(args)
        if model_type == 'initial':
            model_path = base_paths.get_model_path('fp16') if args.fp16 else base_paths.get_model_path('mlx_4bit')
            print('Loaded model for initial model: ', model_path)
        elif model_type == 'finetuned':
            if args.ft_type == "lora":
                model_path = folders.ft_folder
            elif args.ft_type == "qlora":
                model_path = folders.ft_folder
            print('Loaded model for finetuned model: ', model_path)
            if model_path is None or not os.path.exists(model_path):
                raise FileNotFoundError(f"Finetuned model file not found at {model_path}")
        return load(model_path)

    def perform_inference(model, tokenizer, sample, description):
        if isinstance(sample, dict) and 'question' in sample:
            question = sample['question']
            eval_prompt = create_eval_prompt(question)
            response = generate(model, tokenizer, eval_prompt, temp=0.2, max_tokens=1024, verbose=True)
            print(response)
            return response
        else:
            print("Sample must be a dictionary containing a 'question' key.")
            return None

    # GGUF model inference
    if args.gguf:
        pass
        model, tokenizer = load_model(args, 'initial')
        initial_model_response = ggml_inference(model, tokenizer, sample, "initial GGUF model")
        model, tokenizer = load_model(args, 'finetuned')
        finetuned_model_file = os.path.join(folders.ft_folder, "ggml-model-q5_K_M.bin")
        finetuned_model_response = ggml_inference(model, tokenizer, sample, finetuned_model_file, "finetuned GGUF model")
    else:
        # MLX inference
        initial_memory = mx.metal.set_cache_limit(0)
        model, tokenizer = load_model(args, 'initial')
        initial_model_response = perform_inference(model, tokenizer, sample, "initial model")
        mx.metal.set_cache_limit(initial_memory)

        model, tokenizer = load_model(args, 'finetuned')
        finetuned_model_response = perform_inference(model, tokenizer, sample, "finetuned model")
        mx.metal.set_cache_limit(initial_memory)
    
    return sample, initial_model_response, finetuned_model_response    

