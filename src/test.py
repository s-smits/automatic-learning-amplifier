from mlx_lm.utils import load, generate
from setup_and_parse import BaseModelPaths
batch_size = 5

file_contents = "Lorem ipusm ..." #6k tokens
base_models = BaseModelPaths()
model,tokenizer = load('mlx-community/Phi-3-mini-128k-instruct-4bit')
prompt_template = f"<|system|>\nYou are a concise summarization assistant. You answer in 10 sentences maximum.<|end|>\n<|user|>\n<prompt>Generate an extensive summary</prompt><file_content>{file_contents}</file_content><|end|>\n<|assistant|>"
prompt_with_file = f'{prompt_template}\n<text_corpus>{file_contents}</text_corpus><|eot_id|><|start_header_id|>assistant<|end_header_id|><json_output>'
print(generate(model, tokenizer, prompt_with_file))