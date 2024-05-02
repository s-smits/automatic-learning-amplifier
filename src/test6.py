from mlx_vlm.utils import get_model_path, load, load_config, load_image_processor, generate

model_path = "mlx-community/nanoLLaVA"
model_path = get_model_path(model_path)
model, processor = load(model_path)
config = load_config(model_path)
image_processor = load_image_processor(config)

prompt = processor.apply_chat_template(
    [{"role": "user", "content": f"<image>\nWhat's so funny about this image?"}],
    tokenize=False,
    add_generation_prompt=True,
)

image_path =  "/Users/air/Repositories/automatic-local-document-finetuner/data/images/image_8.jpg"
output = generate(model, processor, image_path, prompt, image_processor, verbose=False)
print(output)