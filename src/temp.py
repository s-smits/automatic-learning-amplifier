from mlx_vlm import generate

image = "http://images.cocodataset.org/val2017/000000039769.jpg"
caption = generate(model='qnguyen3/nanoLLaVA',
                image=image,
                # processor = Automatically determined by the model choice
                # image_processor = Automatically determined by the model choice
                prompt="Describe this image.",
                temp=0.0,
                max_tokens=100,
                verbose=False,
                formatter=None,
                repetition_penalty=None,
                repetition_context_size=None,
                top_p=1
                )