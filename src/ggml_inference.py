def ggml_inference(prompt, model_file, n_ctx, temperature):
    from llama_cpp import Llama
    def reload_model():
        global llm
        llm = Llama(model_path=model_file, n_ctx=n_ctx, verbose=True)
    
    reload_model()

    output = llm(
        prompt,
        temperature=temperature,
        stop=["}\n]\n}"],
        max_tokens=512,
        echo=True)
    
    print(output)
    return output
