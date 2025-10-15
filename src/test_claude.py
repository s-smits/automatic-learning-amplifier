from config import initialize_setup

args, _ = initialize_setup()

def compare_anthropic(args, sample):
    import anthropic
    import os
    from mlx_lm.utils import load
    from config import FolderPaths
    
    def create_eval_prompt(question):
        return f'''<system>You are a helpful assistant.</system><prompt>{question}</prompt>'''

    def load_model(args, model_type):
        folders = FolderPaths(args)
        if model_type == 'finetuned':
            model_path = folders.ft_folder
            print('Loaded model for finetuned model: ', model_path)
            if model_path is None or not os.path.exists(model_path):
                raise FileNotFoundError(f"Finetuned model file not found at {model_path}")
        return load(model_path)

    def perform_inference_anthropic(sample):
        from dotenv import load_dotenv
        load_dotenv()
        
        if isinstance(sample, dict) and 'question' in sample:
            question = sample['question']
            prompt = create_eval_prompt(question)
            
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.2,
            )
            print("API Response Content:", response.content)
            # try:
            #     response_content = response.content[0]['text']  # Corrected access to response content
            #     print("Extracted Response Text:", response_content)
            # except:
            #     print("Error processing API response content. for option 1")
            #     return None
            try:
                response_content = response.content[0].text  # Corrected access to response content
                print("Extracted Response Text:", response_content)
            except AttributeError:
                print("Error processing API response content. for option 2")
                return None
            try:
                response_content = response.TextBlock.text  # Corrected access to response content
                print("Extracted Response Text:", response_content)
            except AttributeError:
                print("Error processing API response content. for option 3")
                return None

            return response_content
        else:
            print("Sample must be a dictionary containing a 'question' key.")
            return None
    perform_inference_anthropic(sample)

sample = {'question': 'What is the name of the author of this code?'}
response = compare_anthropic(args, sample)
print("Final response from Anthropic API:", response)

