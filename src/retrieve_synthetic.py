# # retrieve_synthetic.py
# import requests
# import json
# from json_extractor import extract_json
# from mlx_lm.utils import load, generate
# import os
# import mlx.core as mx
# from stqdm import stqdm
# from verify_outputs import verify_outputs
# import streamlit as st



    
# def infere_claude(prompt, file):
#     import anthropic

#     with open(os.path.join(file), 'r') as f:
#         text_chunk = f.read()
    
#     prompt = f"""{prompt} {text_chunk}
#     """
    
#     client = anthropic.Anthropic(
#         api_key = os.getenv("ANTHROPIC_API_KEY"),
#     )
    
#     response = client.messages.create(
#         model="claude-3-haiku-20240307",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=1536,
#         # temperature=0.6, old choice
#         temperature=0.3,
#     )
        
#     response_content = response.content
#     print(response_content)

#     # Extract the JSON string from the ContentBlock object
#     content_block = response_content[0]
#     json_string = content_block.text

#     # Use the extract_json function to extract the JSON component
#     data_formatted = extract_json(json_string)

#     def write_questions(data_formatted, file):
#         output_file = os.path.splitext(file)[0] + ".json"
#         output_path = os.path.join("qa_json", output_file)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, 'w') as f:
#             f.write(data_formatted)
#             print(f"Questions and answers written to {output_file}")

#     write_questions(data_formatted, file)

# def infere_openrouter(prompt, file):
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()

#     with open(os.path.join(file), 'r') as f:
#         text_chunk = f.read()

#     prompt = f"""{prompt} {text_chunk}"""
    
#     response = requests.post(
#         url="https://openrouter.ai/api/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
#             "HTTP-Referer": None,
#             "X-Title": None,
#         },
#         data=json.dumps({
#             "model": "mistralai/mixtral-8x7b-instruct:nitro",
#             "messages": [{"role": "user", "content": prompt}]
#         })
#     )

#     response_content = response.json()
#     print(response_content)
#     return response_content