#json_extractor.py
import json

def extract_json(output):
    try:
        start_index = output.find('{')
        end_index = output.rfind('}') + 1
        json_str = output[start_index:end_index]
        data = json.loads(json_str)
        json_data = data['text']
        print('Succesfully extracted JSON')
        return json_data
    except json.JSONDecodeError:
        return None  # Return None if JSON is not correctly parsed
