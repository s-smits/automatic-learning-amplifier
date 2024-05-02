import json
import os

def write_conversations(conversations, file):
    with open(os.path.join('mixtral_outputs', f'output_{file}.txt'), 'w') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')