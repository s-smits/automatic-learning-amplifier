import subprocess
import sys

command = [
    sys.executable, 
    '-m', 'mlx_vlm.generate',
    '--model', 'qnguyen3/nanoLLaVA',
    '--max-tokens', '100',
    '--temp', '0.0',
    '--image', "http://images.cocodataset.org/val2017/000000039769.jpg",
    '--verbose', 'True'
]

result = subprocess.run(command, capture_output=True, text=True)
caption = result.stdout
print(caption)