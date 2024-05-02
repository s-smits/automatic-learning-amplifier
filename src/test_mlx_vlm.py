import subprocess
import sys

command = [
    sys.executable, "-m", "mlx_vlm.generate",
    "--model", "mlx-community/llava-1.5-7b-4bit",
    "--prompt", "USER: <image>\nWhat are these?\nASSISTANT:",
    "--image", "http://images.cocodataset.org/val2017/000000039769.jpg",
    "--max-tokens", "128",
    "--temp", "0.0"
]

result = subprocess.run(command, text=True, capture_output=True)
print(result.stdout)