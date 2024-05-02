import os

# Step 1: Create a text file
video_paths = [
    "/Users/air/Desktop/Screen Recording 2024-05-02 at 19.53.03.mov",
    "/Users/air/Desktop/Screen Recording 2024-05-02 at 20.30.43.mov",
    "/Users/air/Desktop/Screen Recording 2024-05-02 at 20.45.40.mov"
]

with open("mylist.txt", "w") as f:
    for path in video_paths:
        f.write(f"file '{path}'\n")

# Step 2: Use ffmpeg to concatenate the videos
os.system("ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mov")