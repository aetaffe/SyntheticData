import glob
import os
from pathlib import Path
import shutil

file_counter = 0
folders = [f for f in glob.glob("data/train/*")]
Path("data/segmentation_data").mkdir(parents=True, exist_ok=True)

for folder in folders:
    files = [f for f in glob.glob(folder + "/*.jpg")]
    for img_file_path in files:
        filename = os.path.basename(img_file_path)
        new_img_filename = f"data/segmentation_data/{file_counter}.jpg"
        new_json_filename = f"data/segmentation_data/{file_counter}.json"
        json_file_path = img_file_path.replace('.jpg', '.json')
        if os.path.exists(json_file_path):
            shutil.move(img_file_path, new_img_filename)
            shutil.move(json_file_path, new_json_filename)
            file_counter += 1
            print(f"Moved {img_file_path} to {new_img_filename}")
