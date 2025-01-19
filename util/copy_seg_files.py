import glob
import os
from pathlib import Path
import shutil

print('Copying segmentation files...')
base_folder = '/media/alex/1TBSSD/SSD/FLImDataset/'
file_counter = 0
folders = [f for f in glob.glob(base_folder + "images/*")]
Path(base_folder + "segmentation_data").mkdir(parents=True, exist_ok=True)

for folder in folders:
    img_files = [f for f in glob.glob(f'{folder}/*.jpg')]
    for img_file_path in img_files:
        new_img_filename = f"{base_folder}/images/{file_counter}.jpg"
        new_json_filename = f"{base_folder}/images/{file_counter}.json"
        json_file_path = img_file_path.replace('.jpg', '.json')
        if os.path.exists(json_file_path):
            shutil.move(img_file_path, new_img_filename)
            shutil.move(json_file_path, new_json_filename)
            file_counter += 1
            print(f"Moved {img_file_path} to {new_img_filename}")
print('Done')