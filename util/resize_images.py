import glob
import cv2
from pathlib import Path


base_folder = '/media/alex/1TBSSD/SSD/FLImDataset/FLIm-Images-256x256'

if __name__ == '__main__':
    print('Resizing images...')
    img_files = glob.glob(f'{base_folder}/*.jpg')
    Path(f'{base_folder}/resized_images').mkdir(parents=True, exist_ok=True)
    for img_file in img_files:
        img = cv2.imread(img_file)
        img = cv2.resize(img, (256, 256))
        filename = f'{base_folder}/resized_images/{img_file.split("/")[-1]}'
        cv2.imwrite(filename, img)
print('Done!')