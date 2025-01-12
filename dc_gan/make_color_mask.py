import glob
import cv2
from pathlib import Path
import numpy as np


red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
magenta = (255, 0, 255)
cyan = (255, 255, 0)

colors = [red, green, blue, yellow, magenta, cyan]

base_folder = 'snapshots/2000_epochs'
# 'snapshots/500_epochs/color_masks/snapshots/500_epochs/mask_6.jpg'
if __name__ == '__main__':
    print('Converting masks to color masks...')
    mask_files = glob.glob(f'{base_folder}/mask_*.jpg')
    Path(f'{base_folder}/color_masks').mkdir(parents=True, exist_ok=True)
    for mask_file in mask_files:
        img = cv2.imread(mask_file)
        img = np.where(img > 3, 0, img)
        for i in range(3):
            img = np.where(img == (i + 1), colors[i], img)
        filename = f'{base_folder}/color_masks/{mask_file.split("/")[-1]}'
        cv2.imwrite(filename, img)

    print('Done')

