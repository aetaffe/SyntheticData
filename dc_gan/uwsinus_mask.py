import glob
import cv2
from pathlib import Path
import numpy as np


base_folder = '/media/alex/1TBSSD/SSD/uw-sinus-surgery-CL/uw-sinus-surgery-CL/live/labels'
# 'snapshots/500_epochs/color_masks/snapshots/500_epochs/mask_6.jpg'


if __name__ == '__main__':
    print('Converting masks to color masks...')
    mask_files = glob.glob(f'{base_folder}/*.png')
    Path(f'{base_folder}/color_masks').mkdir(parents=True, exist_ok=True)
    for mask_file in mask_files:
        img = cv2.imread(mask_file)
        img = np.where(img == 1, (0, 0, 255), img)
        filename = f'{base_folder}/color_masks/{mask_file.split("/")[-1]}'
        cv2.imwrite(filename, img)

    print('Done')

