import glob
import json
from polygon_to_mask import get_mask
import numpy as np
import cv2

json_files = [f for f in glob.glob('data/*.json')]


def get_cutout(img, mask):
    cutout = np.zeros(img.shape)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] == 1:
                cutout = img[row, col, :]
    return cutout


def make_cutouts():
    print("Making cutouts of sugical tool")
    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            instrument_mask = get_mask(data['shapes'], data['imageWidth'], data['imageHeight']).astype(np.uint8)
            image_file = json_file.replace('.json', '.jpg')
            img = cv2.imread(image_file)
            # cutout = get_cutout(img, instrument_mask)
            instrument_mask = np.dstack([instrument_mask] * 3)
            cutout = np.where(instrument_mask > 0, img, 0)
            cutout_filename = f'{image_file.split('.')[0]}_cutout.jpg'
            cv2.imwrite(cutout_filename, cutout)
    print(f"Files saved to {image_file.split('/')[0]}")


if __name__ == '__main__':
    make_cutouts()