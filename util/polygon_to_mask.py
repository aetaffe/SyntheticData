import supervision as sv
import glob
import json
import numpy as np
import cv2
from pathlib import Path

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
magenta = (255, 0, 255)
cyan = (255, 255, 0)

colors = [red, green, blue, yellow, magenta, cyan]
show = False
files = mylist = [f for f in glob.glob("data/*.json")]
Path("../data/masks_color").mkdir(parents=True, exist_ok=True)


def get_mask(polygons, width, height):
    mask = np.zeros((height, width))
    mask_num = 1
    for idx, shape in enumerate(polygons):
        points = shape['points']
        instr_mask = sv.polygon_to_mask(np.array(points), (width, height))
        instr_mask = np.where(instr_mask == 1, mask_num, 0)
        mask += instr_mask
        mask_num += 1
    return mask


for file in files:
    with open(file) as f:
        data = json.load(f)
        instrument_mask = get_mask(data['shapes'], data['imageWidth'], data['imageHeight']).astype(np.uint8)
        mask_filename = file.replace('.json', '_mask.png')
        print(f'Saving mask to {mask_filename}')
        cv2.imwrite(mask_filename, instrument_mask * 50)