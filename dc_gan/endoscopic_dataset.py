import glob
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import supervision as sv

class EndoscopicSurgicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in glob.glob(root_dir + "/*.json")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        mask = None
        with open(file) as f:
            data = json.load(f)
            mask = _get_masks(data['shapes'], data['imageWidth'], data['imageHeight']).astype(np.uint8)

        image_filename = file.replace('.json', '.jpg')
        image = Image.open(image_filename)
        image = image.convert('RGB')
        assert len(image.getbands()) == 3, 'Image must have 3 channels'
        image = np.array(image)
        image = image.astype(np.uint8)
        image = np.dstack((image, mask))
        if self.transform:
            image = self.transform(image)
        return image

def _get_masks(polygons, width, height):
    mask = np.zeros((height, width))
    for idx, shape in enumerate(polygons):
        points = shape['points']
        instr_mask = sv.polygon_to_mask(np.array(points), (width, height))
        instr_mask = np.where(instr_mask == 1, idx + 1, 0)
        mask += instr_mask

    return mask