import torch
from generator import Generator
import numpy as np
import cv2
from pathlib import Path


if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Generator(1, 100, 128, 4).to(device)
    model.load_state_dict(torch.load('netG.pth', weights_only=True))
    model.eval()

    fake = model(torch.randn(100, 100, 1, 1, device=device)).detach().cpu()
    fake_images = [np.transpose((fake[idx, :3, :, :] * 0.5) + 0.5, (1, 2, 0)) for idx in range(fake.shape[0])]

    Path(f'snapshots/generated').mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(fake_images):
        cv2.imwrite(f'snapshots/generated/fake_image_{idx}.jpg', cv2.cvtColor(img.numpy() * 255, cv2.COLOR_RGB2BGR))