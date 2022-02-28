import re
import random
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torchvision import transforms

import utils

required_images = {
    "diffuse": "diffuse.hdr",
    "local": "local.hdr",
    "normal": "normal.hdr",
    "depth": "z.hdr",
    "global": "global.hdr",
    "indirect": "indirect.hdr",
}


def all_images_exist(folder: Path):
    """Checks if folder contains all required images"""
    return all((folder / img).is_file() for img in required_images.values())


class DataLoaderHelper(Dataset):
    def __init__(self, root_dir: str, is_train: bool, pattern: str = ".*"):
        super().__init__()

        self.is_train = is_train

        cond = re.compile(pattern)

        self.valid_folders = []
        for folder in Path(root_dir).iterdir():
            if cond.match(folder.stem) and all_images_exist(folder):
                self.valid_folders.append(folder)

        self.base_transform = [
            transforms.Resize((256, 256)),
            transforms.Lambda(utils.hdr2ldr),
            utils.normalize,
        ]

    def __getitem__(self, index):

        transform = list(self.base_transform)
        if self.is_train:
            if random.random() > 0.5:  # Random horizontal flipping
                transform.append(transforms.Lambda(TF.hflip))

            if random.random() > 0.5:  # Random vertical flipping
                transform.append(transforms.Lambda(TF.vflip))
        transform = transforms.Compose(transform)

        result = []
        for img in required_images.values():
            path = self.valid_folders[index] / img
            image = cv2.imread(str(path.resolve()), flags=cv2.IMREAD_ANYDEPTH)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_torch = torch.from_numpy(image_rgb).permute(2, 0, 1)
            result.append(transform(image_torch))
        return result

    def __len__(self):
        return len(self.valid_folders)
