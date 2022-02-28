import re
import random
import typing as T
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torchvision import transforms

from common import hdr2ldr, normalize, required_images, Stage


def all_images_exist(folder: Path, images: T.List[str]):
    """Checks if folder contains all required images"""
    return all((folder / img).is_file() for img in images.values())


class DataLoaderHelper(Dataset):
    def __init__(self, root_dir: str, stage: Stage, pattern: str = ".*"):
        super().__init__()

        self.stage = stage
        self.required_images = dict(required_images)
        self.valid_folders = []
        self.base_transform = [
            transforms.Resize((256, 256)),
            transforms.Lambda(hdr2ldr),
            normalize,
        ]

        # remove need for ground truth when predicting
        if stage is Stage.PREDICT:
            self.required_images.pop("global")
            self.required_images.pop("indirect")

        # add one or more valid folders
        root_folder = Path(root_dir)
        if all_images_exist(root_folder, self.required_images):
            self.valid_folders.append(root_folder)
        else:
            cond = re.compile(pattern)
            for folder in root_folder.iterdir():
                if cond.match(folder.stem) and all_images_exist(folder, self.required_images):
                    self.valid_folders.append(folder)

    def __getitem__(self, index):
        transform = list(self.base_transform)
        if self.stage is Stage.TRAIN:
            if random.random() > 0.5:  # Random horizontal flipping
                transform.append(transforms.Lambda(TF.hflip))

            if random.random() > 0.5:  # Random vertical flipping
                transform.append(transforms.Lambda(TF.vflip))
        transform = transforms.Compose(transform)

        result = []
        for img in self.required_images.values():
            path = self.valid_folders[index] / img
            image = cv2.imread(str(path.resolve()), flags=cv2.IMREAD_ANYDEPTH)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_torch = torch.from_numpy(image_rgb).permute(2, 0, 1)
            result.append(transform(image_torch))
        return result

    def __len__(self):
        return len(self.valid_folders)
