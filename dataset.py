import random
from os import listdir
from os.path import join, isfile
from enum import Enum
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import re
import cv2
import torch
import utils


class RequiredImgs(Enum):
    ALBEDO = "diffuse.hdr"
    DIRECT = "local.hdr"
    NORMAL = "normal.hdr"
    DEPTH = "z.hdr"
    GT = "global.hdr"
    INDIRECT = "indirect.hdr"

    @classmethod
    def present_in(cls, folder):
        """Checks if folder contains all required images"""
        return all(isfile(join(folder, img.value)) for img in cls)


class DataLoaderHelper(Dataset):
    def __init__(self, root_dir: str, is_train: bool, pattern: str = ".*"):
        super().__init__()

        self.is_train = is_train

        cond = re.compile(pattern)

        self.valid_folders = []
        for folder in listdir(root_dir):
            path = join(root_dir, folder)
            if cond.match(folder) and RequiredImgs.present_in(path):
                self.valid_folders.append(path)

        transform = [
            transforms.Resize((256, 256)),
            transforms.Lambda(utils.hdr2ldr),
            utils.normalize,
        ]

        if self.is_train:
            if random.random() > 0.5:  # Random horizontal flipping
                transform.append(transforms.Lambda(TF.hflip))

            if random.random() > 0.5:  # Random vertical flipping
                transform.append(transforms.Lambda(TF.vflip))

        self.transform = transforms.Compose(transform)

    def __getitem__(self, index):

        result = []
        for img in RequiredImgs:
            path = Path(self.valid_folders[index], img.value)
            image = cv2.imread(str(path.resolve()), flags=cv2.IMREAD_ANYDEPTH)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_torch = torch.from_numpy(image_rgb).permute(2, 0, 1)
            result.append(self.transform(image_torch))
        return result

    def __len__(self):
        return len(self.valid_folders)
