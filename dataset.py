import random
from os import listdir
from os.path import join, isfile
from enum import Enum

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import re
import cv2
import torch

from pathlib import Path
from functools import partial
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

    def __getitem__(self, index):

        item_folder = Path(self.valid_folders[index])
        result_dict = {}
        for img in RequiredImgs:
            image = cv2.imread(str((item_folder / img.value).resolve()), flags=cv2.IMREAD_ANYDEPTH)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_torch = torch.from_numpy(image_rgb).permute(2, 0, 1)
            result_dict[img] = image_torch

        transform = []
        if self.is_train:
            if random.random() > 0.5:  # Random horizontal flipping
                transform.append(transforms.Lambda(TF.hflip))

            if random.random() > 0.5:  # Random vertical flipping
                transform.append(transforms.Lambda(TF.vflip))

        result = []
        for img in RequiredImgs:
            if img in (RequiredImgs.DIRECT, RequiredImgs.INDIRECT, RequiredImgs.GT):
                max_light = result_dict[RequiredImgs.GT].max()
            else:
                max_light = result_dict[img].max()

            per_transform = list(transform)
            per_transform.extend(
                [
                    transforms.Resize((256, 256)),
                    transforms.Lambda(partial(utils.hdr2ldr, max_light)),
                    utils.norm,
                ]
            )
            composition = transforms.Compose(per_transform)
            result.append(composition(result_dict[img]))
        return result

    def __len__(self):
        return len(self.valid_folders)
