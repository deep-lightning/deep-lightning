import random
from os import listdir
from os.path import join, isfile
from enum import Enum

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class RequiredImgs(Enum):
    ALBEDO = "diffuse.png"
    DIRECT = "local.png"
    NORMAL = "normal.png"
    DEPTH = "z.png"
    GT = "global.png"
    # INDIRECT = "indirect.png"

    @classmethod
    def present_in(cls, folder):
        """Checks if folder contains all required images"""
        return all(isfile(join(folder, img.value)) for img in cls)


class DataLoaderHelper(Dataset):
    def __init__(self, root_dir: str, is_train: bool):
        super().__init__()

        self.is_train = is_train

        self.valid_folders = [
            join(root_dir, folder) for folder in listdir(root_dir) if RequiredImgs.present_in(join(root_dir, folder))
        ]

        self.base_transforms = [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

    def __getitem__(self, index):
        transformations = list(self.base_transforms)

        if self.is_train:
            if random.random() > 0.5:  # Random horizontal flipping
                transformations.insert(0, transforms.Lambda(lambda x: TF.hflip(x)))

            if random.random() > 0.5:  # Random vertical flipping
                transformations.insert(0, transforms.Lambda(lambda x: TF.vflip(x)))

        composition = transforms.Compose(transformations)

        return tuple(
            composition(Image.open(join(self.valid_folders[index], img.value)).convert("RGB")) for img in RequiredImgs
        )

    def __len__(self):
        return len(self.valid_folders)
