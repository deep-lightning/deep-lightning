from os import listdir
from os.path import join, isfile

from enum import Enum
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image


class RequiredImgs(Enum):
    ALBEDO = "diffuse.png"
    DIRECT = "local.png"
    NORMAL = "normal.png"
    DEPTH = "z.png"
    GT = "global.png"

    @classmethod
    def present_in(cls, folder):
        """Checks if folder contains all required images"""
        return all(isfile(join(folder, img.value)) for img in cls)


class DataLoaderHelper(Dataset):
    def __init__(self, root_dir):
        super().__init__()

        self.valid_folders = [
            join(root_dir, folder)
            for folder in listdir(root_dir)
            if RequiredImgs.present_in(join(root_dir, folder))
        ]

        self.transforms = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor()]
        )

    def __getitem__(self, index):

        return tuple(
            self.transforms(
                Image.open(join(self.valid_folders[index], img.value)).convert("RGB")
            )
            for img in RequiredImgs
        )

    def __len__(self):
        return len(self.valid_folders)
