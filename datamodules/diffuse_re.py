from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import DataLoaderHelper


class DiffuseDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, data_regex: str):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_regex = data_regex

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_regex = val_regex = test_regex = ".*"

        if self.data_regex == "vanilla":
            train_regex = "(cube|sphere|dragon)_.*"
            val_regex = "bunny_.*"
            test_regex = "buddha_.*"

        # multiple lights
        if self.data_regex == "lights":
            train_regex = "(bunny|buddha)_([0-9]?[0-9]?[0-9]|1[0-1][0-9][0-9]).*"
            val_regex = "(bunny|buddha)_1[2-5][0-9][0-9].*"
            test_regex = "(bunny|buddha)_1[6-9][0-9][0-9].*"

        # cube and sphere
        if self.data_regex == "objects":
            train_regex = "cube_sphere_([0-9]?[0-9]?[0-9]|1[0-1][0-9][0-9])_.*"
            val_regex = "cube_sphere_1[2-5][0-9][0-9]_.*"
            test_regex = "cube_sphere_1[6-9][0-9][0-9]_.*"

        # multiple cameras
        if self.data_regex == "cameras":
            train_regex = "(bunny|buddha)_([0-9]?[0-9]?[0-9]|1[0-1][0-9][0-9]).*"
            val_regex = "(bunny|buddha)_1[2-5][0-9][0-9].*"
            test_regex = "(bunny|buddha)_1[6-9][0-9][0-9].*"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = DataLoaderHelper(self.data_dir, True, train_regex)
            self.val_dataset = DataLoaderHelper(self.data_dir, False, val_regex)

        # Assign test dataset for use in dataloader
        if stage == "test" or stage is None:
            self.test_dataset = DataLoaderHelper(self.data_dir, False, test_regex)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )
