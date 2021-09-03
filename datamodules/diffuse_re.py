from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import DataLoaderHelper


class DiffuseDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = DataLoaderHelper(self.data_dir, True, "(cube|sphere|dragon)_.*")
            self.val_dataset = DataLoaderHelper(self.data_dir, False, "bunny_.*")
            # self.train_dataset = DataLoaderHelper(self.data_dir, True, "bunny_1?[0-1]?[0-9]?[0-9]")
            # self.val_dataset = DataLoaderHelper(self.data_dir, False, "bunny_1[2-5]?[0-9]?[0-9]")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = DataLoaderHelper(self.data_dir, False, "buddha_.*")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
