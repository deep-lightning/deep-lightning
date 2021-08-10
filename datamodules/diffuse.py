from os.path import join

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
            train_dir = join(self.data_dir, "train")
            self.train_dataset = DataLoaderHelper(train_dir)

            val_dir = join(self.data_dir, "val")
            self.val_dataset = DataLoaderHelper(val_dir)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_dir = join(self.data_dir, "test")
            self.test_dataset = DataLoaderHelper(test_dir)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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
