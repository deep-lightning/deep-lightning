from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import DataLoaderHelper


class DataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, data_regex: str):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_regex = data_regex

    def setup(self, stage=None):
        train_regex = val_regex = test_regex = ".*"

        train_nums = "([0-9]?[0-9]?[0-9]|1[0-1][0-9][0-9])"
        val_nums = "(1[2-5][0-9][0-9])"
        test_nums = "(1[6-9][0-9][0-9])"
        light = "(up|down|back|left|right)"

        if self.data_regex == "vanilla":
            train_regex = "^(cube|sphere|dragon)_.*$"
            val_regex = "^bunny_.*$"
            test_regex = "^buddha_.*$"

        if self.data_regex == "positions":
            train_regex = f"^(bunny|buddha|cube|dragon|sphere)_{train_nums}(_{light})?$"
            val_regex = f"^(bunny|buddha|cube|dragon|sphere)_{val_nums}(_{light})?$"
            test_regex = f"^(bunny|buddha|cube|dragon|sphere)_{test_nums}(_{light})?$"

        # multiple lights
        if self.data_regex == "lights":
            train_regex = f"^(bunny|buddha)_{train_nums}(_{light})?$"
            val_regex = f"^(bunny|buddha)_{val_nums}(_{light})?$"
            test_regex = f"^(bunny|buddha)_{test_nums}(_{light})?$"

        # cube and sphere
        if self.data_regex == "objects":
            train_regex = f"^cube_sphere_{train_nums}_(horizontal|vertical)(_{light})?$"
            val_regex = f"^cube_sphere_{val_nums}_(horizontal|vertical)(_{light})?$"
            test_regex = f"^cube_sphere_{test_nums}_(horizontal|vertical)(_{light})?$"

        # multiple cameras
        if self.data_regex == "cameras":
            train_regex = f"^(bunny|buddha)_{train_nums}(_{light})?$"
            val_regex = f"^(bunny|buddha)_{val_nums}(_{light})?$"
            test_regex = f"^(bunny|buddha)_{test_nums}(_{light})?$"

        # walls
        if self.data_regex == "walls":
            train_regex = f"^(bunny|buddha)_{train_nums}(_{light})?$"
            val_regex = f"^(bunny|buddha)_{val_nums}(_{light})?$"
            test_regex = f"^(bunny|buddha)_{test_nums}(_{light})?$"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = DataLoaderHelper(self.data_dir, True, train_regex)
            self.val_dataset = DataLoaderHelper(self.data_dir, False, val_regex)
            print(len(self.train_dataset), len(self.val_dataset))

        # Assign test dataset for use in dataloader
        if stage == "test" or stage is None:
            self.test_dataset = DataLoaderHelper(self.data_dir, False, test_regex)
            print(len(self.test_dataset))

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
