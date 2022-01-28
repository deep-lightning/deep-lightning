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
        self.light = "(up|down|back|left|right)"

    def buildPositionsRegex(self, nums):
        return f"^(bunny|buddha|cube|dragon|sphere)_{nums}(_{self.light})?$"

    def buildBunnyBuddhaRegex(self, nums):
        return f"^(bunny|buddha)_{nums}(_{self.light})?$"

    def buildCubeSphereRegex(self, nums):
        return f"^cube_sphere_{nums}_(horizontal|vertical)(_{self.light})?$"

    def setup(self, stage=None):
        train_regex = val_regex = test_regex = ".*"

        train_nums = "([0-9]?[0-9]?[0-9]|1[0-1][0-9][0-9])"
        val_nums = "(1[2-5][0-9][0-9])"
        test_nums = "(1[6-9][0-9][0-9])"

        if self.data_regex == "vanilla":
            train_regex = "^(cube|sphere|dragon)_.*$"
            val_regex = "^bunny_.*$"
            test_regex = "^buddha_.*$"

        if self.data_regex == "positions":
            train_regex = self.buildPositionsRegex(train_nums)
            val_regex = self.buildPositionsRegex(val_nums)
            test_regex = self.buildPositionsRegex(test_nums)

        # multiple cameras, multiple lights or walls
        if self.data_regex in {"cameras", "lights", "walls"}:
            train_regex = self.buildBunnyBuddhaRegex(train_nums)
            val_regex = self.buildBunnyBuddhaRegex(val_nums)
            test_regex = self.buildBunnyBuddhaRegex(test_nums)

        # cube and sphere
        if self.data_regex == "objects":
            train_regex = self.buildCubeSphereRegex(train_nums)
            val_regex = self.buildCubeSphereRegex(val_nums)
            test_regex = self.buildCubeSphereRegex(test_nums)

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
