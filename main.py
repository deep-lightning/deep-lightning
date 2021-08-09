import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from torchmetrics import MetricCollection, SSIM, MeanSquaredError

from torchmetrics.functional.regression import ssim, mean_squared_error

from model import D, G, weights_init

from dataset import DataLoaderHelper
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from os.path import join
from argparse import ArgumentParser, Namespace


class CGan(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.generator = G(
            hparams.n_channel_input * 4,
            hparams.n_channel_output,
            hparams.n_generator_filters,
        )
        self.generator.apply(weights_init)

        self.discriminator = D(
            hparams.n_channel_input * 4,
            hparams.n_channel_output,
            hparams.n_discriminator_filters,
        )
        self.discriminator.apply(weights_init)

        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()

        # metrics = MetricCollection({"mse": MeanSquaredError()})
        # self.train_metrics = metrics.clone(prefix="train_")
        # self.val_metrics = metrics.clone(prefix="val_")
        # self.test_metrics = metrics.clone(prefix="test_")

        self.lr = hparams.lr
        self.beta1 = hparams.beta1
        self.lambda_factor = hparams.lambda_factor
        self.dataset = hparams.dataset
        self.batch_size = hparams.batch_size
        self.workers = hparams.workers

        # self.train_ssim = []
        self.train_mse = []

        # self.val_ssim = []
        self.val_mse = []

        # self.test_ssim = []
        self.test_mse = []

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.generator(x)

    def train_dataloader(self):
        train_dir = join(self.dataset, "train")
        train_dataset = DataLoaderHelper(train_dir)
        return DataLoader(
            dataset=train_dataset,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dir = join(self.dataset, "val")
        val_dataset = DataLoaderHelper(val_dir)
        return DataLoader(
            dataset=val_dataset,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        test_dir = join(self.dataset, "test")
        test_dataset = DataLoaderHelper(test_dir)
        return DataLoader(
            dataset=test_dataset,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def generator_loss(self, albedo, direct, normal, depth, gt):

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        prediction = self.discriminator(torch.cat((z, fake), 1))
        y = torch.ones(prediction.size(), device=self.device)

        gan_loss = self.gan_loss(prediction, y)
        l1_loss = self.l1_loss(fake, gt)
        # self.train_metrics(fake, gt)
        # self.train_ssim.append(ssim(fake, gt))
        self.train_mse.append(mean_squared_error(fake, gt))

        return gan_loss + self.lambda_factor * l1_loss

    def discriminator_loss(self, albedo, direct, normal, depth, gt):

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        # train on real data
        real_data = torch.cat((z, gt), 1)
        prediction_real = self.discriminator(real_data)
        y_real = torch.ones(prediction_real.size(), device=self.device)

        # calculate error and backpropagate
        real_loss = self.gan_loss(prediction_real, y_real)

        # train on fake data
        fake_data = torch.cat((z, fake), 1)
        prediction_fake = self.discriminator(fake_data)
        y_fake = torch.zeros(prediction_real.size(), device=self.device)

        # calculate error and backpropagate
        fake_loss = self.gan_loss(prediction_fake, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        return real_loss + fake_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_loss(*batch)
            self.log("g_loss", result, on_epoch=True)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_loss(*batch)
            self.log("d_loss", result, on_epoch=True)

        return result

    def training_epoch_end(self, outputs) -> None:
        # self.log_dict(self.train_metrics.compute(), prog_bar=True)
        # self.train_metrics.reset()
        # self.log("train_ssim", torch.stack(self.train_ssim).mean(), prog_bar=True)
        self.log("train_mse", torch.stack(self.train_mse).mean(), prog_bar=True)
        self.train_mse = []
        # self.train_ssim = []

    def validation_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx == 0:
            logger = self.logger.experiment
            logger.add_images("val_fake", fake, self.current_epoch)
            logger.add_images("val_real", gt, self.current_epoch)

        # self.val_metrics(fake, gt)

        # self.val_ssim.append(ssim(fake, gt))
        self.val_mse.append(mean_squared_error(fake, gt))

    def validation_epoch_end(self, outputs) -> None:
        # self.log_dict(self.val_metrics.compute(), prog_bar=True)
        # self.val_metrics.reset()
        # self.log("val_ssim", torch.stack(self.val_ssim).mean(), prog_bar=True)
        self.log("val_mse", torch.stack(self.val_mse).mean(), prog_bar=True)
        self.val_mse = []
        # self.val_ssim = []

    def test_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx == 0:
            logger = self.logger.experiment
            logger.add_images("test_fake", fake, self.current_epoch)
            logger.add_images("test_real", gt, self.current_epoch)

        # self.test_metrics(fake, gt)

        # self.test_ssim.append(ssim(fake, gt))
        self.test_mse.append(mean_squared_error(fake, gt))

    def test_epoch_end(self, outputs) -> None:
        # self.log_dict(self.test_metrics.compute(), prog_bar=True)
        # self.test_metrics.reset()
        # self.log("test_ssim", torch.stack(self.test_ssim).mean(), prog_bar=True)
        self.log("test_mse", torch.stack(self.test_mse).mean(), prog_bar=True)
        self.test_mse = []
        # self.test_ssim = []

    def configure_optimizers(self):
        opt_d = Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        opt_g = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_g, opt_d]


def main(hparams):
    pl.seed_everything(42, workers=True)

    callbacks = [
        EarlyStopping(monitor="val_mse", mode="min", patience=3),
        # EarlyStopping(monitor="val_ssim", mode="max"),
        ModelCheckpoint(
            monitor="val_mse",
            filename="cgan-{epoch:02d}-{val_mse:.2f}",
            save_top_k=3,
        ),
    ]

    model = CGan(hparams)
    # print(model)

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        callbacks=callbacks,
        deterministic=True,
        # auto_lr_find=True
        # auto_scale_batch_size="power",
    )

    # train_dir = join(hparams.dataset, "train")
    # train_dataset = DataLoaderHelper(train_dir)
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     num_workers=hparams.workers,
    #     batch_size=hparams.train_batch_size,
    #     shuffle=True,
    # )

    # val_dir = join(hparams.dataset, "val")
    # val_dataset = DataLoaderHelper(val_dir)
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     num_workers=hparams.workers,
    #     batch_size=hparams.val_batch_size,
    #     shuffle=False,
    # )

    # trainer.tune(model, train_dataloaders=[train_loader], val_dataloaders=[val_loader])

    # trainer.tune(model)

    trainer.fit(model)

    # test_dir = join(hparams.dataset, "test")
    # test_dataset = DataLoaderHelper(test_dir)
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     num_workers=hparams.workers,
    #     batch_size=hparams.test_batch_size,
    #     shuffle=False,
    # )

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", default=False)
    parser.add_argument("--gpus", type=int, default=0)

    parser.add_argument(
        "--dataset", required=True, help="location of train, val and test folders"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--n_channel_input", type=int, default=3, help="number of input channels"
    )
    parser.add_argument(
        "--n_channel_output", type=int, default=3, help="number of output channels"
    )
    parser.add_argument(
        "--n_generator_filters",
        type=int,
        default=64,
        help="number of initial generator filters",
    )
    parser.add_argument(
        "--n_discriminator_filters",
        type=int,
        default=64,
        help="number of initial discriminator filters",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1")
    parser.add_argument(
        "--lambda_factor", type=int, default=100, help="L1 regularization factor"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads for data loader"
    )
    args = parser.parse_args()
    main(args)
