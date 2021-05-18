import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from torchmetrics import SSIM, MeanSquaredError

from model import D, G, weights_init

from dataset import DataLoaderHelper
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from os.path import join
from argparse import ArgumentParser


class CGan(pl.LightningModule):
    def __init__(
        self,
        n_channel_input,
        n_channel_output,
        n_generator_filters,
        n_discriminator_filters,
        lr,
        beta1,
        lambda_factor,
    ):
        super().__init__()

        self.generator = G(n_channel_input * 4, n_channel_output, n_generator_filters)
        self.generator.apply(weights_init)

        self.discriminator = D(
            n_channel_input * 4, n_channel_output, n_discriminator_filters
        )
        self.discriminator.apply(weights_init)

        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()

        self.ssim = SSIM()
        self.mse = MeanSquaredError()

        self.lr = lr
        self.beta1 = beta1
        self.lambda_factor = lambda_factor

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.generator(x)

    def generator_loss(self, albedo, direct, normal, depth, gt):

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self(z)

        prediction = self.discriminator(torch.cat((z, fake), 1))
        y = torch.ones(prediction.size(), device=self.device)

        gan_loss = self.gan_loss(prediction, y)
        l1_loss = self.l1_loss(fake, gt)

        return gan_loss + self.lambda_factor * l1_loss

    def discriminator_loss(self, albedo, direct, normal, depth, gt):

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self(z)

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
        albedo, direct, normal, depth, gt = batch

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_loss(albedo, direct, normal, depth, gt)
            self.log("g_loss", result, on_epoch=True, prog_bar=True)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_loss(albedo, direct, normal, depth, gt)
            self.log("d_loss", result, on_epoch=True, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self(z)

        if batch_idx == 0:
            logger = self.logger.experiment
            logger.add_images("fake", fake)
            logger.add_images("real", gt)

        ssim_score = self.ssim(fake, gt)
        self.log("ssim", ssim_score, on_epoch=True, prog_bar=True)

        mse_score = self.mse(fake, gt)
        self.log("mse", mse_score, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt_d = Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        opt_g = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_g, opt_d]


def main(hparams):
    pl.seed_everything(42, workers=True)

    callbacks = [
        EarlyStopping(monitor="ssim"),
        ModelCheckpoint(
            monitor="ssim",
            filename="cgan-{epoch:02d}-{ssim:.2f}-{mse:.2f}",
            save_top_k=3,
        ),
    ]

    model = CGan(
        n_channel_input=hparams.n_channel_input,
        n_channel_output=hparams.n_channel_output,
        n_generator_filters=hparams.n_generator_filters,
        n_discriminator_filters=hparams.n_discriminator_filters,
        lr=hparams.lr,
        beta1=hparams.beta1,
        lambda_factor=hparams.lambda_factor,
    )

    trainer = pl.Trainer(gpus=hparams.gpus, callbacks=callbacks, deterministic=True)

    if hparams.test:
        test_dir = join(hparams.dataset, "test")
        test_dataset = DataLoaderHelper(test_dir)
        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=hparams.workers,
            batch_size=hparams.test_batch_size,
            shuffle=False,
        )

        trainer.test(model, test_dataloaders=test_loader)
    else:
        train_dir = join(hparams.dataset, "train")
        train_dataset = DataLoaderHelper(train_dir)
        train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=hparams.workers,
            batch_size=hparams.train_batch_size,
            shuffle=True,
        )

        val_dir = join(hparams.dataset, "val")
        val_dataset = DataLoaderHelper(val_dir)
        val_loader = DataLoader(
            dataset=val_dataset,
            num_workers=hparams.workers,
            batch_size=hparams.val_batch_size,
            shuffle=False,
        )

        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", default=False)
    parser.add_argument("--gpus", default=None)

    parser.add_argument(
        "--dataset", required=True, help="location of train, val and test folders"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="batch size for validating"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=4, help="batch size for testing"
    )
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
