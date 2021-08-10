import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam

from model import D, G, weights_init

from argparse import Namespace

from torchmetrics import MetricCollection, MeanSquaredError
from metrics.ssim import SSIM


class CGan(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

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

        # losses
        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()

        # metrics
        metrics = MetricCollection({"mse": MeanSquaredError(), "ssim": SSIM()})
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        # hparams
        self.lr = hparams.lr
        self.beta1 = hparams.beta1
        self.lambda_factor = hparams.lambda_factor

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.generator(x)

    def generator_loss(self, albedo, direct, normal, depth, gt) -> torch.Tensor:

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        prediction = self.discriminator(torch.cat((z, fake), 1))
        y = torch.ones(prediction.size(), device=self.device)

        with torch.no_grad():
            self.train_metrics(fake, gt)

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        gan_loss = self.gan_loss(prediction, y)
        l1_loss = self.l1_loss(fake, gt)

        return gan_loss + self.lambda_factor * l1_loss

    def discriminator_loss(self, albedo, direct, normal, depth, gt) -> torch.Tensor:

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
            self.log("Loss/G", result, on_step=False, on_epoch=True)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_loss(*batch)
            self.log("Loss/D", result, on_step=False, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx == 0:
            logger = self.logger.experiment
            logger.add_images("Validation/fake", fake, self.current_epoch)
            logger.add_images("Validation/real", gt, self.current_epoch)

        with torch.no_grad():
            self.val_metrics(fake, gt)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx == 0:
            logger = self.logger.experiment
            logger.add_images("Test/fake", fake, self.current_epoch)
            logger.add_images("Test/real", gt, self.current_epoch)

        with torch.no_grad():
            self.test_metrics(fake, gt)

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt_d = Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        opt_g = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_g, opt_d]

    def get_progress_bar_dict(self):
        # don't show the train loss
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items