import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam

from utils import to_display, weights_init

from models.generator import Generator
from models.discriminator import Discriminator

from torchmetrics import MetricCollection, MeanSquaredError, PSNR
from metrics.ssim import SSIM


class CGan(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            self.hparams.n_channel_input * 4,
            self.hparams.n_channel_output,
            self.hparams.n_generator_filters,
        )
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(
            self.hparams.n_channel_input * 4,
            self.hparams.n_channel_output,
            self.hparams.n_discriminator_filters,
        )
        self.discriminator.apply(weights_init)

        # losses
        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()

        # metrics
        metrics = MetricCollection({"mse": MeanSquaredError(), "ssim": SSIM(), "psnr": PSNR(data_range=2)})
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

    def forward(self, x):
        return self.generator(x)

    def generator_loss(self, albedo, direct, normal, depth, gt, indirect, batch_idx) -> torch.Tensor:

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake)
            logger.add_image("Train", ldr_img, self.current_epoch)

        prediction = self.discriminator(torch.cat((z, fake), 1))
        y = torch.ones(prediction.size(), device=self.device)

        with torch.no_grad():
            self.train_metrics(fake, indirect)

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        gan_loss = self.gan_loss(prediction, y)
        l1_loss = self.l1_loss(fake, indirect)

        self.log_dict(
            {"Train/gan_loss": gan_loss, "Train/l1_loss": l1_loss},
            on_step=False,
            on_epoch=True,
        )

        return gan_loss + self.hparams.lambda_factor * l1_loss

    def discriminator_loss(self, albedo, direct, normal, depth, gt, indirect) -> torch.Tensor:

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        # train on real data
        real_data = torch.cat((z, indirect), 1)
        prediction_real = self.discriminator(real_data)
        y_real = torch.ones(prediction_real.size(), device=self.device)

        # calculate error and backpropagate
        real_loss = self.gan_loss(prediction_real, y_real)

        # train on fake data
        fake_data = torch.cat((z, fake), 1)
        prediction_fake = self.discriminator(fake_data.detach())
        y_fake = torch.zeros(prediction_real.size(), device=self.device)

        # calculate error and backpropagate
        fake_loss = self.gan_loss(prediction_fake, y_fake)

        self.log_dict(
            {
                "Train/D(x)": prediction_real.mean(),
                "Train/D(G(z))": prediction_fake.mean(),
            },
            on_step=False,
            on_epoch=True,
        )

        # gradient backprop & optimize ONLY D's parameters
        return (real_loss + fake_loss) * 0.5

    def training_step(self, batch, batch_idx, optimizer_idx):
        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_loss(*batch, batch_idx)
            self.log("Loss/G", result, on_step=False, on_epoch=True)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_loss(*batch)
            self.log("Loss/D", result, on_step=False, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt, indirect = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake)
            logger.add_image(f"Validation/{batch_idx}", ldr_img, self.current_epoch)

        with torch.no_grad():
            self.val_metrics(fake, indirect)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt, indirect = batch

        z = torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake)
            logger.add_image(f"Test/{batch_idx}", ldr_img, self.current_epoch)

        with torch.no_grad():
            self.test_metrics(fake, indirect)

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_g = Adam(self.generator.parameters(), lr=self.hparams.lr, betas=betas)
        opt_d = Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=betas)
        return [opt_g, opt_d]

    def get_progress_bar_dict(self):
        # don't show the train loss
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
