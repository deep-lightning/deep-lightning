import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam

from utils import to_display, weights_init

from models.generator import Generator
from models.discriminator import Discriminator

from torchmetrics import MetricCollection, MeanSquaredError, PSNR
from torchmetrics.functional import ssim, mean_squared_error, psnr
from metrics.ssim import SSIM

from utils import denormalize


class CGan(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        multiplier = 1 if self.hparams.local_buffer_only else 4

        self.generator = Generator(
            self.hparams.n_channel_input * multiplier,
            self.hparams.n_channel_output,
            self.hparams.n_generator_filters,
        )
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(
            self.hparams.n_channel_input * multiplier,
            self.hparams.n_channel_output,
            self.hparams.n_discriminator_filters,
        )
        self.discriminator.apply(weights_init)

        # losses
        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()

        # metrics
        metrics = MetricCollection({"mse": MeanSquaredError(), "ssim": SSIM(), "psnr": PSNR(data_range=1)})
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        self.best_val = None
        self.best_val_ssim = 0

        self.worst_val = None
        self.worst_val_ssim = float("inf")

    def forward(self, x):
        return self.generator(x)

    def generator_loss(self, albedo, direct, normal, depth, gt, indirect, batch_idx) -> torch.Tensor:

        z = direct if self.hparams.local_buffer_only else torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        target = gt if self.hparams.use_global else indirect

        if batch_idx == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image("Train", ldr_img, self.current_epoch)

        prediction = self.discriminator(torch.cat((z, fake), 1))
        y = torch.ones(prediction.size(), device=self.device)

        with torch.no_grad():
            self.train_metrics(denormalize(fake), denormalize(target))

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        gan_loss = self.gan_loss(prediction, y)
        l1_loss = self.l1_loss(fake, target) * self.hparams.lambda_factor

        self.log_dict(
            {"Train/gan_loss": gan_loss, "Train/l1_loss": l1_loss},
            on_step=False,
            on_epoch=True,
        )

        return gan_loss + l1_loss

    def discriminator_loss(self, albedo, direct, normal, depth, gt, indirect) -> torch.Tensor:

        z = direct if self.hparams.local_buffer_only else torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        target = gt if self.hparams.use_global else indirect

        # train on real data
        real_data = torch.cat((z, target), 1)
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
            on_step=True,
            on_epoch=False,
        )

        # gradient backprop & optimize ONLY D's parameters
        return (real_loss + fake_loss) * 0.5

    def training_step(self, batch, batch_idx, optimizer_idx):
        result = None

        # train discriminator
        if optimizer_idx == 0:
            result = self.discriminator_loss(*batch)
            self.log("Loss/D", result, on_step=True, on_epoch=False)

        # train generator
        if optimizer_idx == 1:
            result = self.generator_loss(*batch, batch_idx)
            self.log("Loss/G", result, on_step=True, on_epoch=False)

        return result

    def validation_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt, indirect = batch
        target = gt if self.hparams.use_global else indirect

        z = direct if self.hparams.local_buffer_only else torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        individual_ssim = ssim(denormalize(fake), denormalize(target), reduction="none", data_range=1).mean(dim=0)
        max_ssim = individual_ssim.max()
        min_ssim = individual_ssim.min()
        if max_ssim > self.best_val_ssim:
            self.best_val_ssim = max_ssim
            self.best_val = (direct, gt, indirect, fake, target)
        if min_ssim < self.worst_val_ssim:
            self.worst_val_ssim = min_ssim
            self.worst_val = (direct, gt, indirect, fake, target)

        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image(f"Validation/{batch_idx}", ldr_img, self.current_epoch)

            met_ssim = ssim(denormalize(fake), denormalize(target), reduction="none", data_range=1)
            for i in range(len(target)):
                met_psnr = psnr(denormalize(fake[i]), denormalize(target[i]), data_range=1)
                met_mse = mean_squared_error(denormalize(fake[i]), denormalize(target[i]))

                logger.add_text(
                    f"Validation/{batch_idx}_{i}",
                    (
                        f"* ssim: {met_ssim[i].mean().cpu().item()}\n"
                        f"* psnr: {met_psnr.mean().cpu().item()}\n"
                        f"* mse: {met_mse.mean().cpu().item()}\n"
                    ),
                    self.current_epoch,
                )

        with torch.no_grad():
            self.val_metrics(denormalize(fake), denormalize(target))

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs) -> None:

        direct, gt, indirect, fake, target = self.best_val

        logger = self.logger.experiment
        ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
        logger.add_image(f"Validation/best_sample", ldr_img, self.current_epoch)

        met_ssim = ssim(denormalize(fake), denormalize(target), reduction="none", data_range=1)
        for i in range(len(target)):
            met_psnr = psnr(denormalize(fake[i]), denormalize(target[i]), data_range=1)
            met_mse = mean_squared_error(denormalize(fake[i]), denormalize(target[i]))

            logger.add_text(
                f"Validation/best_sample_{i}",
                (
                    f"* ssim: {met_ssim[i].mean().cpu().item()}\n"
                    f"* psnr: {met_psnr.mean().cpu().item()}\n"
                    f"* mse: {met_mse.mean().cpu().item()}\n"
                ),
                self.current_epoch,
            )

        direct, gt, indirect, fake, target = self.worst_val

        ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
        logger.add_image(f"Validation/worst_sample", ldr_img, self.current_epoch)

        met_ssim = ssim(denormalize(fake), denormalize(target), reduction="none", data_range=1)
        for i in range(len(target)):
            met_psnr = psnr(denormalize(fake[i]), denormalize(target[i]), data_range=1)
            met_mse = mean_squared_error(denormalize(fake[i]), denormalize(target[i]))

            logger.add_text(
                f"Validation/worst_sample_{i}",
                (
                    f"* ssim: {met_ssim[i].mean().cpu().item()}\n"
                    f"* psnr: {met_psnr.mean().cpu().item()}\n"
                    f"* mse: {met_mse.mean().cpu().item()}\n"
                ),
                self.current_epoch,
            )

        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        albedo, direct, normal, depth, gt, indirect = batch
        target = gt if self.hparams.use_global else indirect

        z = direct if self.hparams.local_buffer_only else torch.cat((albedo, direct, normal, depth), 1)
        fake = self.generator(z)

        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image(f"Test/{batch_idx}", ldr_img, self.current_epoch)

        with torch.no_grad():
            self.test_metrics(denormalize(fake), denormalize(target))

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_g = Adam(self.generator.parameters(), lr=self.hparams.lr, betas=betas)
        opt_d = Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=betas)
        return [opt_d, opt_g]

    def get_progress_bar_dict(self):
        # don't show the train loss
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
