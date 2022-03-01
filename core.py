import cv2
import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics import MetricCollection, MeanSquaredError

from common import to_display, weights_init, ldr2hdr, hdr2ldr, denormalize
from metrics.tracker import Tracker
from metrics.ssim import SSIM

from models.generator import Generator
from models.discriminator import Discriminator


class CGan(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        """Initialize CGan model

        Expected in kwargs:
            n_channel_input: Number of input channels
            n_channel_output: Number of output channels
            n_generator_filters: Number of initial generator filters
            n_discriminator_filters: Number of initial discriminator filters
            lr: Initial learning rate
            beta1: Adam's beta1
            beta2: Adam's beta2
            lambda_factor: L1 regularization factor
            use_global: Learn global illumination
            local_buffer_only: Use only local buffer as input
        """
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
        self.test_metrics_global = metrics.clone(prefix="Test_global/")

        self.val_tracker = Tracker()
        self.test_tracker = Tracker()

    def forward(self, batch):
        diffuse, direct, normal, depth = batch
        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
        return self.generator(z)

    def generator_loss(self, diffuse, direct, normal, depth, gt, indirect, batch_idx) -> torch.Tensor:

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
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

    def discriminator_loss(self, diffuse, direct, normal, depth, gt, indirect) -> torch.Tensor:

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
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
        diffuse, direct, normal, depth, gt, indirect = batch
        target = gt if self.hparams.use_global else indirect

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
        fake = self.generator(z)

        per_image_ssim = torch.mean(
            structural_similarity_index_measure(denormalize(fake), denormalize(target), reduction="none", data_range=1),
            dim=(1, 2, 3),
        )

        with torch.no_grad():
            self.val_tracker(per_image_ssim, (direct, gt, indirect, fake, target))
            self.val_metrics(denormalize(fake), denormalize(target))

        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image(f"Validation/{batch_idx}", ldr_img, self.current_epoch)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs) -> None:

        stats = self.val_tracker.compute()
        self.val_tracker.reset()

        for x in ("best", "worst"):
            direct, gt, indirect, fake, target = stats[f"{x}_buffers"]

            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image(f"Validation/{x}_sample", ldr_img, self.current_epoch)

            sample_psnr = peak_signal_noise_ratio(denormalize(fake), denormalize(target), data_range=1)
            sample_mse = mean_squared_error(denormalize(fake), denormalize(target))

            self.log_dict(
                {
                    f"Validation_sample/{x}_ssim": stats[f"{x}_value"],
                    f"Validation_sample/{x}_psnr": sample_psnr,
                    f"Validation_sample/{x}_mse": sample_mse,
                },
                on_step=False,
                on_epoch=True,
            )

        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        diffuse, direct, normal, depth, gt, indirect = batch
        target = gt if self.hparams.use_global else indirect

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
        fake = self.generator(z)

        per_image_ssim = torch.mean(
            structural_similarity_index_measure(denormalize(fake), denormalize(target), reduction="none", data_range=1),
            dim=(1, 2, 3),
        )

        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image(f"Test/{batch_idx}", ldr_img, self.current_epoch)

        with torch.no_grad():
            self.test_tracker(per_image_ssim, (direct, gt, indirect, fake, target))
            self.test_metrics(denormalize(fake), denormalize(target))
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

            if not self.hparams.use_global:
                fake_un = denormalize(fake)
                direct_un = denormalize(direct)

                fake_untoned = ldr2hdr(fake_un)
                direct_untoned = ldr2hdr(direct_un)

                fake_gt = fake_untoned + direct_untoned

                self.test_metrics_global(hdr2ldr(fake_gt), denormalize(gt))
                self.log_dict(self.test_metrics_global, on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs) -> None:

        stats = self.test_tracker.compute()
        self.test_tracker.reset()

        for x in ("best", "worst"):
            direct, gt, indirect, fake, target = stats[f"{x}_buffers"]

            logger = self.logger.experiment
            ldr_img = to_display(direct, gt, indirect, fake, self.hparams.use_global)
            logger.add_image(f"Test/{x}_sample", ldr_img, self.current_epoch)

            sample_psnr = peak_signal_noise_ratio(denormalize(fake), denormalize(target), data_range=1)
            sample_mse = mean_squared_error(denormalize(fake), denormalize(target))

            self.log_dict(
                {
                    f"Test_sample/{x}_ssim": stats[f"{x}_value"],
                    f"Test_sample/{x}_psnr": sample_psnr,
                    f"Test_sample/{x}_mse": sample_mse,
                },
                on_step=False,
                on_epoch=True,
            )

        return super().test_epoch_end(outputs)

    def on_predict_epoch_end(self, results):
        flat_results = torch.cat([pred for batch in results for pred in batch])
        predict_folders = self.trainer.datamodule.predict_dataloader().dataset.valid_folders
        for idx, output in enumerate(flat_results):
            output = ldr2hdr(denormalize(output))

            # save output
            image_torch_final = (output.permute(1, 2, 0)).detach().cpu().numpy()
            image_bgr = cv2.cvtColor(image_torch_final, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str((predict_folders[idx] / "output.hdr").resolve()), image_bgr)

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
