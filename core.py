from pathlib import Path

import cv2
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import MeanSquaredError, MetricCollection
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from common import denormalize, hdr2ldr, ldr2hdr, ssim_to_rgb, to_display, weights_init
from metrics.ssim import SSIM
from metrics.tracker import Tracker
from models.discriminator import Discriminator
from models.generator import Generator
from plot import save_score_histogram


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
        metrics = MetricCollection(
            {"mse": MeanSquaredError(), "ssim": SSIM(), "psnr": PeakSignalNoiseRatio(data_range=1)}
        )
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

    def generator_loss(self, diffuse, direct, normal, depth, gi, indirect, batch_idx) -> torch.Tensor:

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
        fake = self.generator(z)

        target = gi if self.hparams.use_global else indirect

        if batch_idx == 0:
            logger = self.logger.experiment
            ldr_img = to_display(direct, gi, indirect, fake, self.hparams.use_global)
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

    def discriminator_loss(self, diffuse, direct, normal, depth, gi, indirect) -> torch.Tensor:

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
        fake = self.generator(z)

        target = gi if self.hparams.use_global else indirect

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
        self.eval_step(batch, batch_idx, is_test=False)

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, is_test=False)

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, is_test=True)

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, is_test=True)

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

    def eval_step(self, batch, batch_idx, is_test=True):
        eval_tracker = self.test_tracker if is_test else self.val_tracker
        eval_metrics = self.test_metrics if is_test else self.val_metrics

        diffuse, direct, normal, depth, gi, indirect = batch
        target = gi if self.hparams.use_global else indirect

        z = direct if self.hparams.local_buffer_only else torch.cat((diffuse, direct, normal, depth), 1)
        fake = self.generator(z)

        fake_toned = denormalize(fake)
        target_toned = denormalize(target)

        # compute metrics
        per_image_ssim = structural_similarity_index_measure(fake_toned, target_toned, reduction="none", data_range=1)
        eval_tracker(per_image_ssim.mean(dim=(1, 2, 3)), (direct, gi, indirect, fake, target))

        eval_metrics(fake_toned, target_toned)
        self.log_dict(eval_metrics, on_step=False, on_epoch=True)

        # write scores and object to file
        if is_test:
            self.log_batch_to_file("data.csv", batch_idx, fake_toned, target_toned)

            if not self.hparams.use_global:
                gi_toned = denormalize(gi)
                direct_toned = denormalize(direct)

                fake_untoned = ldr2hdr(fake_toned)
                direct_untoned = ldr2hdr(direct_toned)

                fake_gi = fake_untoned + direct_untoned
                fake_gi_toned = hdr2ldr(fake_gi)

                self.test_metrics_global(fake_gi_toned, gi_toned)
                self.log_dict(self.test_metrics_global, on_step=False, on_epoch=True)
                self.log_batch_to_file("data_global.csv", batch_idx, fake_gi_toned, gi_toned)

        # log samples
        if batch_idx % 128 == 0:
            logger = self.logger.experiment
            name = "Test" if is_test else "Validation"

            ldr_img = to_display(direct, gi, indirect, fake, self.hparams.use_global)

            logger.add_image(f"{name}/{batch_idx}", ldr_img, self.current_epoch)
            logger.add_image(f"{name}/{batch_idx}_ssim", ssim_to_rgb(per_image_ssim), self.current_epoch)

    def eval_epoch_end(self, outputs, is_test=True):
        eval_tracker = self.test_tracker if is_test else self.val_tracker
        stats = eval_tracker.compute()
        eval_tracker.reset()

        logger = self.logger.experiment
        name = "Test" if is_test else "Validation"

        for sample in ("best", "worst"):
            direct, gi, indirect, fake, target = stats[f"{sample}_buffers"]

            fake_toned = denormalize(fake)
            target_toned = denormalize(target)

            sample_psnr = peak_signal_noise_ratio(fake_toned, target_toned, data_range=1)
            sample_mse = mean_squared_error(fake_toned, target_toned)
            ssim_windows = structural_similarity_index_measure(fake_toned, target_toned, reduction="none", data_range=1)

            ldr_img = to_display(direct, gi, indirect, fake, self.hparams.use_global)
            logger.add_image(f"{name}/{sample}_sample", ldr_img, self.current_epoch)
            logger.add_image(f"{name}/{sample}_ssim", ssim_to_rgb(ssim_windows), self.current_epoch)

            self.log_dict(
                {
                    f"{name}_sample/{sample}_ssim": stats[f"{sample}_value"],
                    f"{name}_sample/{sample}_psnr": sample_psnr,
                    f"{name}_sample/{sample}_mse": sample_mse,
                },
                on_step=False,
                on_epoch=True,
            )

        if is_test:
            root = Path(self.trainer.logger.log_dir)
            save_score_histogram(root / "data.csv")
            if not self.hparams.use_global:
                save_score_histogram(root / "data_global.csv")

    def log_batch_to_file(self, filename, batch_idx, fake, target):
        test_folders = self.trainer.datamodule.test_dataloader().dataset.valid_folders
        results_path = Path(self.trainer.logger.log_dir) / filename

        # compute metrics
        ssims = structural_similarity_index_measure(fake, target, reduction="none", data_range=1).mean(dim=(1, 2, 3))
        psnrs = peak_signal_noise_ratio(fake, target, reduction="none", dim=(1, 2, 3), data_range=1)
        mses = [mean_squared_error(fake[i], target[i]) for i in range(len(fake))]

        # write header if it doesn't exist
        if not results_path.exists():
            with results_path.open("w") as f:
                f.write("object,ssim,mse,psnr\n")

        # get objects present for each sample in current batch
        objects = ["buddha", "bunny", "cube", "dragon", "sphere"]
        batch_folder_id = batch_idx * self.hparams.batch_size
        batch_folders = test_folders[batch_folder_id : batch_folder_id + self.hparams.batch_size]
        present = ["_".join(obj for obj in objects if obj in folder.stem) for folder in batch_folders]

        # for each sample in the current batch write metrics and the objects present
        batch = [f"{obj},{ssim},{mse},{psnr}\n" for (obj, ssim, mse, psnr) in zip(present, ssims, mses, psnrs)]
        with results_path.open("a") as f:
            f.writelines(batch)
