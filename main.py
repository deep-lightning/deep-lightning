from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from core import CGan
from datamodules.diffuse_re import DiffuseDataModule

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="location of train, val and test folders")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--n_channel_input", type=int, default=3, help="number of input channels")
    parser.add_argument("--n_channel_output", type=int, default=3, help="number of output channels")
    parser.add_argument("--n_generator_filters", type=int, default=64, help="num of initial generator filters")
    parser.add_argument("--n_discriminator_filters", type=int, default=64, help="num of initial discriminator filters")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2")
    parser.add_argument("--lambda_factor", type=int, default=100, help="L1 regularization factor")
    parser.add_argument("--num_workers", type=int, default=4, help="number of threads for data loader")
    parser.add_argument("--data_regex", choices=["vanilla", "positions", "lights", "cameras", "objects"])

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.deterministic = True

    callbacks = [
        ModelCheckpoint(
            monitor="Validation/psnr",
            mode="max",
            auto_insert_metric_name=False,
            filename="psnr_epoch={epoch:02d}-psnr={Validation/psnr:.4f}-ssim={Validation/ssim:.4f}-mse={Validation/mse:.6f}",
        ),
        ModelCheckpoint(
            monitor="Validation/ssim",
            mode="max",
            auto_insert_metric_name=False,
            filename="ssim_epoch={epoch:02d}-psnr={Validation/psnr:.4f}-ssim={Validation/ssim:.4f}-mse={Validation/mse:.6f}",
        ),
    ]

    data = DiffuseDataModule(hparams.dataset, hparams.batch_size, hparams.num_workers, hparams.data_regex)

    model = CGan(**vars(hparams))

    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=callbacks)

    trainer.fit(model, data)
    # trainer.test(model, data)
