import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from argparse import ArgumentParser

from core import CGan
from datamodules.diffuse import DiffuseDataModule


def main(hparams):
    pl.seed_everything(42, workers=True)

    callbacks = [
        EarlyStopping(monitor="Validation/mse", mode="min", patience=3),
        # EarlyStopping(monitor="val_ssim", mode="max"),
        ModelCheckpoint(monitor="Validation/mse", filename="cgan-{epoch:02d}"),
    ]

    diffuse = DiffuseDataModule(
        hparams.dataset, hparams.batch_size, hparams.num_workers
    )

    model = CGan(hparams)

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        callbacks=callbacks,
        deterministic=True,
        # fast_dev_run=True
        # auto_lr_find=True
        # auto_scale_batch_size="power",
    )

    trainer.fit(model, diffuse)
    trainer.test(model, diffuse)


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
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2")
    parser.add_argument(
        "--lambda_factor", type=int, default=100, help="L1 regularization factor"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of threads for data loader"
    )
    args = parser.parse_args()
    main(args)
