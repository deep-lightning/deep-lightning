import time
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from core import CGan
from datamodule import DataModule


def main(args):
    # initialize model
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    if args.ckpt:
        model = CGan.load_from_checkpoint(args.ckpt, **kwargs)
        print("Loaded checkpoint")
    else:
        kwargs["use_global"] = kwargs.get("use_global", False)
        kwargs["local_buffer_only"] = kwargs.get("local_buffer_only", False)
        model = CGan(**kwargs)
    new_hparams = Namespace(**model.hparams)

    hpar_dict = vars(new_hparams)
    data = DataModule(
        hpar_dict.get("dataset"),
        hpar_dict.get("batch_size"),
        hpar_dict.get("num_workers"),
        hpar_dict.get("data_regex"),
    )

    # initialize a trainer
    callbacks = [
        ModelCheckpoint(
            monitor="Validation/ssim",
            mode="max",
            save_top_k=5,
            every_n_epochs=50,
            auto_insert_metric_name=False,
            filename="ssim_epoch={epoch:02d}-psnr={Validation/psnr:.4f}-ssim={Validation/ssim:.4f}-mse={Validation/mse:.6f}",
        ),
        ModelCheckpoint(
            monitor="Validation/ssim",
            mode="max",
            auto_insert_metric_name=False,
            filename="ssim_epoch={epoch:02d}-psnr={Validation/psnr:.4f}-ssim={Validation/ssim:.4f}-mse={Validation/mse:.6f}",
        ),
    ]
    trainer = pl.Trainer.from_argparse_args(new_hparams, callbacks=callbacks)

    # Train the model
    if new_hparams.train:
        trainer.fit(model, data)

    # Test the model
    if new_hparams.test:
        trainer.test(model, data)

    # Predict with the model
    if new_hparams.predict:
        trainer.predict(model, data)

    # Benchmark the model
    if new_hparams.bench:
        model.eval()
        data.setup("predict")
        data.num_workers = 0  # disable parallel loading
        loader = data.predict_dataloader()

        bench_time = 30  # seconds

        with torch.no_grad():
            # cpu pytorch
            model_cpu = model.cpu()
            iters = 0
            total_time = 0
            total_data_time = 0
            total_model_time = 0
            items = iter(loader)
            start_time = time.perf_counter()
            while total_time < bench_time:
                try:
                    start_data = time.perf_counter()
                    batch = next(items)
                    end_data = time.perf_counter()
                    total_data_time += end_data - start_data

                    start = time.perf_counter()
                    result = model_cpu(batch)
                    end = time.perf_counter()
                    total_model_time += end - start

                    iters += 1
                    total_time = time.perf_counter() - start_time
                except StopIteration:
                    break
            print(
                f"Preprocessing average time: {(total_data_time / iters) * 1000} ms on {iters} runs\n"
                f"PyTorch average time on CPU: {(total_model_time / iters) * 1000} ms on {iters} runs\n"
            )

            # gpu pytorch
            if torch.cuda.is_available():
                model_gpu = model.cuda()
                iters = 0
                total_time = 0
                total_data_time = 0
                total_model_time = 0
                items = iter(loader)
                start_time = time.perf_counter()
                while total_time < bench_time:
                    try:
                        start_data = time.perf_counter()
                        batch = next(items)
                        batch_cuda = [x.cuda() for x in batch]
                        torch.cuda.synchronize()
                        end_data = time.perf_counter()
                        total_data_time += end_data - start_data

                        start = time.perf_counter()
                        result = model_gpu(batch_cuda)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        total_model_time += end - start

                        iters += 1
                        total_time = time.perf_counter() - start_time
                    except StopIteration:
                        break
                print(
                    f"Preprocessing average time: {(total_data_time / iters) * 1000} ms on {iters} runs\n"
                    f"PyTorch average time on GPU: {(total_model_time / iters) * 1000} ms on {iters} runs\n"
                )


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # add lightning trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")

    modes = parser.add_argument_group("Script modes")
    modes.add_argument("--train", action="store_true", help="Train a model")
    modes.add_argument("--test", action="store_true", help="Test a model")
    modes.add_argument("--bench", action="store_true", help="Benchmark a model")
    modes.add_argument("--predict", action="store_true", help="Make a prediction")

    opt = parser.add_argument_group("Data options")
    opt.add_argument("--dataset", type=str, help="Folder where the samples are stored")
    opt.add_argument("--batch_size", type=int, default=4, help="Number of samples to use per batch")
    opt.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses to use for data loading")
    opt.add_argument(
        "--data_regex",
        help="Predefined regex for splitting data",
        choices=["vanilla", "positions", "cameras", "lights", "walls", "objects", "all"],
    )

    opt = parser.add_argument_group("Model options")
    opt.add_argument("--n_channel_input", type=int, default=3, help="Number of input channels of each image")
    opt.add_argument("--n_channel_output", type=int, default=3, help="Number of output channels of each image")
    opt.add_argument("--n_generator_filters", type=int, default=64, help="Number of initial generator filters")
    opt.add_argument("--n_discriminator_filters", type=int, default=64, help="Number of initial discriminator filters")
    opt.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate")
    opt.add_argument("--beta1", type=float, default=0.5, help="Adam's beta1")
    opt.add_argument("--beta2", type=float, default=0.999, help="Adam's beta2")
    opt.add_argument("--lambda_factor", type=int, default=100, help="L1 regularization factor")
    opt.add_argument("--use_global", action="store_const", const=True, help="Learn global illumination")
    opt.add_argument("--local_buffer_only", action="store_const", const=True, help="Use only local buffer as input")

    hparams = parser.parse_args()
    hparams.deterministic = True
    main(hparams)
