import time
from argparse import ArgumentParser, Namespace
import cv2

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from core import CGan
from datamodule import DataModule
import utils

from torchvision import transforms

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # add lightning trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)

    modes = parser.add_argument_group("Script modes")
    modes.add_argument("--train", action="store_true", help="Train a model")
    modes.add_argument("--test", action="store_true", help="Test a model")
    modes.add_argument("--bench", action="store_true", help="Benchmark a model")
    modes.add_argument("--predict", action="store_true", help="Make a prediction")

    options = parser.add_argument_group("Script options")
    options.add_argument("--batch_size", type=int, default=4, help="batch size")
    options.add_argument("--n_channel_input", type=int, default=3, help="number of input channels")
    options.add_argument("--n_channel_output", type=int, default=3, help="number of output channels")
    options.add_argument("--n_generator_filters", type=int, default=64, help="num of initial generator filters")
    options.add_argument("--n_discriminator_filters", type=int, default=64, help="num of initial discriminator filters")
    options.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    options.add_argument("--beta1", type=float, default=0.5, help="beta1")
    options.add_argument("--beta2", type=float, default=0.999, help="beta2")
    options.add_argument("--lambda_factor", type=int, default=100, help="L1 regularization factor")
    options.add_argument("--num_workers", type=int, default=4, help="number of threads for data loader")
    options.add_argument("--data_regex", choices=["vanilla", "positions", "lights", "cameras", "objects", "walls"])
    options.add_argument("--ckpt", type=str, help="Checkpoint path")
    options.add_argument("--use_global", action="store_true", help="Learn global illumination")
    options.add_argument("--local_buffer_only", action="store_true", help="Use only local buffer as input")

    inference = parser.add_argument_group("Script inference")
    inference.add_argument("--diffuse", type=str, help="path to diffuse")
    inference.add_argument("--local", type=str, help="path to local")
    inference.add_argument("--normal", type=str, help="path to normal")
    inference.add_argument("--depth", type=str, help="path to depth")

    (partial, _) = parser.parse_known_args()
    options.add_argument("--dataset", required=not partial.ckpt, help="location of train, val and test folders")

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

    # initialize model
    kwargs = {k: v for k, v in vars(hparams).items() if v is not None}
    if hparams.ckpt:
        model = CGan.load_from_checkpoint(hparams.ckpt, **kwargs)
        print("Loaded checkpoint")
    else:
        model = CGan(**kwargs)
    new_hparams = Namespace(**model.hparams)

    data = DataModule(new_hparams.dataset, new_hparams.batch_size, new_hparams.num_workers, new_hparams.data_regex)

    if new_hparams.predict:

        def get_image(path):
            image = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_torch = torch.from_numpy(image_rgb).permute(2, 0, 1)

            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.Lambda(utils.hdr2ldr),
                    utils.normalize,
                ]
            )
            return transform(image_torch).unsqueeze(0)

        # load input images
        diffuse = get_image(new_hparams.diffuse)
        local = get_image(new_hparams.local)
        normal = get_image(new_hparams.normal)
        depth = get_image(new_hparams.depth)

        # concatenate them
        z = torch.cat((diffuse, local, normal, depth), 1)

        # call predict
        model.eval()
        output = model.generator(z)
        output = utils.ldr2hdr(utils.denormalize(output))

        # save output
        image_torch_final = (output[0].permute(1, 2, 0)).detach().cpu().numpy()
        image_bgr = cv2.cvtColor(image_torch_final, cv2.COLOR_RGB2BGR)
        cv2.imwrite("output.hdr", image_bgr)

    print(new_hparams)

    # initialize a trainer
    trainer = pl.Trainer.from_argparse_args(new_hparams, callbacks=callbacks)

    if new_hparams.auto_lr_find:
        trainer.tune(model, data)

    # Train the model
    if new_hparams.train:
        trainer.fit(model, data)

    # Test the model
    if new_hparams.test:
        trainer.test(model, data)

    # Benchmark the model
    if new_hparams.bench:
        model.eval()
        data.setup("test")
        data.num_workers = 0  # disable parallel loading
        loader = data.test_dataloader()

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
                    albedo, direct, normal, depth, _, _ = next(items)
                    z = torch.cat((albedo, direct, normal, depth), 1)
                    end_data = time.perf_counter()
                    total_data_time += end_data - start_data

                    start = time.perf_counter()

                    result = model_cpu(z)
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
                        albedo, direct, normal, depth, _, _ = next(items)
                        z = torch.cat((albedo, direct, normal, depth), 1)
                        end_data = time.perf_counter()
                        total_data_time += end_data - start_data

                        z_cuda = z.cuda()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = model_gpu(z_cuda)
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
