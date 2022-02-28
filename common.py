from enum import Enum
import torch
from torch import nn

import torchvision
from torchvision import transforms


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


required_images = {
    "diffuse": "diffuse.hdr",
    "local": "local.hdr",
    "normal": "normal.hdr",
    "depth": "z.hdr",
    "global": "global.hdr",
    "indirect": "indirect.hdr",
}


mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

# normalize image in range [0,1] to [-1,1]
normalize = transforms.Normalize(mean.tolist(), std.tolist())

# normalize image in range [-1,1] to [0,1]
denormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def hdr2ldr(image):
    """Scales image from range [0, infinity] to [0,1]"""
    return image / (1 + image)


def ldr2hdr(image):
    """Scales image from range [0,1] to [0, infinity]"""
    return image / (1 - image)


def to_display(
    direct: torch.Tensor, gt: torch.Tensor, indirect: torch.Tensor, fake: torch.Tensor, use_global: bool = False
):
    gt_un = denormalize(gt)
    fake_un = denormalize(fake)

    gt_untoned = ldr2hdr(gt_un)
    fake_untoned = ldr2hdr(fake_un)

    if use_global:
        diff = torch.abs(gt_untoned - fake_untoned)
        batch = torch.cat((fake_untoned, gt_untoned, diff), 3)
    else:
        direct_un = denormalize(direct)
        indirect_un = denormalize(indirect)

        direct_untoned = ldr2hdr(direct_un)
        indirect_untoned = ldr2hdr(indirect_un)

        fake_gt = fake_untoned + direct_untoned
        real_gt = indirect_untoned + direct_untoned

        diff = torch.abs(indirect_untoned - fake_untoned)
        batch = torch.cat((fake_untoned, indirect_untoned, diff, fake_gt, real_gt, gt_untoned), 3)

    batch_hdr = hdr2ldr(batch)
    gamma_batch = (batch_hdr ** (1 / 2.2)).clip(0, 1)
    return torchvision.utils.make_grid(gamma_batch, nrow=1)
