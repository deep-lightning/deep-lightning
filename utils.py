import torch
from torch import nn

import torchvision
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

# normalize image in range [0,1] to [-1,1]
norm = transforms.Normalize(mean.tolist(), std.tolist())

# normalize image in range [-1,1] to [0,1]
unnorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def hdr2ldr(max_light, image):
    """Applies Reinhard TMO based on http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf"""
    return (image * (1 + image / (max_light ** 2))) / (1 + image)


def ldr2hdr(max_light, image):
    """Reverts Reinhard TMO based on https://www.wolframalpha.com/input/?i=inverse+of+f%28x%29+%3D+%28x*%281%2Bx%2Fa%5E2%29%29%2F%281%2Bx%29"""
    return 0.5 * ((max_light ** 2) * image - (max_light ** 2)) + 0.5 * (
        (
            (max_light ** 2)
            * ((max_light ** 2) * (image ** 2) - 2 * (max_light ** 2) * image + (max_light ** 2) + 4 * image)
        )
        ** 0.5
    )


def to_display(
    direct: torch.Tensor, gt: torch.Tensor, indirect: torch.Tensor, fake: torch.Tensor, use_global: bool = False
):

    direct_un = unnorm(direct)
    gt_un = unnorm(gt)
    indirect_un = unnorm(indirect)
    fake_un = unnorm(fake)

    max_light = gt_un.amax(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
    direct_untoned = ldr2hdr(max_light, direct_un)
    gt_untoned = ldr2hdr(max_light, gt_un)
    indirect_untoned = ldr2hdr(max_light, indirect_un)
    fake_untoned = ldr2hdr(max_light, fake_un)

    fake_gt = fake_untoned + direct_untoned
    real_gt = indirect_untoned + direct_untoned

    if use_global:
        diff = torch.abs(gt_untoned - fake_untoned)
        batch = torch.cat((fake_untoned, gt_untoned, diff), 3)
    else:
        diff = torch.abs(indirect_untoned - fake_untoned)
        batch = torch.cat((fake_untoned, indirect_untoned, diff, fake_gt, real_gt, gt_untoned), 3)
    rein = hdr2ldr(max_light, batch)
    gamma_rein = (rein ** (1 / 2.2)).clip(0, 1)
    return torchvision.utils.make_grid(gamma_rein, nrow=1)
