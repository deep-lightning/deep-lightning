import torch
import torch.nn as nn
import torchvision

import cv2
import numpy as np

from pathlib import Path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def denormalize(input: torch.Tensor) -> torch.Tensor:
    undo = (input * 0.5) + 0.5
    return undo / (1 - undo)  # watchout zero div, almost sure cant happen


def to_ldr(direct: torch.Tensor, gt: torch.Tensor, indirect: torch.Tensor, fake: torch.Tensor, path: Path):
    direct_un = denormalize(direct)
    gt_un = denormalize(gt)
    indirect_un = denormalize(indirect)
    fake_un = denormalize(fake)

    fake_gt = fake_un + direct_un
    real_gt = indirect_un + direct_un

    batch = torch.cat((fake_un, indirect_un, fake_gt, real_gt, gt_un), 3)

    epoch = torchvision.utils.make_grid(batch, 1)
    image = epoch.permute(1, 2, 0).cpu().detach().numpy()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)

    images = []
    base = batch.permute(0, 2, 3, 1).cpu().detach().numpy()
    for img in base:
        tonemap = cv2.createTonemapDrago(gamma=1.4, saturation=0.6)
        res_base = tonemap.process(img)
        res_base_8bit = np.clip(res_base * 255, 0, 255).astype("uint8")
        images.append(res_base_8bit)

    tonemapped_torch = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2))
    return torchvision.utils.make_grid(tonemapped_torch, nrow=1)
