from enum import Enum

import torch.nn as nn


class Position(Enum):
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, position: Position = Position.MIDDLE):
        super().__init__()
        layers = nn.ModuleList()

        if position is not Position.FIRST:
            layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(in_ch, out_ch, 4, stride, 1))

        if position is Position.MIDDLE:
            layers.append(nn.BatchNorm2d(out_ch))

        if position is Position.LAST:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_filt: int):
        super().__init__()

        self.model = nn.Sequential(
            EncoderBlock(in_ch + out_ch, n_filt, 2, Position.FIRST),
            EncoderBlock(n_filt, n_filt * 2, 2),
            EncoderBlock(n_filt * 2, n_filt * 4, 2),
            EncoderBlock(n_filt * 4, n_filt * 8, 1),
            EncoderBlock(n_filt * 8, 1, 1, Position.LAST),
        )

    def forward(self, x):
        return self.model(x)
