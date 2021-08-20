from enum import Enum

import torch
import torch.nn as nn


class Position(Enum):
    INNER = "inner"
    OUTER = "outer"
    MIDDLE = "middle"


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, position: Position):
        super().__init__()
        layers = nn.ModuleList()

        if position is not Position.OUTER:
            layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1))

        if position is Position.MIDDLE:
            layers.append(nn.BatchNorm2d(out_ch))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool, position: Position):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))

        if position is not Position.OUTER:
            layers.append(nn.BatchNorm2d(out_ch))

        if dropout:
            layers.append(nn.Dropout(0.5))

        if position is Position.OUTER:
            layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        in_enc: int,
        out_enc: int,
        in_dec: int,
        out_dec: int,
        dropout: bool = False,
        sub_block: "EncoderDecoder" = None,
        position: Position = Position.MIDDLE,
    ) -> None:
        super().__init__()
        self.outer = position is Position.OUTER
        self.encoder = EncoderBlock(in_enc, out_enc, position)
        self.sub_block = sub_block
        self.decoder = DecoderBlock(in_dec, out_dec, dropout, position)

    def forward(self, x):
        y = self.encoder(x)
        y = self.sub_block(y) if self.sub_block else y
        y = self.decoder(y)
        return y if self.outer else torch.cat([x, y], 1)


class Generator(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_filt: int):
        super().__init__()
        block = EncoderDecoder(n_filt * 8, n_filt * 8, n_filt * 8, n_filt * 8, position=Position.INNER)
        block = EncoderDecoder(n_filt * 8, n_filt * 8, n_filt * 8 * 2, n_filt * 8, dropout=True, sub_block=block)
        block = EncoderDecoder(n_filt * 8, n_filt * 8, n_filt * 8 * 2, n_filt * 8, dropout=True, sub_block=block)
        block = EncoderDecoder(n_filt * 8, n_filt * 8, n_filt * 8 * 2, n_filt * 8, dropout=True, sub_block=block)
        block = EncoderDecoder(n_filt * 4, n_filt * 8, n_filt * 8 * 2, n_filt * 4, sub_block=block)
        block = EncoderDecoder(n_filt * 2, n_filt * 4, n_filt * 4 * 2, n_filt * 2, sub_block=block)
        block = EncoderDecoder(n_filt, n_filt * 2, n_filt * 2 * 2, n_filt, sub_block=block)
        block = EncoderDecoder(in_ch, n_filt, n_filt * 2, out_ch, sub_block=block, position=Position.OUTER)
        self.model = block

    def forward(self, x):
        return self.model(x)
