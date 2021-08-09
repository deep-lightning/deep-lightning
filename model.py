import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, channels_in, channels_out, outer=False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 4, 2, 1)
        self.batch_norm = nn.BatchNorm2d(channels_out) if not outer else None

    @property
    def activation(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.activation(self.conv(x))
        return self.batch_norm(x) if self.batch_norm else x


class EncoderG(Encoder):
    def __init__(self, channels_in, channels_out, outer) -> None:
        super().__init__(channels_in, channels_out, outer=outer)
        self.activation = nn.LeakyReLU(0.2, True)


class EncoderD(Encoder):
    def __init__(self, channels_in, channels_out, outer, last=False) -> None:
        super().__init__(channels_in, channels_out, outer=outer)
        self.activation = nn.LeakyReLU(0.2, True) if not last else nn.Sigmoid()


class Decoder(nn.Module):
    def __init__(self, channels_in, channels_out, outer=False) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels_in, channels_out, 4, 2, 1)
        self.activation = nn.Tanh() if outer else nn.ReLU(True)
        self.batch_norm = nn.BatchNorm2d(channels_out) if not outer else None

    def forward(self, x):
        x = self.activation(self.deconv(x))
        return self.batch_norm(x) if self.batch_norm else x


class Gen(nn.Module):
    def __init__(self, channels_in, channels_out, n_filters):
        super().__init__()
        self.encoder1 = EncoderG(channels_in, n_filters, outer=True)
        self.encoder2 = EncoderG(n_filters, n_filters * 2)
        self.encoder3 = EncoderG(n_filters * 2, n_filters * 4)
        self.encoder4 = EncoderG(n_filters * 4, n_filters * 8)
        self.encoder5 = EncoderG(n_filters * 8, n_filters * 8)
        self.encoder6 = EncoderG(n_filters * 8, n_filters * 8)
        self.encoder7 = EncoderG(n_filters * 8, n_filters * 8)
        self.encoder8 = EncoderG(n_filters * 8, n_filters * 8)

        self.decoder1 = Decoder(n_filters * 8, n_filters * 8)
        self.decoder2 = Decoder(n_filters * 8 * 2, n_filters * 8)
        self.decoder3 = Decoder(n_filters * 8 * 2, n_filters * 8)
        self.decoder4 = Decoder(n_filters * 8 * 2, n_filters * 8)
        self.decoder5 = Decoder(n_filters * 8 * 2, n_filters * 4)
        self.decoder6 = Decoder(n_filters * 4 * 2, n_filters * 2)
        self.decoder7 = Decoder(n_filters * 2 * 2, n_filters)
        self.decoder8 = Decoder(n_filters * 2, channels_out, outer=True)

    def forward(self, x):
        # encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        enc8 = self.encoder8(enc7)

        # decoder + skip connections
        dec1 = torch.cat((self.decoder1(enc8), enc7), 1)
        dec2 = torch.cat((self.decoder2(dec1), enc6), 1)
        dec3 = torch.cat((self.decoder3(dec2), enc5), 1)
        dec4 = torch.cat((self.decoder4(dec3), enc4), 1)
        dec5 = torch.cat((self.decoder5(dec4), enc3), 1)
        dec6 = torch.cat((self.decoder6(dec5), enc2), 1)
        dec7 = torch.cat((self.decoder7(dec6), enc1), 1)
        dec8 = self.decoder8(dec7)

        return dec8


class Disc(nn.Module):
    def __init__(self, channels_in, channels_out, n_filters):
        super().__init__()
        self.encoder1 = EncoderD(channels_in + channels_out, n_filters, outer=True)
        self.encoder2 = EncoderD(n_filters, n_filters * 2)
        self.encoder3 = EncoderD(n_filters * 2, n_filters * 4)
        self.encoder4 = EncoderD(n_filters * 4, n_filters * 8)
        self.encoder5 = EncoderD(n_filters * 8, n_filters * 8, outer=True, last=True)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        return enc5


class G(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 4, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(n_filters * 4 * 2, n_filters * 2, 4, 2, 1)
        self.deconv7 = nn.ConvTranspose2d(n_filters * 2 * 2, n_filters, 4, 2, 1)
        self.deconv8 = nn.ConvTranspose2d(n_filters * 2, n_channel_output, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(n_filters)
        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.batch_norm8(self.conv5(self.leaky_relu(encoder4)))
        encoder6 = self.batch_norm8(self.conv6(self.leaky_relu(encoder5)))
        encoder7 = self.batch_norm8(self.conv7(self.leaky_relu(encoder6)))
        encoder8 = self.conv8(self.leaky_relu(encoder7))

        decoder1 = self.dropout(self.batch_norm8(self.deconv1(self.relu(encoder8))))
        decoder1 = torch.cat((decoder1, encoder7), 1)

        decoder2 = self.dropout(self.batch_norm8(self.deconv2(self.relu(decoder1))))
        decoder2 = torch.cat((decoder2, encoder6), 1)

        decoder3 = self.dropout(self.batch_norm8(self.deconv3(self.relu(decoder2))))
        decoder3 = torch.cat((decoder3, encoder5), 1)

        decoder4 = self.batch_norm8(self.deconv4(self.relu(decoder3)))
        decoder4 = torch.cat((decoder4, encoder4), 1)

        decoder5 = self.batch_norm4(self.deconv5(self.relu(decoder4)))
        decoder5 = torch.cat((decoder5, encoder3), 1)

        decoder6 = self.batch_norm2(self.deconv6(self.relu(decoder5)))
        decoder6 = torch.cat((decoder6, encoder2), 1)

        decoder7 = self.batch_norm(self.deconv7(self.relu(decoder6)))
        decoder7 = torch.cat((decoder7, encoder1), 1)

        decoder8 = self.deconv8(self.relu(decoder7))
        output = self.tanh(decoder8)
        return output


class D(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input + n_channel_output, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, 1, 4, 1, 1)

        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.conv5(self.leaky_relu(encoder4))
        output = self.sigmoid(encoder5)
        return output
