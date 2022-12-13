import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def conv_output_length(input_length, kernel_size, padding=0, dilation=1, stride=1):
    return int((input_length + 2*padding - dilation * (kernel_size-1)-1)/stride)+1


class Encoder(nn.Module):
    """
    The encoder module for a simple 1D VAE.
    """
    def __init__(self, conv_length):
        super(Encoder, self).__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.linear = nn.Linear(conv_length*32, 256)
        self.mu = nn.Linear(256, 2)
        self.sigma = nn.Linear(256, 2)

    def forward(self, x):
        x = self.encoder_block(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, conv_length):
        super().__init__()

        self.decoder_linear = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(True),
            nn.Linear(256, 32*conv_length),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, conv_length))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAE(nn.Module):
    def __init__(self, length):
        super(VariationalAE, self).__init__()
        output_length = conv_output_length(conv_output_length(conv_output_length(length, 3, padding=1, stride=2),
                                                              3, padding=1, stride=2), 3, stride=2)
        self.encoder = Encoder(output_length)
        self.decoder = Decoder(output_length)

    def reparam(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        mu, sigma = self.encoder(x)
        return self.reparam(mu, sigma), mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, sigma = self.encode(x)
        return self.decode(z), mu, sigma
