from torch import nn
import torch
import torch.nn.functional as F


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding,
            bias = bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, num_classes, in_channels: int = 128):
        super().__init__()
        # N x Z x 1 x 1 -> N x 128 x 7 x 7
        self.generation_block_in = GenBlock(in_channels, 128, 7, 1, 0)
        
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, 49),
            nn.ReLU()
        )
        
        self.generation_sequence = nn.Sequential(
            # N x (128 + 1) x 7 x 7 -> N x 64 x 14 x 14
            GenBlock(128 + 1, 64, 4, 2, 1),
            # N x 64 x 14 x 14 -> N x 32 x 28 x 28
            GenBlock(64, 32, 4, 2, 1),
        )
        
        # N x 32 x 28 x 28 -> N x 1 x 28 x 28
        self.output_block = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )


    def forward(self, x, y):
        x = x.view(x.size(0), x.size(1), 1, 1)                          # x: NxZ -> NxZx1x1
        x = self.generation_block_in(x)                                 # x: NxZx1x1 -> Nx128x7x7
        y = self.label_embedding(y)                                     # y: NxCls -> Nx49
        y = y.view(y.size(0), 1, x.size(2), x.size(3))                  # y: Nx49 -> Nx1x7x7

        z = self.generation_sequence(torch.cat([x, y], dim=1))          # z: cat x, y -> Nx129x7x7
        z = self.output_block(z)
        return z


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding,
            bias = bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes, in_channels: int = 1):
        super().__init__()
        # NxCls -> Nx784
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, 28 * 28),
            nn.ReLU(),
        )

        self.discrimination_sequence = nn.Sequential(
            DisBlock(in_channels + 1, 64, 4, 2, 1),         # N x (1+1) x 28 x 28 -> N x 64 x 14 x 14
            DisBlock(64, 128, 4, 2, 1),                     # N x 64 x 14 x 14 -> N x 128 x 7 x 7
        )
        self.output_block = nn.Sequential(
            nn.Conv2d(128, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = self.label_embedding(y)                             # y: NxCls -> Nx784
        y = y.view(y.size(0), 1, x.size(2), x.size(3))          # y: Nx784 -> Nx1x28x28 
        x = self.discrimination_sequence(torch.cat([x, y], dim=1))
        x = self.output_block(x)
        return x