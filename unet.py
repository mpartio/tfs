import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, dim=64):
        super(UNet, self).__init__()

        # encoder
        self.conv1 = self.conv_block(1, dim)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = self.conv_block(dim, dim * 2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = self.conv_block(dim * 2, dim * 4)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = self.conv_block(dim * 4, dim * 8)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # processor
        self.bridge = self.conv_block(dim * 8, dim * 16)

        # decoder
        self.upconv4 = nn.ConvTranspose2d(
            dim * 16, dim * 8, kernel_size=2, stride=2, padding=0
        )
        self.conv5 = self.conv_block(dim * 16, dim * 8)

        self.upconv3 = nn.ConvTranspose2d(
            dim * 8, dim * 4, kernel_size=2, stride=2, padding=0
        )
        self.conv6 = self.conv_block(dim * 8, dim * 4)

        self.upconv2 = nn.ConvTranspose2d(
            dim * 4, dim * 2, kernel_size=2, stride=2, padding=0
        )
        self.conv7 = self.conv_block(dim * 4, dim * 2)

        self.upconv1 = nn.ConvTranspose2d(
            dim * 2, dim, kernel_size=2, stride=2, padding=0
        )
        self.conv8 = self.conv_block(dim * 2, dim)

        self.prediction_head = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x = self.mp1(x1)
        x2 = self.conv2(x)
        x = self.mp2(x2)
        x3 = self.conv3(x)
        x = self.mp3(x3)
        x4 = self.conv4(x)
        x = self.mp4(x4)

        # processor
        x = self.bridge(x)

        # decoder
        x = self.upconv4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.conv5(x)
        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv7(x)
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv8(x)

        x = self.prediction_head(x)

        return x
