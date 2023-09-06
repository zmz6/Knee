from typing import Optional, Callable

import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3, conv1x1


def double_conv(in_channels, out_channels, drop_out=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(drop_out),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class ResUNet(nn.Module):
    def __init__(self, num_class, in_chns, dropout=0.0):
        super(ResUNet, self).__init__()
        ln = [64, 128, 256, 512]
        self.dconv_down1 = BasicBlock(in_chns, ln[0], dropout)
        self.dconv_down2 = BasicBlock(ln[0], ln[1], dropout)
        self.dconv_down3 = BasicBlock(ln[1], ln[2], dropout)
        self.dconv_down4 = BasicBlock(ln[2], ln[3], dropout)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = BasicBlock(ln[2] + ln[3], ln[2], dropout)
        self.dconv_up2 = BasicBlock(ln[1] + ln[2], ln[1], dropout)
        self.dconv_up1 = BasicBlock(ln[1] + ln[0], ln[0], dropout)

        self.conv_last = nn.Conv2d(ln[0], num_class, 1)

    def forward(self, x):
        # Local
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        Lout = self.conv_last(x)

        return Lout


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 dropout=0.5,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation=nn.LeakyReLU
                 ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act1 = activation(inplace=True)
        self.act2 = activation(inplace=True)

        # self.downsample = downsample

        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes * BasicBlock.expansion, stride),
            norm_layer(planes * BasicBlock.expansion),
        )
        self.dropout = nn.Dropout(dropout)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


if __name__ == '__main__':
    dump_input = torch.rand(
        (1, 1, 384, 384)
    )
    model = FourUNet(18, 1, dropout=0.5)
    out = model(dump_input)
    print(out.shape)
