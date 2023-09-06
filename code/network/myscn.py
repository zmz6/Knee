from functools import partial

import torch.nn as nn
import torch.nn.functional as F

import torch

from network.myunet import NLUNet





def contracting_block(in_channels, out_channels, dropout=0.5):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


def combine(parallel_node, upsample_node):
    return parallel_node + upsample_node


def parallel_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


nonlinearity = partial(F.leaky_relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class LocalNet(nn.Module):
    def __init__(self, filter=128, num_class=28, dropout=0.5):
        super(LocalNet, self).__init__()
        ln = [filter] * 4
        self.head = nn.Sequential(nn.Conv2d(1, ln[0], kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(ln[0]),
                                  nn.LeakyReLU(negative_slope=0.1))
        self.dconv_down1 = contracting_block(ln[0], ln[0], dropout=dropout)
        self.dconv_down2 = contracting_block(ln[0], ln[1], dropout=dropout)
        self.dconv_down3 = contracting_block(ln[1], ln[2], dropout=dropout)
        self.dconv_down4 = contracting_block(ln[2], ln[3], dropout=dropout)
        self.parallel1 = parallel_block(ln[0], ln[0])
        self.parallel2 = parallel_block(ln[1], ln[1])
        self.parallel3 = parallel_block(ln[2], ln[2])
        # self.parallel4 = parallel_block(ln[3], ln[3])
        # self.parallel4 = DACblock(ln[3])

        self.parallel4 = _NonLocalBlockND(ln[3], dimension=2)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_last = nn.Conv2d(ln[0], num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Local
        input = self.head(x)
        conv1 = self.dconv_down1(input)
        p_1 = self.parallel1(conv1)
        x = self.pool(conv1)
        conv2 = self.dconv_down2(x)
        p_2 = self.parallel2(conv2)
        x = self.pool(conv2)
        conv3 = self.dconv_down3(x)
        p_3 = self.parallel3(conv3)
        x = self.pool(conv3)
        small = self.dconv_down4(x)
        p_4 = self.parallel4(small)

        x = self.upsample(p_4)
        x = combine(x, p_3)

        x = self.upsample(x)
        x = combine(x, p_2)

        x = self.upsample(x)
        x = combine(x, p_1)

        Lout = self.conv_last(x)
        return Lout


class SPNet(nn.Module):
    def __init__(self, dowasample_factor, num_landmarks, filters_bases, kernel_size):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=dowasample_factor)

        self.init_conv = nn.Sequential(
            nn.Conv2d(num_landmarks, filters_bases, kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(filters_bases),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(filters_bases, filters_bases, kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(filters_bases),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(filters_bases, filters_bases, kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(filters_bases),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(filters_bases, num_landmarks, kernel_size=11, stride=1, padding=int((kernel_size - 1) / 2)),
            nn.Tanh()
        )

        self.upsample = nn.Upsample(scale_factor=dowasample_factor, mode='bicubic', align_corners=True)

    def forward(self, x):
        x = self.downsample(x)
        conv = self.init_conv(x)
        out = self.conv1(conv)
        out = self.conv2(out)
        out = self.out_conv(out)
        out = self.upsample(out)
        return out


class NLSCN(nn.Module):
    def __init__(self, num_class=28, dropout=0.5):
        super().__init__()
        self.unet = LocalNet(filter=128, num_class=num_class,
                             dropout=dropout)
        # 20，64，11，4, 2
        self.sp_net = SPNet(
            dowasample_factor=16,
            num_landmarks=num_class,
            filters_bases=64,
            kernel_size=11
        )

    def forward(self, x):
        local_heatmaps = self.unet(x)
        spatial_heatmaps = self.sp_net(local_heatmaps)
        return spatial_heatmaps * local_heatmaps


class HeadSCN(nn.Module):
    def __init__(self, num_class=28, dropout=0.0, nl=False, sub=True):
        super().__init__()
        self.unet = NLUNet(num_class=num_class,in_chns=1, dropout=dropout, nl=nl, sub_sample=sub)
        # 20，64，11，4, 2
        self.sp_net = SPNet(
            dowasample_factor=8,
            num_landmarks=num_class,
            filters_bases=64,
            kernel_size=11
        )

    def forward(self, x):
        local_heatmaps = self.unet(x)
        spatial_heatmaps = self.sp_net(local_heatmaps)
        return spatial_heatmaps * local_heatmaps


if __name__ == '__main__':
    dump_input = torch.rand(
        (1, 1, 256, 256)
    )
    model = HeadSCN(nl=True, sub=False)
    out = model(dump_input)
    print(out.shape)
    # model = get_pose_net()
    # 计算模型的参数和计算复杂度
