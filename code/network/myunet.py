import torch
import torch.nn as nn

from network.common.no_local import _NonLocalBlockND


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


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


class NLUNet(nn.Module):
    def __init__(self, num_class, in_chns, dropout=0, nl=False, sub_sample=True):
        super(NLUNet, self).__init__()
        ln = [64, 128, 256, 512, 1024]
        self.dconv_down1 = double_conv(in_chns, ln[0])
        self.dconv_down2 = double_conv(ln[0], ln[1], dropout)
        self.dconv_down3 = double_conv(ln[1], ln[2], dropout)
        self.dconv_down4 = double_conv(ln[2], ln[3], dropout)
        self.dconv_down5 = double_conv(ln[3], ln[4], dropout)
        self.maxpool = nn.MaxPool2d(2)
        self.nl = nl
        if self.nl:
            self.nl_net = _NonLocalBlockND(ln[4], dimension=2, sub_sample=sub_sample)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up4 = double_conv(ln[3] + ln[4], ln[3], dropout)
        self.dconv_up3 = double_conv(ln[2] + ln[3], ln[2], dropout)
        self.dconv_up2 = double_conv(ln[1] + ln[2], ln[1], dropout)
        self.dconv_up1 = double_conv(ln[1] + ln[0], ln[0], dropout)

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
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)
        if self.nl:
            x = self.nl_net(x)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
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



class NLFourUNet(nn.Module):
    def __init__(self, num_class, in_chns, dropout=0):
        super(NLFourUNet, self).__init__()
        ln = [64, 128, 256, 512]
        self.dconv_down1 = double_conv(in_chns, ln[0], dropout)
        self.dconv_down2 = double_conv(ln[0], ln[1], dropout)
        self.dconv_down3 = double_conv(ln[1], ln[2], dropout)
        self.dconv_down4 = double_conv(ln[2], ln[3], dropout)
        self.maxpool = nn.MaxPool2d(2)
        self.nl_net = _NonLocalBlockND(ln[3], dimension=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(ln[2] + ln[3], ln[2], dropout)
        self.dconv_up2 = double_conv(ln[1] + ln[2], ln[1], dropout)
        self.dconv_up1 = double_conv(ln[1] + ln[0], ln[0], dropout)

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
        conv4 = self.nl_net(conv4)
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


if __name__ == '__main__':
    dump_input = torch.rand(
        (1, 1, 256, 256)
    )
    model = NLFourUNet(18, 1)
    out = model(dump_input)
    print(out.shape)
    # model = get_pose_net()
    # 计算模型的参数和计算复杂度