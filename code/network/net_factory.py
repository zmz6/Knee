from torch.utils.checkpoint import checkpoint

from network.ResUNet import ResUNet
import torch
import torch.nn as nn


from network.hourglass import HgNet
from network.hrnet import HRNet
from network.myscn import NLSCN, HeadSCN
from network.myunet import NLUNet, NLFourUNet
from network.regession.net import AvgRegessionNet, RegessionNet
from network.regession.predict import AdapAvgPoolPredictor
from network.scn import SCN

from network.simple import SimpleNet
from network.unet import UNet, FourUNet


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def net_factory(net_type="unet", in_chns=1, class_num=19, dropout=0):
    if net_type == "scn":
        net = SCN(class_num, dropout=0.5).cuda()
    elif net_type == "unet":
        net = UNet(class_num, in_chns, dropout).cuda()
    elif net_type == "four_unet":
        net = FourUNet(class_num, in_chns, dropout).cuda()
    # elif net_type == "ResUNet":
    #     net = ResUNet(class_num, in_chns, dropout).cuda()
    elif net_type == "hg_4":
        net = HgNet().cuda()
    elif net_type == "simple_34":
        net = SimpleNet().cuda()
    elif net_type == "hrnet_18":
        net = HRNet().cuda()
    elif net_type == "nl_scn":
        net = NLSCN(class_num, dropout=0.5).cuda()
    elif net_type == "nl_unet":
        net = NLUNet(class_num, in_chns, dropout, nl=True).cuda()
    elif net_type == "nl_four_unet":
        net = NLFourUNet(class_num, in_chns, dropout).cuda()
    elif net_type == "headscn":
        net = HeadSCN(class_num, dropout, nl=False).cuda()
    elif net_type == "nl_headscn":
        net = HeadSCN(class_num, dropout, nl=True).cuda()
    elif net_type == "nl_sub_headscn":
        net = HeadSCN(class_num, dropout, nl=True, sub=False).cuda()
    else:
        net = None
    return net


def RegNet_factory(net_type="fc", num_classes=18):
    if net_type == "avg":
        net = AvgRegessionNet(num_classes).cuda()
    elif net_type == "fc":
        net = RegessionNet(num_classes).cuda()
    else:
        net = None
    return net


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(
            n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Linear(ndf * 32, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, map, feature):
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    pass
