import torch
import torch.nn as nn

from network.regession.predict import FCPredictor, AdapAvgPoolPredictor
from network.regession.resnet import ResNet18


class RegessionNet(nn.Module):
    def __init__(self, num_classes):
        super(RegessionNet, self).__init__()
        self.backbone = ResNet18(is_color=False)
        self.predictor = FCPredictor(self.backbone.num_out_feats, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.predictor(x)
        return x


class HeatmapRegessionNet(nn.Module):
    def __init__(self, num_classes):
        super(HeatmapRegessionNet, self).__init__()
        self.backbone = ResNet18(is_color=False)
        self.predictor = FCPredictor(self.backbone.num_out_feats, num_classes)

    def forward(self, x, heatmap):
        x = self.backbone(x)
        x = self.predictor(x)
        return x


class AvgRegessionNet(nn.Module):
    def __init__(self, num_classes):
        super(AvgRegessionNet, self).__init__()
        self.backbone = ResNet18(is_color=False)
        self.predictor = AdapAvgPoolPredictor(self.backbone.num_out_feats, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.predictor(x)
        return x


if __name__ == '__main__':
    input = torch.randn((1, 1, 256, 256))
    net = AvgRegessionNet(18)
    out = net(input)
    print(out.shape)
