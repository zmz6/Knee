import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FCPredictor(nn.Module):
    def __init__(self, in_channels, num_points):
        super(FCPredictor, self).__init__()
        print("Creating FCPredictor ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels
        self.fc1 = nn.Linear(8 * 8 * num_out_feat4, 256)
        self.fc2 = nn.Linear(256, num_points * 2)
        self.relu = nn.ReLU()

    def forward(self, x_dict):
        out = x_dict['out4']
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class AdapAvgPoolPredictor(nn.Module):
    def __init__(self, in_channels, num_points):
        super(AdapAvgPoolPredictor, self).__init__()
        print("Creating AdapAvgPoolPredictor ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels
        self.glob_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_out_feat4, num_out_feat4),
            nn.PReLU(num_parameters=num_out_feat4),
            nn.Linear(num_out_feat4, num_points * 2))

    def forward(self, x):
        x = x['out4']
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MultiFeaturePredictor(nn.Module):
    """
    Build multi-feature model from cfg
    """

    def __init__(self, in_channels, feat_size, num_points, **kwargs):
        super(MultiFeaturePredictor, self).__init__()
        print("Creating MultiFeaturePredictor ......")
        self.avg_glob_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Conv2d(in_channels[-1], 32, kernel_size=3, padding=1,
                                stride=2)  # before: feat_size, after: feat_size//2
        self.conv_2 = nn.Conv2d(32, 128, kernel_size=feat_size // 2)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_channels[-1] + 32 + 128, num_points * 2)

    def forward(self, x_dict):
        x = x_dict['out4']
        out1 = self.avg_glob_pool(x)
        out1 = out1.view(out1.size(0), -1)

        x = self.conv_1(x)
        x = self.relu(x)

        out2 = self.avg_glob_pool(x)
        out2 = out2.view(out2.size(0), -1)

        out3 = self.conv_2(x)
        out3 = out3.view(out3.size(0), -1)

        out = torch.cat([out1, out2, out3], 1)
        out = self.fc(out)

        return out


class MultiFeaturePredictorV2(nn.Module):
    def __init__(self, in_channels, feat_size, num_points, **kwargs):
        super(MultiFeaturePredictorV2, self).__init__()
        _, _, num_out_feat3, num_out_feat4 = in_channels
        self.avg_glob_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_out3 = nn.Sequential(
            nn.Conv2d(num_out_feat3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv_out4 = nn.Sequential(
            nn.Conv2d(num_out_feat4, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv_global_pool = nn.Conv2d(num_out_feat4, 128, kernel_size=feat_size)
        self.fc = nn.Linear(128 + 128 + 128, num_points * 2)

    def forward(self, x_dict):
        in3 = x_dict['out3']
        in4 = x_dict['out4']

        out3 = self.conv_out3(in3)
        out3 = self.avg_glob_pool(out3)
        out3 = out3.view(out3.size(0), -1)

        out4 = self.conv_out4(in4)
        out4 = self.avg_glob_pool(out4)
        out4 = out4.view(out4.size(0), -1)

        out5 = self.conv_global_pool(in4)
        out5 = out5.view(out5.size(0), -1)

        out = torch.cat([out3, out4, out5], 1)
        out = self.fc(out)

        return out
