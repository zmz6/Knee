import torch

from network.hourglass import HgNet
from network.hrnet import HRNet
from network.simple import SimpleNet

if __name__ == '__main__':
    input = torch.randn((1, 1, 256, 256))
    net = SimpleNet()
    print(net)
    outputs = net(input)
    if type(outputs) is list:
        print(outputs[-1].shape)
    else:
        print(outputs.shape)