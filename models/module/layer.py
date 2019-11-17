import torch.nn as nn

class Conv_bn_relu(nn.Module):
    def __init__(
            self,
            in_planes,
            planes,
            kernel_size,
            stride=1,
            padding=0,
            bias=True):
        super(Conv_bn_relu, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes,
                              stride=stride, kernel_size=kernel_size,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv_bn(nn.Module):
    def __init__(
            self,
            in_planes,
            planes,
            kernel_size,
            stride=1,
            padding=0,
            bias=True):
        super(Conv_bn, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes,
                              stride=stride, kernel_size=kernel_size,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
