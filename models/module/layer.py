import torch.nn as nn


class Conv_bn_relu(nn.Module):
    def __init__(self, in_planes, planes,
                 kernel_size, stride=1, padding=0, bias=True):

        super(Conv_bn_relu, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes,
                              stride=stride, kernel_size=kernel_size,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv_bn(nn.Module):
    def __init__(self, in_planes, planes,
                 kernel_size, stride=1, padding=0, bias=True):

        super(Conv_bn, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv_dw(nn.Module):
    def __init__(self, in_planes, planes,
                 stride, kernel_size=3, padding=1, bias=False):

        super(Conv_dw, self).__init__()

        self.dw_conv = nn.Conv2d(in_planes, in_planes,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()

        self.pw_conv = nn.Conv2d(in_planes, planes,
                                 kernel_size=1, stride=1,
                                 padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pw_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x
