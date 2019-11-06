'''
downsampling : stride = 2
channels : 256,384,384,384,512
'''

import torch.nn as nn
from models.module.layer import conv_bn, conv_bn_relu


class Residual(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Residual, self).__init__()
        self.layer1 = conv_bn_relu(in_planes, planes,
                                   kernel_size=3, stride=stride,
                                   padding=1, bias=False)

        self.layer2 = conv_bn(planes, planes,
                              kernel_size=3,
                              padding=1, bias=False)

        self.shortcut = nn.Sequential()

        # down sampling
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_bn(
                    in_planes,
                    self.expansion *
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False))
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.layer1(x)
        x = self.layer2(x)
        x += self.shortcut(identity)
        x = self.relu(x)
        return x


def sampling(in_planes, planes, stride=2):
    return nn.Sequential(
        Residual(in_planes, planes, stride=stride),
        Residual(planes, planes)
    )


def skip_connection(planes):
    return nn.Sequential(
        Residual(planes, planes),
        Residual(planes, planes)
    )


def up_sampling(planes):
    return nn.Sequential(
        Residual(planes, planes),
        nn.UpsamplingNearest2d(scale_factor=2)
    )


class hourglassModule(nn.Module):
    def __init__(self):
        super(hourglassModule, self).__init__()

        # hourglass down sampling
        self.skip0 = skip_connection(256)
        self.down1 = sampling(256, 256)
        self.skip1 = skip_connection(256)
        self.down2 = sampling(256, 384)
        self.skip2 = skip_connection(384)
        self.down3 = sampling(384, 384)
        self.skip3 = skip_connection(384)
        self.down4 = sampling(384, 384)
        self.skip4 = skip_connection(384)
        self.down5 = sampling(384, 512)
        # hourglass bottleneck
        self.neck = nn.Sequential(
            sampling(512, 512, stride=1),
            sampling(512, 512, stride=1),
            Residual(512, 512, stride=1)
        )
        # hourglass upsampling
        '''
        Nearest Neighbor Upsampling
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]]]])
        >>> m = nn.UpsamplingNearest2d(scale_factor=2)
        >>> m(input)
        tensor([[[[ 1.,  1.,  2.,  2.],
                  [ 1.,  1.,  2.,  2.],
                  [ 3.,  3.,  4.,  4.],
                  [ 3.,  3.,  4.,  4.]]]])
        '''
        self.up1 = up_sampling(512)
        self.convert1 = Residual(512, 384)

        self.up2 = up_sampling(384)
        self.convert2 = Residual(384, 384)

        self.up3 = up_sampling(384)
        self.convert3 = Residual(384, 384)

        self.up4 = up_sampling(384)
        self.convert4 = Residual(384, 256)

        self.up5 = up_sampling(256)
        self.convert5 = Residual(256, 256)

    def forward(self, x):
        skip0 = self.skip0(x)
        x = self.down1(x)
        skip1 = self.skip1(x)
        x = self.down2(x)
        skip2 = self.skip2(x)
        x = self.down3(x)
        skip3 = self.skip3(x)
        x = self.down4(x)
        skip4 = self.skip4(x)
        x = self.down5(x)

        x = self.neck(x)

        # element-wise addition product
        x = self.up1(x)
        x = self.convert1(x) + skip4
        x = self.up2(x)
        x = self.convert2(x) + skip3
        x = self.up3(x)
        x = self.convert3(x) + skip2
        x = self.up4(x)
        x = self.convert4(x) + skip1
        x = self.up5(x)
        x = self.convert5(x) + skip0

        return x


class hourglassNet(nn.Module):
    def __init__(self):
        super(hourglassNet, self).__init__()

        # hourglass input
        self.pre_conv = nn.Sequential(
            conv_bn_relu(3, 128, kernel_size=7, stride=2, padding=2),
            Residual(128, 256, stride=2),
        )

        self.pre_skip = nn.Sequential(
            conv_bn(256, 256, kernel_size=1)
        )

        self.hourglass1 = hourglassModule()

        self.middle = nn.Sequential(
            conv_bn_relu(256, 256, kernel_size=3, padding=1),
            conv_bn(256, 256, kernel_size=1)
        )

        self.relu = nn.ReLU()
        self.residual = Residual(256, 256)

        self.hourglass2 = hourglassModule()

        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pre_conv(x)
        skip = self.pre_skip(x)

        x = self.hourglass1(x)

        x = self.middle(x) + skip
        x = self.relu(x)
        x = self.residual(x)

        x = self.hourglass2(x)
        x = self.conv(x)

        return x
