'''
downsampling : stride = 2
channels : 256,384,384,384,512
'''

import torch.nn as nn
from utils.tester import model_test

class Residual(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(Residual, self).__init__()
        self.C1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(planes)
        self.A1 = nn.ReLU()

        self.C2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        # downsampling
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.A2 = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.A1(self.BN1(self.C1(x)))
        x = self.BN2(self.C2(x))
        x += self.shortcut(identity)
        x = self.A2(x)
        return x

class Hourglass104(nn.Module):
    def __init__(self):
        super(Hourglass104,self).__init__()

        # hourglass input
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3,128,kernel_size=7,stride=2),
            nn.BatchNorm2d(128),
            Residual(128, 256, stride=2),
        )

        self.pre_skip = nn.Conv2d(256,256,kernel_size=1)

        # hourglass downsampling
        self.skip0 = self.skip_connection(256)

        self.layer1 = self.sampling(256,256)
        self.skip1 = self.skip_connection(256)

        self.layer2 = self.sampling(256,384,stride=1)
        self.layer3 = self.sampling(384,384)

        self.skip2 = self.skip_connection(384)

        self.layer4 = self.sampling(384,384,stride=1)
        self.layer5 = self.sampling(384,384)

        self.skip3 = self.skip_connection(384)

        self.layer6 = self.sampling(384,384,stride=1)
        self.layer7 = self.sampling(384,384)

        self.skip4 = self.skip_connection(384)

        self.layer8 = self.sampling(384, 512, stride=1)

        # hourglass middle

        # hourglass upsampling

    def skip_connection(self, planes):
        return nn.Sequential(
            Residual(planes, planes),
            Residual(planes, planes)
        )

    def sampling(self, in_planes, planes, stride=2):
        return nn.Sequential(
            Residual(in_planes, planes, stride=stride),
            Residual(planes, planes)
        )

    def forward(self, x):
        x = self.pre_conv(x)
        pre_skip = self.pre_skip(x)
        skip0 = self.skip0
        x = self.layer1(x)
        skip1 = self.skip1
        x = self.layer2(x)
        x = self.layer3(x)
        skip2 = self.skip2
        x = self.layer4(x)
        x = self.layer5(x)
        skip3 = self.skip3
        x = self.layer6(x)
        x = self.layer7(x)
        skip4 = self.skip4
        x = self.layer8(x)

        return x


tester = model_test(Hourglass104())
tester.summary((3, 511, 511))

