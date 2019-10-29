'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''

import torch.nn as nn
from utils.tester import model_test

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.C1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(planes)
        self.A1 = nn.ReLU()

        self.C2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # downsample
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.C1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.BN1 = nn.BatchNorm2d(planes)
        self.A1 = nn.ReLU()

        self.C2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(planes)
        self.A2 = nn.ReLU()

        self.C3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.A3 = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.A1(self.BN1(self.C1(x)))
        x = self.A2(self.BN2(self.C2(x)))
        x = self.BN3(self.C3(x))
        x += self.shortcut(identity)
        x = self.A3(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, classes=1000):
        super(ResNet,self).__init__()
        self.in_planes = 64

        self.C1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.BN1 = nn.BatchNorm2d(64)
        self.A1 = nn.ReLU()
        self.P1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Block1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.Block2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.Block3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.Block4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.P2 = nn.AvgPool2d(kernel_size=4)
        self.F1 = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.C1(x)
        x = self.BN1(x)
        x = self.A1(x)
        x = self.P1(x)

        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)

        x = self.P2(x)
        x = x.view(x.size(0), -1)
        x = self.F1(x)

        return x

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

tester = model_test(ResNet18())
tester.summary((3,224,224))

