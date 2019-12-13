import torch.nn as nn
from models.module.layers import Conv_bn_relu


# 12884
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()

        self.layer1 = Conv_bn_relu(in_planes, planes, kernel_size=1,
                                   stride=1, padding='SAME', bias=False)
        self.layer2 = Conv_bn_relu(planes, planes * 2, kernel_size=3,
                                   stride=1, padding='SAME', bias=False)

    def forward(self, x):
        shortcut = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += shortcut

        return x


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.layer1 = Conv_bn_relu(3, 32, kernel_size=3,
                                   stride=1, padding='SAME', bias=False, leaky=True)

        self.block1 = self.make_block(32, 64, 1)
        self.block2 = self.make_block(64, 128, 2)
        self.block3 = self.make_block(128, 256, 8)
        self.block4 = self.make_block(256, 512, 8)
        self.block5 = self.make_block(512, 1024, 4)

    @staticmethod
    def make_block(in_planes, planes, num_block):
        layers = []
        layers.append(Conv_bn_relu(in_planes, planes, kernel_size=3,
                                   stride=2, padding=1, bias=False, leaky=True))

        for _ in range(0, num_block):
            layers.append(BasicBlock(planes, in_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.block1(x)
        x = self.block2(x)
        route1 = self.block3(x)
        route2 = self.block4(route1)
        x = self.block5(route2)

        return route1, route2, x
