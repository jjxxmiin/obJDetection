'''
REFERENCE : https://github.com/feifeiwei/Pytorch-CornerNet
'''

import torch.nn as nn
from models.Feature.Hourglass import hourglassNet
from models.module.corner_pooling import *
from models.module.layer import Conv_bn, Conv_bn_relu


class CornerNet(nn.Module):
    def __init__(self, classes=20):
        super(CornerNet, self).__init__()
        self.backbone = hourglassNet()

        # top
        self.t_conv = Conv_bn_relu(256, 256, kernel_size=3, padding=1)
        self.l_conv = Conv_bn_relu(256, 256, kernel_size=3, padding=1)
        self.tl_conv = Conv_bn(256, 256, kernel_size=3, padding=1)

        self.conv_bn_1x1_tl = Conv_bn(256, 256, kernel_size=1)

        self.out_tl = nn.Sequential(
            nn.ReLU(),
            Conv_bn_relu(256, 256, kernel_size=3, padding=1)
        )

        self.h_tl = Conv_bn(256, 256, kernel_size=3, padding=1)
        self.e_tl = Conv_bn(256, 256, kernel_size=3, padding=1)
        self.o_tl = Conv_bn(256, 256, kernel_size=3, padding=1)

        self.out_h_tl = nn.Conv2d(256, classes, kernel_size=1)
        self.out_e_tl = nn.Conv2d(256, 1, kernel_size=1)
        self.out_o_tl = nn.Conv2d(256, 2, kernel_size=1)

        # bottom
        self.b_conv = Conv_bn_relu(256, 256, kernel_size=3, padding=1)
        self.r_conv = Conv_bn_relu(256, 256, kernel_size=3, padding=1)
        self.br_conv = Conv_bn(256, 256, kernel_size=3, padding=1)

        self.conv_bn_1x1_br = Conv_bn(256, 256, kernel_size=1)

        self.out_br = nn.Sequential(
            nn.ReLU(),
            Conv_bn_relu(256, 256, kernel_size=3, padding=1)
        )

        self.h_br = Conv_bn(256, 256, kernel_size=3, padding=1)
        self.e_br = Conv_bn(256, 256, kernel_size=3, padding=1)
        self.o_br = Conv_bn(256, 256, kernel_size=3, padding=1)

        self.out_h_br = nn.Conv2d(256, classes, kernel_size=1)
        self.out_e_br = nn.Conv2d(256, 1, kernel_size=1)
        self.out_o_br = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        t = self.t_conv(x)
        l = self.l_conv(x)
        top = top_pool()(t)
        left = left_pool()(l)
        top_left = self.tl_conv(top + left)

        conv_bn_tl = self.conv_bn_1x1_tl(x)

        out_tl = self.out_tl(top_left + conv_bn_tl)

        heat_tl = self.out_h_tl(self.h_tl(out_tl))
        embed_tl = self.out_e_tl(self.e_tl(out_tl))
        off_tl = self.out_o_tl(self.o_tl(out_tl))

        b = self.b_conv(x)
        r = self.r_conv(x)
        bottom = bottom_pool()(b)
        right = right_pool()(r)
        bottom_right = self.br_conv(bottom + right)

        conv_bn_br = self.conv_bn_1x1_br(x)

        out_br = self.out_br(bottom_right + conv_bn_br)

        heat_br = self.out_h_br(self.h_br(out_br))
        embed_br = self.out_e_br(self.e_br(out_br))
        off_br = self.out_o_br(self.o_br(out_br))

        return [heat_tl, heat_br, off_tl, off_br, embed_tl, embed_br]

'''
from utils.tester import model_summary

# test
tester = model_summary(CornerNet())
tester.summary((3, 511, 511))
'''