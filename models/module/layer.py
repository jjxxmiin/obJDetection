import torch.nn as nn


class conv_bn_relu(nn.Module):
    def __init__(self,in_planes,planes,kernel_size,stride=1,padding=0,bias=True):
        super(conv_bn_relu,self).__init__()

        self.C = nn.Conv2d(in_planes,planes,
                           stride=stride,kernel_size=kernel_size,
                           padding=padding,bias=bias)
        self.BN = nn.BatchNorm2d(planes)
        self.A = nn.ReLU()

    def forward(self, x):
        return self.A(self.BN(self.C(x)))


class conv_bn(nn.Module):
    def __init__(self,in_planes,planes,kernel_size,stride=1,padding=0,bias=True):
        super(conv_bn,self).__init__()

        self.C = nn.Conv2d(in_planes, planes,
                           stride=stride,kernel_size=kernel_size,
                           padding=padding,bias=bias)
        self.BN = nn.BatchNorm2d(planes)
        self.A = nn.ReLU()

    def forward(self, x):
        return self.A(self.BN(self.C(x)))
