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


class Deform_Conv2d(nn.Conv2d):
    """
    Reference
    - https://github.com/oeway/pytorch-deform-conv/blob/5270ac7dccbbfbf4dcec12db57080e5d6449c835/torch_deform_conv/layers.py#L10
    """
    def __init__(self, in_planes):
        # conv2d
        super(Deform_Conv2d, self).__init__(in_planes,
                                            in_planes * 2,
                                            kernel_size=3,
                                            padding=1,
                                            bias=False)
        self.in_planes = in_planes

    def forward(self, x):
        x_shape = x.size()
        
