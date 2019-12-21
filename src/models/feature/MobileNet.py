from src.models.module.layers import *


class MobileNetv1(nn.Module):
    def __init__(self, classes=1000):
        super(MobileNetv1, self).__init__()

        self.mobilenet_v1 = nn.Sequential(
            Conv_bn(in_planes=3, planes=32, stride=2, kernel_size=3),
            Conv_dw(in_planes=32, planes=64, stride=1),
            Conv_dw(in_planes=64, planes=128, stride=2),
            Conv_dw(in_planes=128, planes=128, stride=1),
            Conv_dw(in_planes=128, planes=256, stride=2),
            Conv_dw(in_planes=256, planes=256, stride=1),
            Conv_dw(in_planes=256, planes=512, stride=2),

            Conv_dw(in_planes=512, planes=512, stride=1),
            Conv_dw(in_planes=512, planes=512, stride=1),
            Conv_dw(in_planes=512, planes=512, stride=1),
            Conv_dw(in_planes=512, planes=512, stride=1),
            Conv_dw(in_planes=512, planes=512, stride=1),

            Conv_dw(in_planes=512, planes=1024, stride=2),
            Conv_dw(in_planes=1024, planes=1024, stride=1),
            nn.AvgPool2d(kernel_size=7),

        )

        self.linear = nn.Linear(1024, classes)

    def forward(self, x):
        x = self.mobilenet_v1(x)
        x = x.view(-1, 1024)
        x = self.linear(x)

        return x


'''
tester = model_tester(MobileNetv1())
tester.summary((3, 300, 300))
'''