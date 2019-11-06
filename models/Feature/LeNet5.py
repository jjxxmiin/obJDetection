import torch.nn as nn


# nn.Module을 상속받는다.
class LeNet5(nn.Module):
    def __init__(self, classes=10):
        # 다중 상속 중복 문제 해결
        super(LeNet5, self).__init__()
        # 1x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,  # default
                               padding=0,  # default
                               bias=True)
        self.sigmoid1 = nn.Sigmoid()
        # 28x28x6 -> 14x14x6
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2)
        # 14x14x6 -> 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               bias=True)
        self.sigmoid2 = nn.Sigmoid()
        # 10x10x16 -> 5x5x16
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2)
        # 5x5x16 -> 1x1x120
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=5,
                               bias=True)
        self.sigmoid3 = nn.Sigmoid()
        # 120 -> 84
        self.dense1 = nn.Linear(120, 84)
        self.sigmoid4 = nn.Sigmoid()
        # 84 -> 10
        self.output = nn.Linear(84, classes)

    def forward(self, x):
        '''
        C : convolution
        P : average pooling
        A : sigmoid
        F : fully connected
        '''

        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.avg_pool1(x)

        x = self.conv2(x)
        x = self.sigmoid2(x)
        x = self.avg_pool2(x)

        x = self.conv3(x)
        x = self.sigmoid3(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.sigmoid4(x)

        x = self.output(x)

        return x
