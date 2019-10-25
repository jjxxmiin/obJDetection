import torch.nn as nn

## nn.Module을 상속받는다.
class LeNet5(nn.Module):
    def __init__(self,classes=10):
        # 다중 상속 중복 문제 해결
        super(LeNet5, self).__init__()
        # 1x32x32 -> 6x28x28
        self.C1 = nn.Conv2d(in_channels=1,
                            out_channels=6,
                            kernel_size=5,
                            stride=1, # default
                            padding=0, # default
                            bias=True)
        self.A1 = nn.Sigmoid()
        # 28x28x6 -> 14x14x6
        self.P1 = nn.AvgPool2d(kernel_size=2)
        # 14x14x6 -> 10x10x16
        self.C2 = nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            bias=True)
        self.A2 = nn.Sigmoid()
        # 10x10x16 -> 5x5x16
        self.P2 = nn.AvgPool2d(kernel_size=2)
        # 5x5x16 -> 1x1x120
        self.C3 = nn.Conv2d(in_channels=16,
                            out_channels=120,
                            kernel_size=5,
                            bias=True)
        self.A3 = nn.Sigmoid()
        # 120 -> 84
        self.F1 = nn.Linear(120,84)
        self.A4 = nn.Sigmoid()
        # 84 -> 10
        self.output = nn.Linear(84,classes)

    def forward(self, x):
        '''
        C : convolution
        P : average pooling
        A : sigmoid
        F : fully connected
        '''

        x = self.C1(x)
        x = self.A1(x)
        x = self.P1(x)

        x = self.C2(x)
        x = self.A2(x)
        x = self.P2(x)

        x = self.C3(x)
        x = self.A3(x)

        x = x.view(x.size(0), -1)

        x = self.F1(x)
        x = self.A4(x)

        x = self.output(x)

        return x
