import torch.nn as nn
from utils.tester import model_test

class ZFNet(nn.Module):
    '''
    visualization Network
    '''
    def __init__(self,classes=1000):
        super(ZFNet,self).__init__()

        self.C1 = nn.Conv2d(3,96,
                            kernel_size=7,
                            stride=2,
                            padding=1)

        self.A1 = nn.ReLU()
        self.P1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.C2 = nn.Conv2d(96,256,
                            kernel_size=5,
                            stride=2,
                            padding=0)
        self.A2 = nn.ReLU()
        self.P2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.C3 = nn.Conv2d(256,384,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.A3 = nn.ReLU()

        self.C4 = nn.Conv2d(384,384,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.A4 = nn.ReLU()

        self.C5 = nn.Conv2d(384,256,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.A5 = nn.ReLU()
        self.P3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.F1 = nn.Linear(6*6*256,4096)
        self.Drop1 = nn.Dropout2d()
        self.F2 = nn.Linear(4096,4096)
        self.Drop2 = nn.Dropout2d()
        self.F3 = nn.Linear(4096,classes)


    def forward(self, x):
        x = self.C1(x)
        x = self.A1(x)
        x = self.P1(x)

        x = self.C2(x)
        x = self.A2(x)
        x = self.P2(x)

        x = self.A3(self.C3(x))
        x = self.A4(self.C4(x))

        x = self.P3(self.A5(self.C5(x)))

        x = x.view(x.size(0), -1)

        x = self.Drop1(self.F1(x))
        x = self.Drop2(self.F2(x))
        x = self.F3(x)

        return x

tester = model_test(ZFNet())
tester.summary((3,224,224))