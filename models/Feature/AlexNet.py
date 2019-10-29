import torch.nn as nn

class LRN(nn.Module):
    '''
    Local Response Normalization
    '''
    def __init__(self,kernel_size,alpha,beta):
        super(LRN,self).__init__()

        self.S = nn.AvgPool2d(kernel_size=kernel_size,
                              stride=1,
                              padding=int((kernel_size)/2))

        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        div = x.pow(2)
        div = self.S(div)
        div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x

class AlexNet(nn.Module):
    def __init__(self,classes=1000):
        '''
        GPU : 2
        '''
        super(AlexNet,self).__init__()

        self.C1 = nn.Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4,
                            padding=0,
                            bias=True)
        self.A1 = nn.ReLU()
        self.P1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.L1 = LRN(kernel_size=5,alpha=0.0001,beta=0.75)

        self.C2 = nn.Conv2d(in_channels=96,
                            out_channels=256,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            bias=True,
                            groups=2)

        self.A2 = nn.ReLU()
        self.P2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.L2 = LRN(kernel_size=5, alpha=0.0001, beta=0.75)

        self.C3 = nn.Conv2d(in_channels=256,
                            out_channels=384,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True)
        self.A3 = nn.ReLU()

        self.C4 = nn.Conv2d(384,384,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            groups=2)
        self.A4 = nn.ReLU()

        self.C5 = nn.Conv2d(384,256,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            groups=2)
        self.A5 = nn.ReLU()
        self.P3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.F1 = nn.Linear(6 * 6 * 256,4096)
        self.F2 = nn.Linear(4096,4096)
        self.F3 = nn.Linear(4096,classes)

    def forward(self, x):
        '''
        C : convolution
        A : relu
        P : overlapping pooling
        F : fully connected
        '''
        x = self.C1(x)
        x = self.A1(x)
        x = self.P1(x)
        x = self.L1(x)

        x = self.C2(x)
        x = self.A2(x)
        x = self.P2(x)
        x = self.L2(x)

        x = self.C3(x)
        x = self.A3(x)

        x = self.C4(x)
        x = self.A4(x)

        x = self.C5(x)
        x = self.A5(x)
        x = self.P3(x)

        x = x.view(-1, 6 * 6 * 256)

        x = self.F1(x)
        x = self.F2(x)
        x = self.F3(x)

        return x