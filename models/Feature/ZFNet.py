import torch.nn as nn


class ZFNet(nn.Module):
    '''
    visualization Network
    '''
    def __init__(self,classes=1000):
        super(ZFNet,self).__init__()

        self.conv1 = nn.Conv2d(3,96,
                            kernel_size=7,
                            stride=2,
                            padding=1)

        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 = nn.Conv2d(96,256,
                            kernel_size=5,
                            stride=2,
                            padding=0)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv3 = nn.Conv2d(256,384,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(384,384,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(384,256,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.dense1 = nn.Linear(6*6*256,4096)
        self.drop1 = nn.Dropout2d()
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout2d()
        self.dense3 = nn.Linear(4096,classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool3(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.dense3(x)

        return x
