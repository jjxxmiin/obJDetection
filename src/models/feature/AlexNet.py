import torch.nn as nn


class LRN(nn.Module):
    """
    Local Response Normalization
    """

    def __init__(self, kernel_size, alpha, beta):
        super(LRN, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size,
                                     stride=1,
                                     padding=int((kernel_size) / 2))

        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        div = x.pow(2)
        div = self.avg_pool(div)
        div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(self, classes=1000):
        """
        GPU : 2
        """
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=96,
                               kernel_size=11,
                               stride=4,
                               padding=0,
                               bias=True)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.LRN1 = LRN(kernel_size=5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=256,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True,
                               groups=2)

        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.LRN2 = LRN(kernel_size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            384,
            384,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=2)

        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(
            384,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=2)

        self.relu5 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dense1 = nn.Linear(6 * 6 * 256, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.LRN1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.LRN2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool3(x)

        x = x.view(-1, 6 * 6 * 256)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
