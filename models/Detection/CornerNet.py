import torch.nn as nn
from utils.tester import model_test

class CornerNet(nn.Module):
    def __init__(self):
        super(CornerNet,self).__init__()

    def forward(self, x):

