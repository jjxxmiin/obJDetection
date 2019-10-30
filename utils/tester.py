import torch
from torchsummary import summary

class model_test:
    def __init__(self,model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

    def summary(self,input):
        net = self.model.to(self.device)
        summary(net, input)