import matplotlib.pyplot as plt
import skimage.io as io

import torch
from torchsummary import summary


class model_summary:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

    def summary(self,input):
        net = self.model.to(self.device)
        summary(net, input)

def save_tensor_image(image, file_name='test'):
    '''
    :param image: (tensor) cpu image
    :return: (file) save image
    '''

    io.imshow(image.permute(1, 2, 0).numpy())
    plt.savefig(file_name)
    print('Finish image save testing')
