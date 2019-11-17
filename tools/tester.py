import numpy as np
import cv2
import torch
from torchsummary import summary


class model_summary:
    def __init__(self, model):
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    def summary(self, input):
        net = self.model.to(self.device)
        summary(net, input)


def save_tensor_image(image, targets):
    '''
    :param image: (tensor) cpu image
    :return: (file) save image
    '''

    image = image.permute(2, 1, 0).numpy() * 255
    image = image.astype('uint8')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for target in targets:
        target = np.floor(target)
        image = cv2.rectangle(image,(target[0], target[1]),(target[2], target[3]), (255,0,0), 3)

    cv2.imwrite('test.png', image)

    print('Finish image save testing')
