import torch

class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        # image : H x W x C
        # tensor : C x H x W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image), boxes, labelse