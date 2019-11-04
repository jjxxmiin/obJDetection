import numpy as np
from skimage.transform import resize

import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for trans in self.transforms:
            image, boxes, labels = trans(image, boxes, labels)

        return image, boxes, labels

class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        '''
        image : (numpy)
        boxes : (numpy)
        labels : (numpy)

        numpy : H x W x C -> tensor : C x H x W Float32
        '''
        image = torch.from_numpy(image).permute((2, 0, 1))

        return image, boxes, labels

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, boxes=None, labels=None):
        '''
        :param image: (numpy Image)
        :param boxes: (numpy)
        :param labels: (numpy)
        :return:
        '''

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * h / w
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        scale_w, scale_h = float(new_w) / w, float(new_h) / h

        image_trans = resize(image, (new_h, new_w))
        boxes_trans = boxes * [scale_w, scale_h, scale_w, scale_h]

        return image_trans, boxes_trans, labels

def custom_collate(batch):
    '''
    :param batch: init batch
    :return:
    images : (tensor)
    targets : (list) [(tensor), (tensor)]
    '''
    targets = []
    images = []

    for sample in batch:
        images.append(sample[0])
        targets.append(torch.from_numpy(sample[1]))

    return torch.stack(images, 0), targets
