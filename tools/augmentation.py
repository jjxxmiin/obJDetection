import torch
import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for trans in self.transforms:
            image, boxes, labels = trans(image, boxes, labels)

        return image, boxes, labels


class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        """
        image : (numpy)
        boxes : (numpy)
        labels : (numpy)

        numpy : H x W x C -> tensor : C x H x W Float32
        """

        image = image / 255.0
        image = torch.from_numpy(image).permute((2, 0, 1))

        return image, boxes, labels


class Resize(object):
    def __init__(self, output_size):
        """
        output_size: (H, W)
        """
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image, boxes=None, labels=None):
        """
        image: (H, W, C)
        boxes: (numpy)
        labels: (numpy)
        """
        h, w = image.shape[:2]

        new_w, new_h = self.output_size

        new_w, new_h = int(new_w), int(new_h)
        scale_w, scale_h = float(new_w) / w, float(new_h) / h

        image_trans = cv2.resize(image, (new_w, new_h,))
        boxes_trans = boxes * [scale_w, scale_h, scale_w, scale_h,]

        return image_trans, boxes_trans, labels


class HFlip(object):
    def __init__(self, p):
        """
        p: percent
        """
        self.p = p

    def __call__(self, image, boxes, labels):
        r = np.random.choice(2, 1, p=[self.p, 1-self.p])
        w = image.shape[1]

        if r == 0:
            image_trans = cv2.flip(image, 1)
            boxes_trans = []

            for box in boxes:
                box[0] = w - box[0]
                box[2] = w - box[2]
                boxes_trans.append(box)

            boxes_trans = np.array(boxes_trans)
        else:
            return image, boxes, labels

        return image_trans, boxes_trans, labels


class VFlip(object):
    def __init__(self, p):
        """
        p: percent
        """
        self.p = p

    def __call__(self, image, boxes, labels):
        r = np.random.choice(2, 1, p=[self.p, 1 - self.p])
        h = image.shape[0]

        if r == 0:
            image_trans = cv2.flip(image, 0)
            boxes_trans = []

            for box in boxes:
                box[1] = h - box[1]
                box[3] = h - box[3]
                boxes_trans.append(box)

            boxes_trans = np.array(boxes_trans)
        else:
            return image, boxes, labels

        return image_trans, boxes_trans, labels


class Crop(object):
    def __init__(self, p=0.5):
        """
        p: percent
        """
        self.p = p

    def __call__(self, image, boxes, labels):
        image_trans = image[]

        return image_trans, boxes, labels