import torch
from skimage.transform import resize
import cv2

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

        numpy : W x H x C -> tensor : C x H x W Float32
        """

        image = torch.from_numpy(image).permute((2, 1, 0))
        return image, boxes, labels


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, boxes=None, labels=None):
        """
        image: (numpy Image)
        boxes: (numpy)
        labels: (numpy)
        """

        cv2.imwrite('test.png',image)
        print(image.shape)

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
