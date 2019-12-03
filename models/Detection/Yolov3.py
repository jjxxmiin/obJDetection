import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.layers import *
from models.Feature.Darknet import Darknet53

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)
_NUM_CLASS = 80


class Yolo_Block(nn.Module):
    def __init__(self, in_planes, planes):
        super(Yolo_Block, self).__init__()

        self.layer1 = Conv_bn_relu(in_planes, planes, kernel_size=1,
                                   padding='SAME', bias=False, leaky=True)
        self.layer2 = Conv_bn_relu(planes, planes*2, kernel_size=3,
                                   padding='SAME', bias=False, leaky=True)

        self.layer3 = Conv_bn_relu(planes*2, planes, kernel_size=1,
                                   padding='SAME', bias=False, leaky=True)
        self.layer4 = Conv_bn_relu(planes, planes*2, kernel_size=3,
                                   padding='SAME', bias=False, leaky=True)

        self.layer5 = Conv_bn_relu(planes*2, planes, kernel_size=1,
                                   padding='SAME', bias=False, leaky=True)
        self.layer6 = Conv_bn_relu(planes, planes*2, kernel_size=3,
                                   padding='SAME', bias=False, leaky=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        route = self.layer5(x)
        x = self.layer6(route)

        return route, x


class Yolo_Layer(nn.Module):
    def __init__(self, num_branch):
        super(Yolo_Layer, self).__init__()

        self.input_shape = _MODEL_SIZE
        self.anchors = _ANCHORS[num_branch*3:num_branch*3+3]
        self.classes = _NUM_CLASS
        self.n_anchors = len(self.anchors)
        self.planes = self.n_anchors * (5 + self.classes)

    def forward(self, x, targets=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        in_planes = x.size(1)
        grid_shape = x.size(2), x.size(3)

        x = nn.Conv2d(in_planes, self.planes, kernel_size=1, stride=1, bias=True)(x)
        x = x.view(-1, self.n_anchors * grid_shape[0] * grid_shape[1], 5 + self.classes)

        stride = FloatTensor([self.input_shape[0]//grid_shape[0], self.input_shape[1]//grid_shape[1]])

        box_centers, box_shapes, confidence, classes = x.split([2, 2, 1, 80], dim=-1)

        x = torch.arange(start=0, end=grid_shape[0])
        y = torch.arange(start=0, end=grid_shape[1])

        x_offsets, y_offsets = torch.meshgrid(x, y)
        x_offsets = x_offsets.contiguous().view(-1, 1)
        y_offsets = y_offsets.contiguous().view(-1, 1)

        xy_offsets = torch.cat((x_offsets, y_offsets), -1)
        xy_offsets = xy_offsets.repeat(1, self.n_anchors)
        xy_offsets = xy_offsets.view(1, -1, 2).float()

        box_centers = torch.sigmoid(box_centers)
        box_centers = (box_centers + xy_offsets) * stride

        anchors = FloatTensor(self.anchors)
        anchors = anchors.repeat(grid_shape[0] * grid_shape[1], 1)

        box_shapes = torch.exp(box_shapes) * anchors

        confidence = torch.sigmoid(confidence)

        classes = torch.sigmoid(classes)

        output = torch.cat((box_centers, box_shapes, confidence, classes), dim=-1)

        if targets is not None:
            return output
        else:
            return output


class Yolov3(nn.Module):
    def __init__(self, mode='train'):
        super(Yolov3, self).__init__()
        self.mode = mode

        self.backbone = Darknet53()

        self.block1 = Yolo_Block(1024, 512)
        self.branch1 = Yolo_Layer(0)

        self.layer1 = Conv_bn_relu(512, 256, kernel_size=1,
                                   padding='SAME', bias=False, leaky=True)
        self.upsample1 = Upsample(2)

        self.block2 = Yolo_Block(768, 256)
        self.branch2 = Yolo_Layer(1)

        self.layer2 = Conv_bn_relu(256, 128, kernel_size=1,
                                   padding='SAME', bias=False, leaky=True)
        self.upsample2 = Upsample(2)

        self.block3 = Yolo_Block(384, 128)
        self.branch3 = Yolo_Layer(2)

    def forward(self, x, target=None):
        route1, route2, x = self.backbone(x)
        route, x = self.block1(x)
        detection1 = self.branch1(x, target)

        x = self.layer1(route)
        x = self.upsample1(x)
        x = torch.cat((x, route2), dim=1)

        route, x = self.block2(x)
        detection2 = self.branch2(x, target)

        x = self.layer2(route)
        x = self.upsample2(x)
        x = torch.cat((x, route1), dim=1)

        route, x = self.block3(x)
        detection3 = self.branch3(x, target)

        x = (detection1, detection2, detection3)

        if self.mode == 'train':
            loss = sum(x)
            return loss
        else:
            output = torch.cat(x, 1)
            return output