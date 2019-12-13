import torch
import tools.augmentation as augment
import torchvision.transforms as transformer
from tools.utils import *


def custom_collate(batch):
    '''
    :batch:
    :return:
    images : (tensor)
    targets : (list) [(tensor), (tensor)]
    '''
    targets = []
    images = []

    for x in batch:
        targets.append(torch.from_numpy(x[1]))
        images.append(x[0])

    return torch.stack(images, 0), targets


class Yolo_Processing(object):
    def __init__(self):
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    @staticmethod
    def augment():
        custom_transform = augment.Compose([augment.Resize((416, 416)),
                                            augment.ToTensor()])

        torch_transform = transformer.Compose([transformer.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

        return custom_transform, torch_transform

    def collate(self, batch):
        targets = []
        images = []

        for x in batch:
            targets.append(torch.from_numpy(x[1]))
            images.append(x[0])

        return images, targets


class CornerNet_Processing(object):
    def __init__(self):
        self.image_size = 512
        self.output_size = 128
        self.max_selection = 100
        self.num_class = 20
        self.output_stride = 4

    @staticmethod
    def augment():
        custom_transform = augment.Compose([augment.Resize((511, 511)),
                                     augment.ToTensor()])
        torch_transform = None
        return custom_transform, torch_transform

    def collate(self, batch):
        '''
        :batch:
        image : (tensor) image
        target : (list) [xmin,ymin,xmax,ymax,c_id]

        :return:
        images : (tensor)
        targets : (list)
        '''

        targets = []
        images = []

        for x in batch:
            targets.append(torch.from_numpy(x[1]))
            images.append(x[0])

        batch_size = len(images)
        w = images[0].size(1)
        h = images[0].size(2)

        inputs = torch.zeros(batch_size, 3, h, w)

        new_w = int(np.ceil(w / self.output_stride))
        new_h = int(np.ceil(h / self.output_stride))

        tl_heatmap = torch.zeros((batch_size, self.num_class, new_h, new_w))
        br_heatmap = torch.zeros((batch_size, self.num_class, new_h, new_w))

        tl_embedding = torch.zeros((batch_size, self.max_selection,))
        br_embedding = torch.zeros((batch_size, self.max_selection,))

        tl_offset = torch.zeros((batch_size, self.max_selection, 2))
        br_offset = torch.zeros((batch_size, self.max_selection, 2))

        embedding_mask = torch.zeros((batch_size, self.max_selection,))
        embedding_lens = torch.zeros((batch_size,))

        for b in range(batch_size):
            inputs[b] = images[b]
            for xmin, ymin, xmax, ymax, c_id in targets[b]:
                label = c_id

                tl_x = xmin / self.output_stride
                tl_y = ymin / self.output_stride
                br_x = xmax / self.output_stride
                br_y = ymax / self.output_stride

                shift_tl_x = int(tl_x)
                shift_tl_y = int(tl_y)
                shift_br_x = int(br_x)
                shift_br_y = int(br_y)

                shift_width = int((xmax - xmin) / self.output_stride)
                shift_height = int((ymax - ymin) / self.output_stride)

                radius = gaussian_radius(shift_width, shift_height)
                radius = max(0, int(radius))

                draw_gaussian(tl_heatmap[b, int(label)],
                              (shift_tl_x, shift_tl_y), radius)
                draw_gaussian(br_heatmap[b, int(label)],
                              (shift_br_x, shift_br_y), radius)

                embedding_id = embedding_lens[b].long().item()
                # offset, embedding 의 위치를 설정해서 넣어준다.
                tl_offset[b, embedding_id, :] = torch.Tensor([tl_x - shift_tl_x, tl_y - shift_tl_y])
                br_offset[b, embedding_id, :] = torch.Tensor([br_x - shift_br_x, br_y - shift_br_y])
                # 임베딩 값 why??
                tl_embedding[b, embedding_id] = shift_tl_y * self.output_size + shift_tl_x
                br_embedding[b, embedding_id] = shift_br_y * self.output_size + shift_br_x
                # box의 개수
                embedding_lens[b] += 1

            for b in range(batch_size):
                embedding_len = embedding_lens[b].long().item()
                embedding_mask[b, :embedding_len] = 1

        return torch.stack(images, 0), [tl_heatmap,
                                        br_heatmap,
                                        tl_offset,
                                        br_offset,
                                        tl_embedding,
                                        br_embedding,
                                        embedding_mask]
