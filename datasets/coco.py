import os
import numpy as np
import torch.utils.data as data
from skimage import io
from pycocotools.coco import COCO

dataDir = 'coco'
dataType = 'train2017'

img_path = '{}/{}'.format(dataDir, dataType)
ann_path = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
label_path = 'coco_labels.txt'

class CocoDataset(data.Dataset):
    def __init__(self,
                 img_path,
                 ann_path,
                 label_path,
                 transform=None,
                 target_transform=None):
        '''
        :param
            transform: augmentation lib : [img], custom : [img, target]
            target_transform: augmentation [target]
        '''
        self.img_path = img_path
        self.coco = COCO(ann_path)
        self.ids = list(self.coco.imgToAnns.keys())
        self.label = self.parse_label(label_path)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        :param
            index : index

        :return
            img : (Image) img
            target : [xmin,ymin,xmax,ymax,class_id]
        '''
        coco = self.coco

        # input image
        img_id = self.ids[index]
        img_file_name = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_path, img_file_name)
        img = io.imread(img_path)

        # label
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = self.parse_coco(coco.loadAnns(ann_ids))

        # transform
        if self.transform is not None:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        return img, target

    def __len__(self):
        return len(self.ids)

    def parse_coco(self, ann):
        '''
        :param
            ann : coco annotation root
        :return
            [[xmin,ymin,xmax,ymax,c_id],[xmin,ymin,xmax,ymax,c_id],...]
        '''
        res = []

        for object in ann:
            if 'bbox' in object:
                c_id = object['category_id'] - 1
                box = object['bbox']
                box[2] += box[0]
                box[3] += box[1]

                res.append(box + [c_id])

        return np.array(res)

    def parse_label(self, label_path):
        coco_map = {}

        with open(label_path,'r') as f:
            for i, line in enumerate(f):
                coco_map[i] = line[:-1]

        return coco_map

'''
import utils.augment as augment

# test
def test():
    custom_coco = CocoDataset(img_path,ann_path,label_path,transform=augment.ToTensor())
    custom_coco_loader = data.DataLoader(dataset=custom_coco,
                                    batch_size=1,
                                    shuffle=False)

    for i ,c in enumerate(custom_coco_loader):
        print(i)
        print(c)
        break

test()
'''