from pycocotools.coco import COCO
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

'''
# torchvision simple code

import torchvision.datasets as dset

coco_train = dset.CocoDetection(root='coco/train2017',
                                annFile='coco/annotations/instances_train2017.json')

print('Number of samples : ', len(coco_train))

img, target = coco_train[0]
print('coco image size : ', img.size)
'''

class CocoDataset(data.Dataset):
    def __init__(self,img_path,ann_path,transforms=None):
        '''
        download and read data
        '''
        self.img_path = img_path
        self.coco = COCO(ann_path)
        self.ids = list(self.coco.anns.keys())
        self.transform = transforms

    def __getitem__(self, index):
        '''
        return one item
        '''
        coco = self.coco

        # input image
        img_id = self.ids[index]
        img_file_name = coco.loadImgs(img_id)[0]['filename']
        img_path = os.path.join(self.img_path, img_file_name)
        img = Image.open(img_path).convert('RGB')

        # label
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        # transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        '''
        return data length
        '''
        return len(self.ids)


# main test

if __name__ == '__main__':
    dataDir = 'coco'
    dataType = 'train2017'

    img_path = '{}/{}'.format(dataDir, dataType)
    ann_path = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    transformer = transforms.Compose([transforms.ToTensor])

    custom_coco = CocoDataset(img_path,ann_path,transformer)
    custom_coco_loader = data.DataLoader(dataset=custom_coco,
                                    batch_size=32,
                                    shuffle=False)

    for i,c in enumerate(custom_coco_loader):
        break