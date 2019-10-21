import torch.utils.data as data
from PIL import Image
import os
import sys
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

year = '2007'

img_path = 'voc/VOC{}/JPEGImages'.format(year)
ann_path = 'voc/VOC{}/Annotations'.format(year)
split_path = 'voc/VOC{}/ImageSets/Main'.format(year)

class VocDataset(data.Dataset):
    def __init__(self,
                 img_path,
                 ann_path,
                 dataType='train',
                 transform=None,
                 target_transform=None):
        '''
        :param
            transform: augmentation lib : [img], custom : [img, target]
            target_transform: augmentation [target]
        '''
        self.img_path = img_path
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform

        type_file = dataType + '.txt'

        with open(os.path.join(split_path, type_file), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.imgs = [os.path.join(img_path, x + '.jpg') for x in file_names]
        self.anns = [os.path.join(ann_path, x + '.xml') for x in file_names]

        assert (len(self.imgs) == len(self.anns))

    def __getitem__(self, index):
        '''
        :param
            index : index

        :return
            img : (Image) img
            target : [xmin,ymin,xmax,ymax,class_id]
        '''
        img = Image.open(self.imgs[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(open(self.anns[index])).getroot())

        if self.transform is not None:
            img ,target = self.transform(img, target) # i need update transform

        return img, target

    def __len__(self):
        return len(self.imgs)

    def parse_voc_xml(self, xml):
        '''
        :param
            xml_path : xml root
        :return
            [[xmin,ymin,xmax,ymax,c_id],[xmin,ymin,xmax,ymax,c_id],...]
        '''
        objects = xml.findall("object")

        res = []

        for object in objects:
            c = object.find("name").text.lower().strip()
            c_id = VOC_CLASSES.index(c)

            bndbox = object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            res.append([xmin, ymin, xmax, ymax, c_id])

        return np.array(res)

# main test
'''
if __name__ == '__main__':
    custom_voc = VocDataset(img_path,ann_path,transform=transforms.ToTensor())

    custom_voc_loader = data.DataLoader(dataset=custom_voc,
                                    batch_size=32,
                                    shuffle=False)
'''