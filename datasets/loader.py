import torch
from datasets.coco import CocoDataset
from datasets.pascal_voc import VocDataset


class Loader(object):
    def __init__(self,
                 configs,
                 preprocessing):

        self.configs = configs
        self.collate = preprocessing.collate
        self.custom_transform, self.torch_transform = preprocessing.augment()

    def get_loader(self):
        custom_loader = None

        if self.configs['dataset'] == 'COCO':
            # classes 80
            dataDir = './datasets/coco'
            dataType = 'train2017'

            img_path = '{}/{}'.format(dataDir, dataType)
            ann_path = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
            label_path = './datasets/coco_labels.txt'

            custom_coco = CocoDataset(img_path, ann_path, label_path,
                                      torch_transform=self.torch_transform,
                                      custom_transform=self.custom_transform)

            custom_loader = torch.utils.data.DataLoader(
                dataset=custom_coco,
                batch_size=self.configs['batch_size'],
                shuffle=True,
                collate_fn=self.collate)

        elif self.configs['dataset'] == 'VOC':
            # classes 20
            year = '2007'

            img_path = './datasets/voc/VOC{}/JPEGImages'.format(year)
            ann_path = './datasets/voc/VOC{}/Annotations'.format(year)
            split_path = './datasets/voc/VOC{}/ImageSets/Main'.format(year)

            custom_voc = VocDataset(
                img_path,
                ann_path,
                torch_transform=self.torch_transform,
                custom_transform=self.custom_transform)

            custom_loader = torch.utils.data.DataLoader(
                dataset=custom_voc,
                batch_size=self.configs['batch_size'],
                shuffle=True,
                collate_fn=self.collate)

        else:
            assert custom_loader is None

        return custom_loader