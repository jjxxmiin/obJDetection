import torch
import torchvision
from datasets.coco import CocoDataset
from datasets.pascal_voc import VocDataset


class COCO(object):
    def __init__(self,
                 configs,
                 preprocessing):

        self.classes = 80
        self.configs = configs
        self.collate = preprocessing.collate
        self.custom_transform, self.torch_transform = preprocessing.augment()

    def get_loader(self):
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

        return custom_loader


class VOC(object):
    def __init__(self,
                 configs,
                 preprocessing):

        self.classes = 20
        self.configs = configs
        self.collate = preprocessing.collate
        self.custom_transform, self.torch_transform = preprocessing.augment()

    def get_loader(self, year = '2007'):
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

        return custom_loader


class STL10(object):
    """
    image shape : 96 x 96
    ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    unlabeled ++
    """
    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        stl10_dataset = torchvision.datasets.STL10(root='./datasets',
                                                   split=mode,
                                                   transform=transformer,
                                                   download=True)

        stl10_loader = torch.utils.data.DataLoader(stl10_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=shuffle)
        return stl10_loader


class CIFAR10(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        if mode == 'train':
            train = True
        else:
            train = False

        cifar10_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                       train=train,
                                                       transform=transformer,
                                                       download=True)

        cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=shuffle)

        return cifar10_loader
