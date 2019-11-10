import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tools.preprocessing import CornerNet_Processing
from datasets.coco import CocoDataset
from datasets.pascal_voc import VocDataset
from models.Detection.CornerNet import CornerNet
from tools.tester import *

if torch.cuda.is_available():
    device = 'cuda'
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    #torch.set_default_tensor_type('torch.FloatTensor')


dataset = 'VOC'

if dataset == 'COCO':
    dataDir = './datasets/coco'
    dataType = 'train2017'

    img_path = '{}/{}'.format(dataDir, dataType)
    ann_path = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    label_path = './datasets/coco_labels.txt'

    preprocessing = CornerNet_Processing()

    custom_coco = CocoDataset(img_path, ann_path, label_path,
                              torch_transform=None,
                              custom_transform=preprocessing.augment())

    custom_loader = torch.utils.data.DataLoader(
        dataset=custom_coco,
        batch_size=1,
        shuffle=True,
        collate_fn=preprocessing.collate)

if dataset == 'VOC':
    year = '2007'

    img_path = './datasets/voc/VOC{}/JPEGImages'.format(year)
    ann_path = './datasets/voc/VOC{}/Annotations'.format(year)
    split_path = './datasets/voc/VOC{}/ImageSets/Main'.format(year)

    preprocessing = CornerNet_Processing()

    custom_voc = VocDataset(
        img_path,
        ann_path,
        torch_transform=None,
        custom_transform=preprocessing.augment())

    custom_loader = torch.utils.data.DataLoader(
        dataset=custom_voc,
        batch_size=1,
        shuffle=True,
        collate_fn=preprocessing.collate)

net = CornerNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0025)

#criterion = CornerNet_loss()

for i, (images, targets) in enumerate(custom_loader):
    images = images.to(device)
    targets = [target.to(device) for target in targets] # list : [heat_tl, heat_br, embed_tl, embed_br, off_tl, off_br]

    #for t in range(len(targets)):
    #    print(torch.sum(targets[t]))

    #outputs = net(images)
    #for o in range(len(outputs)):
    #    print(outputs[o].shape)

    #optimizer.zero_grad()
    break
