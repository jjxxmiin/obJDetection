import torch
import torchvision.transforms as transforms
import torch.optim as optim
from datasets.coco import CocoDataset
from datasets.pascal_voc import VocDataset
import utils.augment as augment
from models.Detection.CornerNet import CornerNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'VOC'

custom_transform = augment.Compose([augment.Resize((511,511)),
                                    augment.ToTensor()])

if dataset == 'COCO':
    dataDir = './datasets/coco'
    dataType = 'train2017'

    img_path = '{}/{}'.format(dataDir, dataType)
    ann_path = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    label_path = './datasets/coco_labels.txt'

    custom_coco = CocoDataset(img_path, ann_path, label_path,
                              torch_transform=None,
                              custom_transform=custom_transform)

    custom_loader = torch.utils.data.DataLoader(dataset=custom_coco,
                                                     batch_size=2,
                                                     shuffle=True,
                                                     collate_fn=augment.custom_collate)

if dataset == 'VOC':
    year = '2007'

    img_path = './datasets/voc/VOC{}/JPEGImages'.format(year)
    ann_path = './datasets/voc/VOC{}/Annotations'.format(year)
    split_path = './datasets/voc/VOC{}/ImageSets/Main'.format(year)

    custom_voc = VocDataset(img_path,ann_path,torch_transform=None,custom_transform=custom_transform)

    custom_loader = torch.utils.data.DataLoader(dataset=custom_voc,
                                                     batch_size=2,
                                                     shuffle=True,
                                                     collate_fn=augment.custom_collate)


net = CornerNet()
optimizer = optim.Adam(net.parameters(), lr=0.0025)

batch_iterator = iter(custom_loader)

print(batch_iterator)
images, targets = next(batch_iterator)
print(images.shape)
print(targets)