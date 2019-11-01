import torch
import torchvision.transforms as transforms
import torch.optim as optim
from datasets.coco import CocoDataset
import utils.augment as augment
from models.Detection.CornerNet import CornerNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataDir = './datasets/coco'
dataType = 'train2017'

img_path = '{}/{}'.format(dataDir, dataType)
ann_path = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
label_path = './datasets/coco_labels.txt'

torch_transform = transforms.Compose([transforms.Resize((511,511)),
                                      transforms.ToTensor()])

custom_transform = augment.Compose([augment.Resize((511,511)),
                                    #augment.ToTensor()
                                    ])

custom_coco = CocoDataset(img_path, ann_path, label_path,
                          torch_transform=None,
                          custom_transform=custom_transform)
print(custom_coco)
custom_coco_loader = torch.utils.data.DataLoader(dataset=custom_coco,
                                                 batch_size=2,
                                                 shuffle=True)

net = CornerNet()
optimizer = optim.Adam(net.parameters(),
                       lr=0.0025)

batch_iterator = iter(custom_coco_loader)

print(batch_iterator)
images, targets = next(batch_iterator)
print(images)
print(targets)
print(images.shape)
