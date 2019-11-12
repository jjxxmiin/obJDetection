import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tools.preprocessing import CornerNet_Processing
from datasets.coco import CocoDataset
from datasets.pascal_voc import VocDataset
from models.Detection.CornerNet import CornerNet
from tools.tester import *
from models.module.loss import CornerNet_Loss

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

    classes = 80

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
    classes = 20

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

net = CornerNet(classes=classes).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0025)

criterion = CornerNet_Loss()
epoch = 100

'''
# 모델 불러오기
model = torch.load(PATH)
# checkpoint 불러오기
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
'''

for e in range(epoch):
    total_loss = 0
    for i, (images, targets) in enumerate(custom_loader):
        images = images.to(device)
        # targets : [heat_tl, heat_br, embed_tl, embed_br, off_tl, off_br]
        targets = [target.to(device) for target in targets]

        optimizer.zero_grad()
        outputs = net(images)
        loss, hist = criterion(targets, outputs)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print('[Epoch %d, Batch %d/%d] [totalLoss:%.6f] [ht_loss:%.6f, off_loss:%.6f, pull_loss:%.6f, push_loss:%.6f]'
              % (e, i, len(custom_loader), loss.item(), hist[0], hist[1], hist[2], hist[3]))

        if e == 0 and i == 0:
            best_loss = total_loss

    if total_loss < best_loss:
        # checkpoint 저장
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss' : total_loss,
        }, './log/hour104_cornernet.ckpt')

        best_loss = total_loss