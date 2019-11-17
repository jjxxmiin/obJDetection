import torch.optim as optim
from tools.preprocessing import CornerNet_Processing
from datasets.coco import CocoDataset
from datasets.pascal_voc import VocDataset
from models.Detection.CornerNet import CornerNet
from tools.tester import *
from models.module.loss import *

if torch.cuda.is_available():
    device = 'cuda'
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    #torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'model': 'cornernet',
    'backbone': 'hourglass',
    'dataset': 'VOC',
    'lr': 0.0025,
    'classes': 20,
    'mode': 'train',
    'epoch': 100,
    'batch_size': 1,
}

if configs['dataset'] == 'COCO':
    # classes 80
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
        batch_size=configs['batch_size'],
        shuffle=True,
        collate_fn=preprocessing.collate)

if configs['dataset'] == 'VOC':
    # classes 20
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
        batch_size=configs['batch_size'],
        shuffle=True,)
        #collate_fn=preprocessing.collate)

for i, (images, targets) in enumerate(custom_loader):
    save_tensor_image(images[0], targets[0])
    break

'''
net = CornerNet(classes=configs['classes']).to(device)
optimizer = optim.Adam(net.parameters(), lr=configs['lr'])
criterion = CornerNet_Loss()

if configs['mode'] == 'train':
    for e in range(configs['epoch']):
        total_loss = 0
        for i, (images, targets) in enumerate(custom_loader):
            images = images.to(device)
            # targets : [heat_tl, heat_br, embed_tl, embed_br, off_tl, off_br]
            targets = [target.to(device) for target in targets]

            optimizer.zero_grad()
            outputs = net(images)
            loss, hist = criterion(outputs,targets)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print('[Epoch %d, Batch %d/%d] [totalLoss:%.6f] [ht_loss:%.6f, off_loss:%.6f, pull_loss:%.6f, push_loss:%.6f]'
                  % (e, i, len(custom_loader), loss.item(), hist[0], hist[1], hist[2], hist[3]))

        print('saveing model')
        torch.save({
            'epoch': e,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, './log/hour104_cornernet.ckpt')

if configs['mode'] == 'test':
    checkpoint = torch.load('./log/hour104_cornernet_1.ckpt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
'''
