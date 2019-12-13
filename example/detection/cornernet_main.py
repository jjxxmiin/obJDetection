import torch.optim as optim
from tools.preprocessing import CornerNet_Processing
from models.Detection.CornerNet import CornerNet
from tools.tester import *
from models.module.loss import *
from datasets.loader import VOC

if torch.cuda.is_available():
    device = 'cuda'
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    #torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'task': 'detection',
    'model': 'cornernet',
    'backbone': 'hourglass',
    'dataset': 'VOC',
    'classes': 20,
    'mode': 'train',

    'lr': 0.0025,
    'epoch': 100,
    'batch_size': 1,
}

preprocessing = CornerNet_Processing()
loader = VOC(configs, preprocessing)
custom_loader = loader.get_loader()

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
            loss, hist = criterion(targets, outputs)
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
        }, './log/hour104_cornernet1.ckpt')

if configs['mode'] == 'test':
    checkpoint = torch.load('./log/hour104_cornernet_1.ckpt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

'''
for i, (images, targets) in enumerate(custom_loader):
    save_tensor_image(images[0], targets[0])
    break
'''
