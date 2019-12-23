import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
from src.models.feature.ResNet import ResNet18
from src.datasets.loader import CIFAR10, STL10

sys.path.append('.')

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'task': 'classify',
    'model': 'Resnet18',
    'dataset': 'STL10',
    'classes': 10,
    'mode': 'train',

    'lr': 0.001,
    'epochs': 300,
    'batch_size': 64,
    'save_path': './resnet18_stl10_gap_model.pth',
    'load_path': './resnet18_stl10_gap_model.pth'
}

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#class_name = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# augmentation
train_transformer = transforms.Compose([transforms.Resize(128),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(15),
                                        transforms.RandomCrop(size=(128, 128), padding=12),
                                        transforms.ToTensor()])

test_transformer = transforms.Compose([transforms.Resize(128),
                                       transforms.ToTensor()])

# datasets/loader/downloads
datasets = CIFAR10(batch_size=configs['batch_size'])
#datasets = STL10(batch_size=configs['batch_size'])

train_loader = datasets.get_loader(train_transformer, 'train')
test_loader = datasets.get_loader(test_transformer, 'test')

alpha = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
cal = ['plus', 'mul']

for c in cal:
    for a in alpha:
        configs['save_path'] = './resnet18_cifar10_gvap_model_%s_%d.pth' % (c, a*10)
        # model
        model = ResNet18(classes=configs['classes'], alpha=a, cal=c).to(device)
        model.apply(model.weights_init)

        # summary
        summary(model, (3, 128, 128))

        # cost
        criterion = nn.CrossEntropyLoss().to(device)

        # optimizer/scheduler
        optimizer = optim.Adam(model.parameters(), lr=configs['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=[100, 150, 250],
                                                   gamma=0.5)

        # model load
        if os.path.exists(configs['load_path']):
            print("model loading...")
            model.load_state_dict(torch.load(configs['load_path']))
            print("model load complete")
        else:
            assert configs['mode'] != 'test_img', "[Test Mode] load path not existing"

        best_valid_acc = 0
        train_iter = len(train_loader)
        test_iter = len(test_loader)

        # train
        if configs['mode'] == 'train':
            for epoch in range(configs['epochs']):
                train_loss = 0
                valid_loss = 0

                n_train_correct = 0
                n_valid_correct = 0

                scheduler.step()
                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    # forward
                    pred = model(images)
                    # acc
                    _, predicted = torch.max(pred, 1)
                    n_train_correct += (predicted == labels).sum().item()
                    # loss
                    loss = criterion(pred, labels)
                    train_loss += loss.item()
                    # backward
                    loss.backward()
                    # weight update
                    optimizer.step()
                    #print("Batch [%d / %d]"%(train_iter, i))

                train_acc = n_train_correct / (train_iter * configs['batch_size'])
                train_loss = train_loss / train_iter

                with torch.no_grad():
                    for images, labels in test_loader:
                        images, label = images.to(device), labels.to(device)

                        pred = model(images)

                        # acc
                        _, predicted = torch.max(pred, 1)
                        n_valid_correct += (predicted == labels).sum().item()
                        # loss
                        loss = criterion(pred, labels)
                        valid_loss += loss.item()

                valid_acc = n_valid_correct / (test_iter * configs['batch_size'])
                valid_loss = valid_loss / test_iter

                print("Epoch [%d / %d] Train [Acc / Loss] : [%f / %f] || Valid [Acc / Loss] : [%f / %f]"
                      % (configs['epochs'], epoch, train_acc, train_loss, valid_acc, valid_loss))

                if valid_acc > best_valid_acc:
                    print("model saved")
                    torch.save(model.state_dict(), configs['save_path'])
                    best_valid_acc = valid_acc

# test_img
if configs['mode'] == 'test':
    model.eval()

    test_loss = 0
    n_test_correct = 0

    for images, labels in test_loader:
        images, label = images.to(device), labels.to(device)

        pred = model(images)

        # acc
        _, predicted = torch.max(pred, 1)
        n_test_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        test_loss += loss.item()

    test_acc = n_test_correct / (test_iter * configs['batch_size'])
    test_loss = test_loss / test_iter

    print("TEST [Acc / Loss] : [%f / %f]" % (test_acc, test_loss))

'''
from src.tools.tester import save_tensor_image
# testing
for images, labels in test_loader:
    save_tensor_image(images[0], saved_path='./test_img.png')
    break
'''