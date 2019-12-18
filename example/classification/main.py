import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary

from tools.tester import save_tensor_image
from models.Feature.ResNet import ResNet18
from datasets.loader import CIFAR10

sys.path.append('.')

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'task': 'classify',
    'model': 'Inceptionv1',
    'dataset': 'STL10',
    'classes': 10,
    'mode': 'train',

    'lr': 0.001,
    'epochs': 200,
    'batch_size': 64,
    'save_path': './resnet18_model.pth',
    'load_path': './resnet18_model.pth'
}

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# augmentation
train_transformer = transforms.Compose([transforms.Resize(128),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(size=(128, 128), padding=4),
                                        transforms.ToTensor()])

test_transformer = transforms.Compose([transforms.Resize(32),
                                       transforms.ToTensor()])

# datasets/loader/downloads
datasets = CIFAR10(batch_size=configs['batch_size'])

train_loader = datasets.get_loader(train_transformer, 'train')
test_loader = datasets.get_loader(test_transformer, 'test')

# model
model = ResNet18(classes=configs['classes']).to(device)

# summary
summary(model, (3, 128, 128))

# cost
criterion = nn.CrossEntropyLoss().to(device)

# optimizer/scheduler
optimizer = optim.Adam(model.parameters(), lr=configs['lr'])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[30, 60, 90],
                                           gamma=0.5)

# model load
if os.path.exists(configs['load_path']):
    print("model loading...")
    model.load_state_dict(torch.load(configs['load_path']))
    print("model load complete")


# augmentation image testing
for i, (images, labels) in enumerate(train_loader):
    save_tensor_image(images[0], saved_path='test.png')
    break

best_valid_acc = 0
train_iter = len(train_loader)
test_iter = len(test_loader)

# train
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
