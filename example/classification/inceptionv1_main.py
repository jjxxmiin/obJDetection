import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary

from tools.tester import save_tensor_image
from models.Feature.GooLeNet import GoogLeNet
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

    'lr': 0.01,
    'epochs': 200,
    'batch_size': 32,
    'save_path': './model.pth'
}

# augmentation
transformer = transforms.Compose([transforms.Resize(32),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# datasets/loader/downloads
datasets = CIFAR10(batch_size=configs['batch_size'], transformer=transformer)

train_loader = datasets.get_loader('train')
test_loader = datasets.get_loader('test')

# model
model = GoogLeNet(classes=configs['classes']).to(device)

# summary
summary(model, (3, 32, 32))

# cost
criterion = nn.CrossEntropyLoss().to(device)

# optimizer/scheduler
optimizer = optim.Adam(model.parameters(), lr=configs['lr'])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[50, 100, 150],
                                           gamma=0.5)

# augmentation image testing
for i, (images, labels) in enumerate(train_loader):
    save_tensor_image(images[0])
    break

best_valid_acc = 0

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

    train_acc = n_train_correct / (len(train_loader) * configs['batch_size'])
    train_loss = train_loss / len(train_loader)

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

    valid_acc = n_valid_correct / (len(test_loader) * configs['batch_size'])
    valid_loss = valid_loss / len(test_loader)

    print("Train [Acc / Loss] : [%f / %f] || Valid [Acc / Loss] : [%f / %f]" % (train_acc, train_loss, valid_acc, valid_loss))

    if valid_acc > best_valid_acc:
        print("model saved")
        torch.save(model.state_dict(), configs['save_path'])
        best_valid_acc = valid_acc