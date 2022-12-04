import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder


# 학습을 위한 데이터 증가(Augmentation)와 일반화하기
# 단지 검증을 위한 일반화하기
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4704,0.4534,0.4571),(0.1929,0.2107,0.2006))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4704,0.4534,0.4571),(0.1929,0.2107,0.2006))
])

batch_size = 64

data_dir = '/Users/jusuklee/PycharmProjects/Introduction_AI/face_dataset'
train_set = ImageFolder(root='/Users/jusuklee/PycharmProjects/Introduction_AI/face_dataset/facescrub_train', transform= transform_train)
test_set = ImageFolder(root='/Users/jusuklee/PycharmProjects/Introduction_AI/face_dataset/facescrub_test',transform = transform_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
#
# # Get a batch of training data
# for data in train_loader:
#     inputs = data[0]
#     classes = data[1]
#
# class_names = train_set.classes
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
#
# imshow(out, title=[class_names[x] for x in classes])

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

network = torchvision.models.resnet18(pretrained=True)

# network to 8*8*512
# change conv1 stride into (1,1)
temp = network.conv1.weight  # store weight
network.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
network.conv1.weight = temp  # restore weight

temp = network.layer2[0].conv1.weight
network.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
network.layer2[0].conv1.weight = temp

temp = network.layer2[0].downsample[0].weight
network.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
network.layer2[0].downsample[0].weight = temp

# Apply dropout
network.layer1[0].add_module('dropout',nn.Dropout(0.5))
network.layer1[1].add_module('dropout',nn.Dropout(0.5))
network.layer2[0].add_module('dropout',nn.Dropout(0.5))
network.layer2[1].add_module('dropout',nn.Dropout(0.5))
network.layer3[0].add_module('dropout',nn.Dropout(0.5))
network.layer3[1].add_module('dropout',nn.Dropout(0.5))
network.layer4[0].add_module('dropout',nn.Dropout(0.5))
network.layer4[1].add_module('dropout',nn.Dropout(0.5))
# freeze all weights
for param in network.parameters():
    param.requires_grad = False

# See the weights and bias in model
#network.state_dict().keys()


# unfreeze specific weights
for name, param in network.named_parameters():
    if name in ['fc.weight', 'fc.bias', 'layer4.0.conv1.weight','layer4.0.bn1.weight','layer4.0.bn1.bias','layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked']:
        param.requires_grad = True

# 마지막 레이어의 차원을 10차원으로 조절
num_features = network.fc.in_features
network.fc = nn.Linear(num_features, 100)
network = network.to(device)

# hyperparameters
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.Adam(network.parameters(), learning_rate)


# optimizer = torch.optim.SGD([
#     {'params': list(model.parameters())[:-1], 'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4},
#     {'params': list(model.parameters())[-1], 'lr': 5e-3, 'momentum': 0.9, 'weight_decay': 1e-4}
# ])

num_epochs = 10

for i in range(num_epochs):
    correct_train = 0
    correct_test = 0

    network.train()
    for j, data in enumerate(train_loader, 0):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        optimizer.zero_grad()

        outputs = network(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    network.eval()
    with torch.no_grad():
        size_train = 0
        size_test = 0
        for data in train_loader:
            inputs_train = data[0].to(device)
            labels_train = data[1].to(device)
            size_train += labels_train.size(0)
            outputs_train = network(inputs_train)
            _, predicted = torch.max(outputs_train, 1)
            correct_train = correct_train + (predicted == labels_train).sum().item()
            loss_train = loss_fn(outputs_train, labels_train)

        for data in test_loader:
            inputs_test = data[0].to(device)
            labels_test = data[1].to(device)
            size_test += labels_test.size(0)
            outputs_test = network(inputs_test)
            _, predicted = torch.max(outputs_test, 1)
            correct_test = correct_test + (predicted == labels_test).sum().item()
            loss_test = loss_fn(outputs_test, labels_test)

    print(f'train loss 값은 {loss_train.item()}')
    print(f'train accuracy는 {correct_train / size_train}')
    print(f'test loss 값은 {loss_test.item()}')
    print(f'test accuracy는 {correct_test / size_test}')
    train_loss.append(loss_train.item())
    train_accuracy.append(correct_train / size_train)
    test_loss.append(loss_test.item())
    test_accuracy.append(correct_test / size_test)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train loss')
plt.plot(test_loss, label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss value')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train accuracy')
plt.plot(test_accuracy, label='Test accuracy')
plt.title('Training and Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
