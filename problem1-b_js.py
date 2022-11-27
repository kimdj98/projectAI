
import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=1, padding=2), # 28 x 28
            nn.Conv2d(32,32,kernel_size =5, stride=1, padding=2), #28x28
            # nn.BacthNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #14x14

            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 14x14
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  # 14x14
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0),  # 7x7


            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # 7x7
            nn.Conv2d(128, 128, kernel_size=5, padding=2),  # 7x7
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)   # 3x3
        )
        self.fc1 = nn.Linear(128 * 3 * 3, 2)
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        logits = self.features(x)
        logits = logits.view(logits.size(0), 128 * 3 * 3)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        return logits

# Normalize data with mean=0.5, std=1.0
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1.0,))
])

# download path 정의
download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

valid_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []


loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
network = CNN()
optimizer = optim.Adam(network.features.parameters(), learning_rate)
num_epochs = 100

for i in range(num_epochs):
    correct_train = 0
    correct_test = 0

    network.train()
    for j, data in enumerate(train_loader,0):
        inputs = data[0]
        labels = data[1]
        optimizer.zero_grad()

        outputs = network(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


    network.eval()
    with torch.no_grad():
        for data in train_loader:
            inputs_train = data[0]
            labels_train = data[1]
            size_train = len(labels_train)
            outputs_train = network(inputs_train)
            _, predicted = torch.max(outputs_train, 1)
            correct_train = correct_train + (predicted == labels_train).sum().item()
            loss_train = loss_fn(outputs_train, labels_train)
            print(f"train loss 값은 {loss_train}")
            print(f"train accuracy는 {correct_train/size_train}")
        for data in test_loader:
            inputs_test = data[0]
            labels_test = data[1]
            size_test = len(labels_test)
            outputs_test = network(inputs_test)
            _, predicted = torch.max(outputs_test, 1)
            correct_test = correct_test + (predicted == labels_test).sum().item()
            loss_test = loss_fn(outputs_test, labels_test)
            print(f'test loss 값은 {loss_test}')
            print(f'test accuracy는 {correct_test/size_test}')

    train_loss.append(loss_train.item())
    train_accuracy.append(correct_train/size_train)
    test_loss.append(loss_test.item())
    test_accuracy.append(correct_test / size_test)


