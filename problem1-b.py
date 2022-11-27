import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

#==================================================================================================================
#   use configuration file to adjust hyper parameters
#==================================================================================================================
import json

with open('config.json') as config_file:
    configs = json.load(config_file)

#==================================================================================================================
#   load data
#==================================================================================================================

# Normalize data with mean=0.5, std=1.0
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

from torchvision.datasets import MNIST

# download path 정의
download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

# option 값 정의
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

# extract X_train from train_dataset
X_train = np.empty([60000, 28, 28])
for i in range(60000):
    X_train[i, :, :] = train_dataset[i][0]

# extract y_train from train_dataset
y_train = np.empty([60000, 1])
for i in range(60000):
    y_train[i, :] = train_dataset[i][1]

# extract X_test from train_dataset
X_test = np.empty([10000, 28, 28])
for i in range(10000):
    X_test[i, :, :] = test_dataset[i][0]

# extract y_test from train_dataset
y_test = np.empty([10000, 1])
for i in range(10000):
    y_test[i, :] = test_dataset[i][1]

#==================================================================================================================
#   implement network architecture
#==================================================================================================================
class CNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # hyperparameters
        self.learning_rate = self.configs["learning_rate"]
        self.epochs = self.configs["epochs"]

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), # 28 X 28
            nn.Conv2d(32, 32, kernel_size=5, padding=2), # 28 x 28
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 14 x 14

            nn.Conv2d(32, 64, kernel_size=5, padding=2), # 14 x 14
            nn.Conv2d(64, 64, kernel_size=5, padding=2), # 14 x 14
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 7 x 7

            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # 7 x 7
            nn.Conv2d(128, 128, kernel_size=5, padding=2),  # 7 x 7
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 3 x 3
            nn.Flatten(), # 128*3*3
        )
        self.fc1 = nn.Linear(128*3*3,2)
        self.fc2 = nn.Linear(2,10)

        # weight initialization (initialize weight with standard normal distribution)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1.0)
                nn.init.normal(m.bias, mean=0.0, std=1.0)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def extract_2d(self, x):
        x = self.model(x)
        x = self.fc1(x)
        return x

    def train(self, train_loader, test_loader, monitor = True):
        epochs = self.configs["epochs"]
        self.loss_history = np.zeros(epochs)

        # set criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

        for iteration in range(epochs):
            for data in train_loader:
                X_train = data[0]
                y_train = data[1]
                # predict the model and calculate the loss
                y_hat = self.forward(X_train)
                loss = self.criterion(y_hat, y_train)  # apply torch.sqrt to use MSE as loss fn

                # back_propagation one time
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # store loss data for plotting
            self.loss_history[iteration] = loss.item()

            # print to console tendency of loss
            if monitor:
                # if (iteration + 1) % 200 == 0:
                print(f'Epoch: {iteration + 1} / {epochs}, Train Loss: {loss.item():.4f}')
                print(f'Train Accuracy: {self.evaluation(train_loader):.4f}, Test Accuracy: {self.evaluation(test_loader):.4f}')
        return None

    def evaluation(self, dataloader):
        count = 0
        accuracy = 0.0
        for data in dataloader:
            X_test = data[0]
            y_test = data[1]
            y_hat = torch.argmax(self.forward(X_test), axis = 1)
            accuracy += torch.sum((y_test == y_hat).float()) # sum all the matched samples

        accuracy = accuracy / (len(dataloader) * dataloader.batch_size) # divide by len of dataset

        return accuracy


X_test = torch.from_numpy(X_test)
X_test = X_test.view(10000, 1, 28, 28)
X_test = X_test.float()

y_test = torch.from_numpy(y_test)
y_test = y_test.long()

X_train = torch.from_numpy(X_train)
X_train = X_train.view(60000, 1, 28, 28)
X_train = X_train.float()

y_train = torch.from_numpy(y_train)
y_train = y_train.long()

network = CNN(configs)
network.train(train_loader, test_loader)