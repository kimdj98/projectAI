import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

#==================================================================================================================
#   set gpu device
#==================================================================================================================
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device("cuda:0" if USE_CUDA else "cpu")
print(f"device: {torch.cuda.get_device_name(device) }")

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
    transforms.Normalize((0.1307,), (0.3081,))
])

from torchvision.datasets import MNIST

# download path 정의
download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=False)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=False)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=False)



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
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 14 x 14

            nn.Conv2d(32, 64, kernel_size=5, padding=2), # 14 x 14
            nn.Conv2d(64, 64, kernel_size=5, padding=2), # 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 7 x 7

            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # 7 x 7
            nn.Conv2d(128, 128, kernel_size=5, padding=2),  # 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 3 x 3
            nn.Flatten(), # 128*3*3
        )
        self.fc1 = nn.Linear(128*3*3,2)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(2,10)
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def extract_2d(self, x):
        x = self.model(x)
        x = self.fc1(x)
        return x
    #
    # def train(self, train_loader, test_loader, monitor = True):
    #     epochs = self.configs["epochs"]
    #     self.loss_history = np.zeros(epochs)
    #
    #     # set criterion and optimizer
    #     self.criterion = nn.CrossEntropyLoss()
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    #
    #     for iteration in range(epochs):
    #         for data in train_loader:
    #             X_train = data[0]
    #             y_train = data[1]
    #
    #             self.optimizer.zero_grad()
    #
    #             # predict the model and calculate the loss
    #             y_hat = self.forward(X_train)
    #             loss = self.criterion(y_hat, y_train)  # apply torch.sqrt to use MSE as loss fn
    #
    #             # back_propagation one time
    #             loss.backward()
    #             self.optimizer.step()
    #
    #         # store loss data for plotting
    #         self.loss_history[iteration] = loss.item()
    #
    #         # print to console tendency of loss
    #         if monitor:
    #             # if (iteration + 1) % 200 == 0:
    #             print(f'Epoch: {iteration + 1} / {epochs}, Train Loss: {loss.item():.4f}')
    #             print(f'Train Accuracy: {self.evaluation(train_loader):.4f}, Test Accuracy: {self.evaluation(test_loader):.4f}')
    #     return None

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


# function to evaluate network
def evaluation(network, dataloader):
    count = 0
    accuracy = 0.0
    for data in dataloader:
        X_test = data[0].to(device)
        y_test = data[1].to(device)
        y_hat = torch.argmax(network.forward(X_test), axis = 1)
        accuracy += torch.sum((y_test == y_hat).float()) # sum all the matched samples

    accuracy = accuracy / (len(dataloader) * dataloader.batch_size) # divide by len of dataset

    return accuracy



#==================================================================================================================
#   Training Stage
#==================================================================================================================

network = CNN(configs).to(device)
evaluation(network, train_loader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), configs["learning_rate"])

loss_history_per_batch = []
train_accuracy_per_epoch = []
test_accuracy_per_epoch = []
print("Start Training")
for epoch in range(configs["epochs"]):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if (i+1) % 100 == 0:
            running_loss = loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/100:.6f}')
            running_loss = 0.0
            loss_history_per_batch.append(running_loss)
    train_accuracy = evaluation(network, train_loader)
    test_accuracy = evaluation(network, test_loader)

    print(f'train accuracy: {train_accuracy}')
    print(f'test accuracy:  {test_accuracy}')

    train_accuracy_per_epoch.append(train_accuracy)
    test_accuracy_per_epoch.append(test_accuracy)

print('Finished Training')

for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    output_2d = network.extract_2d(inputs)
    if i == 0:
        output_2d_trains = output_2d.cpu().detach().numpy()
    else:
        output_2d_trains = np.vstack((output_2d_trains, output_2d.cpu().detach().numpy()))

labels_2d_trains = train_dataset.train_labels.cpu().detach().numpy()
