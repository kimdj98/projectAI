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
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # 28 x 28
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),  # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=5,stride=1, padding=2),  # 14x14
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # 7x7
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),  # 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 3x3
        )
        self.fc1 = nn.Linear(128 * 3 * 3, 2)
        self.fc2 = nn.Linear(2, 10)
        self.apply(self.__init__weights)

    def forward(self, x):
        logits = self.features(x)
        logits = logits.view(logits.size(0), -1)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        return logits

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


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
# X_train = np.empty([60000,28,28])
# for i in range(60000):
#     X_train[i, :, :] = train_dataset[i][0]
# y_train = np.empty([60000,1])
# for i in range(60000):
#     y_train[i,0] = train_dataset[i][1]
# X_test = np.empty([10000,28,28])
# for i in range(10000):
#     X_test[i,:,:] = test_dataset[i][0]
# y_test = np.empty([10000,1])
# for i in range(10000):
#     y_test[i,0] = test_dataset[i][1]
# X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).long()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).long()


device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
network = CNN().to(device)
optimizer = optim.Adam(network.parameters(), learning_rate)
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
torch.save(network, 'model_weights.pth')
network1 = torch.load("model_weights.pth")
batch_size_train = 60000
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size_train,
                          shuffle=True)
for data in train_loader:
    inputs_train = data[0].to(device)
    labels_train = data[1].to(device)
    stage4_cnn_train = network1.features(inputs_train)  # size [60000?, 128, 3, 3]
    stage4_out_train = network1.fc1(stage4_cnn_train.view(stage4_cnn_train.size(0), -1))
    stage4_out_train = stage4_out_train.cpu().detach().numpy()
    stage4_label_train = labels_train.cpu().detach().numpy()


batch_size_test = 10000
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size_test,
                         shuffle=True)
for data in test_loader:
    inputs_test = data[0].to(device)
    labels_test = data[1].to(device)
    stage4_cnn_test = network1.features(inputs_test)  # size [10000?, 128, 3, 3]
    stage4_out_test = network1.fc1(stage4_cnn_test.view(stage4_cnn_test.size(0), -1))
    stage4_out_test = stage4_out_test.cpu().detach().numpy()
    stage4_label_test = labels_test.cpu().detach().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.plot(train_loss, label='Train loss')
plt.plot(test_loss, label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss value')

plt.subplot(1, 4, 2)
plt.plot(train_accuracy, label='Train accuracy')
plt.plot(test_accuracy, label='Test accuracy')
plt.title('Training and Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1,4,3)
plt.scatter(stage4_out_train[:,0], stage4_out_train[:,1], c = stage4_label_train, cmap='rainbow')
plt.colorbar()
plt.title('training')
plt.xlabel('neuron 1')
plt.ylabel('neuron 2')

plt.subplot(1, 4, 4)
plt.scatter(stage4_out_test[:,0], stage4_out_test[:,1], c = stage4_label_test, cmap='rainbow')
plt.colorbar()
plt.title('testing')
plt.xlabel('neuron 1')
plt.ylabel('neuron 2')
plt.show()







