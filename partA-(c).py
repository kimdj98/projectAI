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



class MLP(nn.Module):
    def __init__(self, i_size, o_size):
        super(MLP, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(i_size, o_size),
            nn.Softmax()
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits



def train(model, input_train, label_train, loss_fn, optimizer):
    train_loss = 0.0
    model.train()
    X = input_train
    y = label_train
    y = y.squeeze(dim=-1)
    # forward + backward + optimize
    pred = model.forward(X)
    loss = loss_fn(pred, y)
    # zero the parameter gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # train_loss += loss.item() * X.size(0)
    train_loss += loss.item()

    return train_loss


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




X_train = np.empty([60000,784])
for i in range(60000):
    X_train[i, :] = train_dataset[i][0].reshape(1, 784)
y_train = np.empty([60000,1])
for i in range(60000):
    y_train[i,0] = train_dataset[i][1]
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)

X_train_2d = torch.from_numpy(X_train_2d).float()
y_train = torch.from_numpy(y_train).long()




loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
i_size = 2
o_size = 10
network = MLP(i_size, o_size)
optimizer = optim.Adam(network.linear_stack.parameters(), learning_rate)
num_epochs = 3000
train_loss = []

for i in range(num_epochs):
    print(f'Epoch{i}----------------')
    loss = train(network, X_train_2d, y_train, loss_fn, optimizer)
    print(loss)
    train_loss.append(loss)

plt.plot(train_loss)
plt.show()




