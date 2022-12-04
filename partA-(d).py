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
    size = y.size(0)
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
def test(model, input_test, label_test, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        X = input_test
        y = label_test
        y = y.squeeze(dim=-1)
        size = y.size(0)
        pred = model.forward(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

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
X_test = np.empty([10000,784])
for i in range(10000):
    X_test[i,:] = test_dataset[i][0].reshape(1,784)
y_test = np.empty([10000,1])
for i in range(10000):
    y_test[i,0] = test_dataset[i][1]

pca = PCA(n_components=2)

X_train_2d = pca.fit_transform(X_train)
X_train_2d = torch.from_numpy(X_train_2d).float()
X_test_2d = pca.fit_transform(X_test)
X_test_2d = torch.from_numpy(X_test_2d).float()

y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()




loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
i_size = 2
o_size = 10
network = MLP(i_size, o_size)
optimizer = optim.Adam(network.linear_stack.parameters(), learning_rate)
num_epochs = 3000
train_loss = []
test_loss = []
test_accuracy = []

for i in range(num_epochs):
    print(f'Epoch{i}----------------')
    loss_train = train(network, X_train_2d, y_train, loss_fn, optimizer)
    loss_test, correct = test(network, X_test_2d, y_test, loss_fn)
    print(f'train loss : {loss_train}')
    train_loss.append(loss_train)
    print(f'test loss : {loss_test}')
    print(f'accuracy : {correct}')
    test_loss.append(loss_test)
    test_accuracy.append(correct)

ax1 = plt.subplot(1,3,1)
ax1.plot(train_loss)
ax2 = plt.subplot(1,3,2)
ax2.plot(test_loss)
ax3 = plt.subplot(1,3,3)
ax3.plot(test_accuracy)
plt.show()




