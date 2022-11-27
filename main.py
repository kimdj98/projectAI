import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

#==================================================================================================================
#   use configuration file to adjust hyper parameters
#==================================================================================================================
import json

with open('config.json') as config_file:
    configs = json.load(config_file)
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
X_train = np.empty([60000, 784])
for i in range(60000):
    X_train[i, :] = train_dataset[i][0].reshape(1,784)

# extract y_train from train_dataset
y_train = np.empty([60000, 1])
for i in range(60000):
    y_train[i, :] = train_dataset[i][1]

# extract X_test from train_dataset
X_test = np.empty([10000, 784])
for i in range(10000):
    X_test[i, :] = test_dataset[i][0].reshape(1,784)

# extract y_test from train_dataset
y_test = np.empty([10000, 1])
for i in range(10000):
    y_test[i, :] = test_dataset[i][1]

# # execute pca analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])

y_train_series = pd.Series(y_train.reshape(-1))
finalDf1 = pd.concat([principalDf, y_train_series], axis = 1)

# plt.scatter(np.array(finalDf1['principal component 1']),
#             np.array(finalDf1['principal component 2']),
#             c = np.array(finalDf1.loc[:,0]))

# execute pca analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_test)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])

y_test_series = pd.Series(y_test.reshape(-1))
finalDf2 = pd.concat([principalDf, y_test_series], axis = 1)

# plt.scatter(np.array(finalDf2['principal component 1']),
#             np.array(finalDf2['principal component 2']),
#             c = np.array(finalDf2.loc[:,0]),
#             label = np.array(finalDf2.loc[:,0]))
# plt.show()


# network architecture
class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # network architecture
        self.num_inputs = self.configs["num_inputs"]
        self.num_outputs = self.configs["num_outputs"]

        # hyperparameters
        self.learning_rate = self.configs["learning_rate"]
        self.epochs = self.configs["epochs"]

        self.model = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_outputs),
            nn.Softmax()
        )

        # set criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        # weight initialization (initialize weight with standard normal distribution)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1.0)
                nn.init.normal(m.bias, mean=0.0, std=1.0)

    def forward(self, data):
        return self.model(data)

    def train(self, X_train, y_train, monitor = True):
        epochs = self.configs["epochs"]
        self.loss_history = np.zeros(epochs)

        for iteration in range(epochs):
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
                if (iteration + 1) % 200 == 0:
                    print(f'Epoch: {iteration + 1} / {epochs}, Loss: {loss.item():.4f}')

        return None

    def evaluation(self, X_test, y_test):
        y_hat = torch.argmax(self.forward(X_test), axis = 1)
        return torch.mean((y_test == y_hat).float())

X_train = np.array(X_train) # pandas dataframe to numpy
y_train = np.array(y_train) # pandas dataframe to numpy
X_test = np.array(X_test) # pandas dataframe to numpy
y_test = np.array(y_test) # pandas dataframe to numpy

X_train = torch.from_numpy(X_train).float() # numpy to torch tensor
y_train = torch.from_numpy(y_train).long() # numpy to torch tensor
X_test = torch.from_numpy(X_test).float() # numpy to torch tensor
y_test = torch.from_numpy(y_test).long() # numpy to torch tensor

X_train = np.array(finalDf1[['principal component 1', 'principal component 2']])
y_train = np.array(finalDf1.loc[:,0])

X_train = torch.from_numpy(X_train).float() # numpy to torch tensor
y_train = torch.from_numpy(y_train).long() # numpy to torch tensor

X_test = np.array(finalDf2[['principal component 1', 'principal component 2']])
y_test = np.array(finalDf2.loc[:,0])

X_test = torch.from_numpy(X_test).float() # numpy to torch tensor
y_test = torch.from_numpy(y_test).long() # numpy to torch tensor

network = MLP(configs)
network.train(X_train, y_train)
