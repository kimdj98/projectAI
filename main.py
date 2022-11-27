import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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

# execute pca analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])
y_train_series = pd.Series(y_train.reshape(-1))
finalDf = pd.concat([principalDf, y_train_series], axis = 1)

plt.scatter(np.array(finalDf['principal component 1']), np.array(finalDf['principal component 2']), cmap = np.array(finalDf.loc[:,0]))
