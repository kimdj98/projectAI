import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import pandas as pd
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
# (b)
X_test = np.empty([10000,784])
for i in range(10000):
    X_test[i,:] = test_dataset[i][0].reshape(1,784)
y_test = np.empty([10000,1])
for i in range(10000):
    y_test[i,0] = test_dataset[i][1]
y_test_series = pd.Series(y_test.reshape(-1))
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_test)
df = pd.DataFrame({'pca_comp1': principalComponents[:, 0],
                    'pca_comp2': principalComponents[:, 1],
                    'label': y_test[:, 0]})
groups = df.groupby('label')
fig, ax = plt.subplots(figsize=(10,10))
for name, group in groups:
    ax.plot(group.pca_comp1,
            group.pca_comp2,
            marker='o',
            linestyle='',
            label=name)
ax.legend(fontsize=12, loc='upper right')
plt.title('2D projection of MNIST', fontsize=20)
plt.xlabel('pca component 1', fontsize=14)
plt.ylabel('pca component 2', fontsize=14)
plt.show()
