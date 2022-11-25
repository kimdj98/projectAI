import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA


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


# Train a PCA projection matrix W
X_train = np.empty([60000,784])
for i in range(60000):
    X_train[i, :] = train_dataset[i][0].reshape(1, 784)
# Show the example image of MINIST data
# plt.imshow(X_train[100].reshape(28,28),cmap='gray')
# plt.show()

# Apply PCA
pca = PCA(n_components = 100)
X_transformed = pca.fit_transform(X_train)

# We center the data and compute the sample covariance matrix.
X_centered = X_train - np.mean(X_train, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / 60000
eigenvalues = pca.explained_variance_
eigenvalues_list = []
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
    # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    eigenvalues_list.append(eigenvalue)

plt.plot(eigenvalues_list)
plt.show()



