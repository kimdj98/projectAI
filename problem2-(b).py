import torch
from torchinfo import summary
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# use GPU if exist
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#==================================================================================================================
#  import pretrained model
#==================================================================================================================
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# change conv1 stride into (1,1)
temp = model.conv1.weight # store weight
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
model.conv1.weight = temp # restore weight

temp = model.layer2[0].conv1.weight
model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.layer2[0].conv1.weight = temp

temp = model.layer2[0].downsample[0].weight
model.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.layer2[0].downsample[0].weight = temp

model

#==================================================================================================================
#  import face dataset
#==================================================================================================================
# 학습을 위한 데이터 증가(Augmentation)와 일반화하기
# 단지 검증을 위한 일반화하기
transform_train = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4704,0.4534,0.4571),(0.1929,0.2107,0.2006))
])

transform_test = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4704,0.4534,0.4571),(0.1929,0.2107,0.2006))
])

batch_size = 64

data_dir = '/home/kimdj/PycharmProjects/pythonProject/projectAI/Problem2_DATASET/face_dataset'
train_set = ImageFolder(root=data_dir, transform = transform_train)
test_set = ImageFolder(root=data_dir, transform = transform_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

#==================================================================================================================
#  problem2 -(a)
#==================================================================================================================

# freeze all weights
for param in model.parameters():
    param.requires_grad = False

# See the weights and bias in model
#model.state_dict().keys()

#####################
# for problem 2-(b) #
#####################
# unfreeze specific weights
for name, param in model.named_parameters():
    if name in ['fc.weight', 'fc.bias']:
        param.requires_grad = True

#####################
# for problem 2-(c) #
#####################
# # unfreeze specific weights
# intermediate_params = []
# last_params = []
# for name, param in model.named_parameters():
#     if "fc2" in name:
#         param.requires_grad = True
#         last_params.append(param)
#     if "layer4" in name:
#         param.requires_grad = True
#         intermediate_params.append(param)

# change last layer feature size 1000 to 100
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)
nn.init.xavier_uniform_(model.fc.weight)
model = model.to(device)

# Apply dropout
model.layer1[0].add_module('dropout',nn.Dropout(0.5))
model.layer1[1].add_module('dropout',nn.Dropout(0.5))
model.layer2[0].add_module('dropout',nn.Dropout(0.5))
model.layer2[1].add_module('dropout',nn.Dropout(0.5))
model.layer3[0].add_module('dropout',nn.Dropout(0.5))
model.layer3[1].add_module('dropout',nn.Dropout(0.5))
model.layer4[0].add_module('dropout',nn.Dropout(0.5))
model.layer4[1].add_module('dropout',nn.Dropout(0.5))

print("params to learn")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)


#==================================================================================================================
#   Training Stage
#==================================================================================================================
def evaluation(network, dataloader):
    total = 0
    for data in dataloader:
        total += len(data[0])

    accuracy = 0.0
    for data in dataloader:
        X_test = data[0].to(device)
        y_test = data[1].to(device)
        y_hat = torch.argmax(network.forward(X_test), axis = 1)
        accuracy += torch.sum((y_test == y_hat).float()) # sum all the matched samples

    accuracy = accuracy / total # divide by len of dataset
    return accuracy

network = model
# evaluation(network, train_loader)

learning_rate = 0.001
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, learning_rate)


# optimizer = torch.optim.SGD([
#     {'params': list(model.parameters())[:-1], 'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4},
#     {'params': list(model.parameters())[-1], 'lr': 5e-3, 'momentum': 0.9, 'weight_decay': 1e-4}
# ])

loss_history_per_batch = []
train_accuracy_per_epoch = []
test_accuracy_per_epoch = []

print('==================================')
print("Start Training")
for epoch in range(epochs):  # loop over the dataset multiple times
    print(f"epoch: {epoch+1}")
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
        if (i+1) % 10 == 0:
            running_loss = loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/10:.6f}')
            loss_history_per_batch.append(running_loss)
            running_loss = 0.0

    train_accuracy = evaluation(network, train_loader)
    test_accuracy = evaluation(network, test_loader)

    print(f'train accuracy: {train_accuracy}')
    print(f'test accuracy:  {test_accuracy}')

    train_accuracy_per_epoch.append(train_accuracy.cpu().detach().numpy())
    test_accuracy_per_epoch.append(test_accuracy.cpu().detach().numpy())

print('Finished Training')

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy_per_epoch, label = 'train_accuracy')
plt.plot(test_accuracy_per_epoch, label = 'test_accuracy')
plt.title('accuracy plot')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss_history_per_batch, label = 'loss')
plt.title('loss plot')
plt.xlabel('batch')
plt.ylabel('Accuracy')
plt.show()