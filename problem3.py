import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms, utils
from torchinfo import summary


# use GPU if exist
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
# Read ImageNet class id to name mapping
    with open('./Problem2_DATASET/imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()][4:]

# read image and transform
original_image = cv2.imread("./Problem2_DATASET/camel-1536x580.jpg")
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])
])

image = transform(image)
image = image.unsqueeze(0)
#==================================================================================================================
#  FullyConvolutionalResnet18 class
#==================================================================================================================
class FullyConvolutionalResnet18(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(block=models.resnet.BasicBlock, layers = [2,2,2,2], num_classes = num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        # replace AdapativeAvgPool2D with Standard AvgPool2D
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        # Convert the original fc layer to a convolutional layer
        self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    def _forward_impl(self, x):
          # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = self.last_conv(x)
        return x


model = FullyConvolutionalResnet18(pretrained=True).eval()

with torch.no_grad():
    preds = model(image)
    preds = torch.softmax(preds, dim = 1)

    print('Response map shape: ', preds.shape)

    # Find the class with the maximum score in the n x m output map
    pred, class_idx = torch.max(preds, dim=1)
    print(class_idx)

    row_max, row_idx = torch.max(pred, dim=1)
    col_max, col_idx = torch.max(row_max, dim=1)
    predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]

    # Print top predicted class
    print('Predicted Class: ', labels[predicted_class], predicted_class)

# Find the n x m score map for the predicted class
score_map = preds[0, predicted_class, :,:].cpu().numpy()
score_map = score_map[0]

# Resize score map to the original image size
score_map = cv2.resize(score_map, (original_image.shape[1], original_image.shape[0]))

# Binarize score map
_, score_map_for_contours = cv2.threshold(score_map, 0.25, 1, type=cv2.THRESH_BINARY)
score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()

# Find the contour of the binary blob
contours,_ = cv2.findContours(score_map_for_contours, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

# Find bounding box around the object
rect = cv2. boundingRect(contours[0])

# apply score map as a mask to original image
score_map = score_map - np.min(score_map[:])
score_map = score_map / np.max(score_map[:])
score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
masked_image = (original_image * score_map).astype(np.uint8)
# Display bounding box
cv2.rectangle(masked_image, rect[:2],(rect[0] + rect[2], rect[1] + rect[3]), (0,0,255), 2)

# display the images
cv2.imshow("Original Image", original_image)
cv2.imshow("scaled_score_map", score_map)
cv2.imshow("activations_and_bbox", masked_image)
cv2.waitKey(0)

