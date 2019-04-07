## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
            
        # input                          # 224*224*3
        self.conv1 = nn.Conv2d(1, 64, 3) # 222*222*64
        # relu
        self.maxpool = nn.MaxPool2d(2, 2) # 111*111*64
        
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3) # 109*109*128
        # relu
        # maxpool                         # 54*54*128
        
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3) # 52*52*256
        # relu
        # maxpool                          # 26*26*256
        
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3) # 24*24*512
        # relu
        # maxpool                           # 12*12*512
        
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3) # 10*10*1024
        # relu
        # maxpool                           # 5*5*1024
        
        # flatten
        self.batchnorm5 = nn.BatchNorm1d(5*5*1024) # 25600
        self.dense1 = nn.Linear(25600, 1024) # 1024
        # relu
        self.dropout = nn.Dropout(p=0.5)
        
        self.dense2 = nn.Linear(1024, 136) #136

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.maxpool(F.relu(self.conv1(x)))
        
        x = self.batchnorm1(x)
        x = self.maxpool(F.relu(self.conv2(x)))
        
        x = self.batchnorm2(x)
        x = self.maxpool(F.relu(self.conv3(x)))
        
        x = self.batchnorm3(x)
        x = self.maxpool(F.relu(self.conv4(x)))
        
        x = self.batchnorm4(x)
        x = self.maxpool(F.relu(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        
        x = self.batchnorm5(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        
        x = self.dense2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
