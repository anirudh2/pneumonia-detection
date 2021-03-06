import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pdb

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import model.aadensenet
import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet121, self).__init__()

#         self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        self.densenet121 = model.aadensenet.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
#         pdb.set_trace()

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x
