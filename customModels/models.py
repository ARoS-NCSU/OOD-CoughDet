import numpy as np

import torchaudio
import copy
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW,SGD
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import timm

from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import models
from tqdm import tqdm as tq

# Source
# https://github.com/dlmacedo/entropic-out-of-distribution-detection

class IsoMaxPlusLossFirstPart(nn.Module):

    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes,temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)
    
    def forward(self, features):
        #distances = torch.abs(self.distance_scale) * F.pairwise_distance(
        #    F.normalize(features).unsqueeze(2), F.normalize(self.prototypes).t().unsqueeze(0), p=2.0)       
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature
    

            
class DiscEntropicResNet(torch.nn.Module):
    
    def __init__(self):
        
        # constructor of torch.nn.Module
        
        super(DiscEntropicResNet, self).__init__()
        
        # initialize feature extractor
        
        self.model = models.resnet50(pretrained=True)
        self.model.conv1=torch.nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size[0], 
                      stride=self.model.conv1.stride[0], padding=self.model.conv1.padding[0])
        num_ftrs = self.model.fc.in_features
        
        # 
        #Replace last layer with IsoMaxPlus loss
        #############################################################################
        #self.classifier = nn.Linear(in_planes, num_classes)
        self.model.fc = IsoMaxPlusLossFirstPart(num_ftrs, 2)
        #############################################################################
        
                                        
    def forward(self, x):
       
        x = self.model(x)
        
        return x

class DiscEntropicEfficientNet(torch.nn.Module):

    def __init__(self):

       # constructor of torch.nn.Module
        super(DiscEntropicEfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21k', pretrained=True, in_chans=1)
        self.model.conv_stem=torch.nn.Conv2d(1, self.model.conv_stem.out_channels, kernel_size=self.model.conv_stem.kernel_size[0],
                     stride=self.model.conv_stem.stride[0], padding=self.model.conv_stem.padding[0])
        num_ftrs = self.model.classifier.in_features

        #
        #Replace last layer with IsoMaxPlus loss
        #############################################################################
        #self.classifier = nn.Linear(in_planes, num_classes)
        self.model.classifier = IsoMaxPlusLossFirstPart(num_ftrs, 2)
        #############################################################################

        
    def forward(self, x):
 
        x = self.model(x)

        return x
    
class DiscBaseline(torch.nn.Module):
    
    def __init__(self):
        
        # constructor of torch.nn.Module
        
        super(DiscBaseline, self).__init__()
        
        # initialize feature extractor
        
        self.model = models.resnet50(pretrained=True)
        self.model.conv1=torch.nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size[0], 
                      stride=self.model.conv1.stride[0], padding=self.model.conv1.padding[0])
        num_ftrs = self.model.fc.in_features
        
        # 
        #DO NOT Replace last layer with IsoMaxPlus loss
        #############################################################################
        self.classifier = nn.Linear(num_ftrs, 2)
        #self.model.fc = IsoMaxPlusLossFirstPart(num_ftrs, 2,mode='Train')
        #############################################################################
        
        
                                        
    def forward(self, x):
       
        x = self.model(x)
        
        return x


class DiscConfidence(torch.nn.Module):
    
    def __init__(self):
        
        # constructor of torch.nn.Module
        
        super(DiscConfidence, self).__init__()
        
       
        self.model = models.resnet50(pretrained=True)
        self.model.conv1=torch.nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size[0], 
                      stride=self.model.conv1.stride[0], padding=self.model.conv1.padding[0])
        
        #strip the last layer to only extract features
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(self.num_ftrs, 2)
        self.conf = torch.nn.Linear(self.num_ftrs, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.reshape(-1, self.num_ftrs)
        pred = self.model.fc(x)
        confidence=self.conf(x)
        return pred,confidence