import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torchvision

class LeNet(nn.Module):
    def __init__(self,class_num):
        super(LeNet, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*4*4,120)
        self.fc2=nn.Linear(120,84)
        self.fc_logits=nn.Linear(84,class_num)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        feas = F.relu(self.fc2(x))
        logits = self.fc_logits(feas)
        prediction=F.softmax(logits,dim=1)
        log_prediction = F.log_softmax(logits,dim=1)
        return log_prediction,prediction,logits,feas

class LeNet_Dropout(nn.Module):
    def __init__(self,p=0.5):
        super(LeNet_Dropout, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*4*4,120)
        self.fc2=nn.Linear(120,84)
        self.dropout=nn.Dropout(p)
        self.fc_logits=nn.Linear(84,3)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))

        feas = F.relu(self.dropout(self.fc2(x)))
        logits = self.fc_logits(feas)
        prediction=F.softmax(logits,dim=1)
        log_prediction = F.log_softmax(logits,dim=1)
        return log_prediction,prediction,logits,feas

class LeNet_BN(nn.Module):
    def __init__(self):
        super(LeNet_BN, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.bn_conv1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,16,5)
        self.bn_conv2=nn.BatchNorm2d(16)
        self.fc1=nn.Linear(16*4*4,120)
        self.bn_fc1=nn.BatchNorm1d(120)
        self.fc2=nn.Linear(120,84)
        self.bn_fc2=nn.BatchNorm1d(84)
        self.fc_logits=nn.Linear(84,10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn_conv1(self.conv1(x))),(2,2))
        x = F.max_pool2d(F.relu(self.bn_conv2(self.conv2(x))),2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        feas = F.relu(self.fc2(x))
        logits = self.fc_logits(feas)
        prediction=F.softmax(logits,dim=1)
        log_prediction = F.log_softmax(logits,dim=1)
        return log_prediction,prediction,logits,feas
