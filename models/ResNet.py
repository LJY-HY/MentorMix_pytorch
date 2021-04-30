import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np

model_urls = {
    'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18(args):
    model = ResNet(BasicBlock, [2,2,2,2], args.num_classes) 
    if args.finetuning:
        model.fc = nn.Linear(512,1000)
        state_dict = load_state_dict_from_url(model_urls[args.arch])
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad=False
        model.fc = nn.Linear(512,args.num_classes)
        model.fc.weight.requires_grad=True
        model.fc.bias.requires_grad=True
    return model

def ResNet34(args):
    model = ResNet(BasicBlock, [3,4,6,3], args.num_classes) 
    if args.finetuning:
        model.fc = nn.Linear(512,1000)
        state_dict = load_state_dict_from_url(model_urls[args.arch])
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad=False
        model.fc = nn.Linear(512,args.num_classes)
        model.fc.weight.requires_grad=True
        model.fc.bias.requires_grad=True
    return model

def ResNet50(args):
    model = ResNet(Bottleneck, [3,4,6,3], args.num_classes) 
    if args.finetuning:
        model.fc = nn.Linear(2048,1000)
        state_dict = load_state_dict_from_url(model_urls[args.arch])
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad=False
        model.fc = nn.Linear(2048,args.num_classes)
        model.fc.weight.requires_grad=True
        model.fc.bias.requires_grad=True
    return model

def ResNet101(args):
    model = ResNet(Bottleneck, [3,4,23,3], args.num_classes) 
    if args.finetuning:
        model.fc = nn.Linear(2048,1000)
        state_dict = load_state_dict_from_url(model_urls[args.arch])
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad=False
        model.fc = nn.Linear(2048,args.num_classes)
        model.fc.weight.requires_grad=True
        model.fc.bias.requires_grad=True
    return model

def ResNet152(args):
    model = ResNet(Bottleneck, [3,8,36,3], args.num_classes) 
    if args.finetuning:
        model.fc = nn.Linear(2048,1000)
        state_dict = load_state_dict_from_url(model_urls[args.arch])
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad=False
        model.fc = nn.Linear(2048,args.num_classes)
        model.fc.weight.requires_grad=True
        model.fc.bias.requires_grad=True
    return model
