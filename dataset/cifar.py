import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
from utils.utils import *
from dataset.CIFAR import CIFAR10,CIFAR100

def cifar10(args, train_TF = None, test_TF = None):
    if train_TF is None and test_TF is None:
        train_TF = get_transform(args.dataset, 'train')
        test_TF = get_transform(args.dataset, 'test')
  
    train_dataset = CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10/', train_MentorNet = args.train_MentorNet, corruption_prob=args.noise_rate, train=True, transform = train_TF, download=True)
    test_dataset = CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10/', train=False, transform = test_TF, download=False)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
    return train_dataloader, test_dataloader

def cifar100(args, train_TF = None, test_TF = None):
    if train_TF is None and test_TF is None:
        train_TF = get_transform(args.dataset, 'train')
        test_TF = get_transform(args.dataset, 'test')

    train_dataset = CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100/', train_MentorNet = args.train_MentorNet, corruption_prob=args.noise_rate, train=True, transform = train_TF, download=True)
    test_dataset = CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100/', train=False, transform = test_TF, download=False)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
    return train_dataloader, test_dataloader