import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from utils.utils import *

def svhn(args, train_TF = None, test_TF = None):
    if train_TF is None and test_TF is None:
        train_TF = get_transform(args.in_dataset, 'train')
        test_TF = get_transform(args.in_dataset, 'test')

    train_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'train', transform = train_TF, download=True)
    test_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'test', transform = test_TF, download=True)
    test_indices = list(range(len(test_dataset)))
    test_10000_dataset = Subset(test_dataset, test_indices[:10000])

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_10000_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    if args.tuning:
        test_indices = list(range(len(test_10000_dataset)))
        val_dataset, test_dataset = Subset(test_10000_dataset, test_indices[:1000]), Subset(test_10000_dataset, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, test_dataloader