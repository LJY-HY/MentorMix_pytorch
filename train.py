import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import argparse
from utils.arguments import get_arguments
from utils.utils import *
from dataset.cifar import *
from utils.MentorMixLoss import *
def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)

    # dataset setting
    if args.dataset in ['cifar10']:
        args.num_classes=10
    elif args.dataset in ['cifar100']:
        args.num_classes=100

    # Get Dataset
    train_dataloader, test_dataloader = globals()[args.dataset](args)
   
    # Get architecture
    args.arch = args.StudentNet
    StudentNet = get_architecture(args)
    args.arch = 'MentorNet'
    MentorNet = get_architecture(args)
    
    # Load MentorNet
    if args.dataset == 'cifar10':
        path_MentorNet = './checkpoint/cifar10/MentorNet'
        MentorNet_filename = path_MentorNet+'/MentorNet_'
    else:
        path_MentorNet = './checkpoint/cifar100/MentorNet'
        MentorNet_filename = path_MentorNet+'/MentorNet_'

    if args.MentorNet_type == 'PD':
        MentorNet_filename = MentorNet_filename+args.MentorNet_type+'_0.0_trial_'+args.trial
    else:
        MentorNet_filename = MentorNet_filename+args.MentorNet_type+'_'+str(args.noise_rate)+'_trial_'+args.trial
    checkpoint = torch.load(MentorNet_filename)
    MentorNet.load_state_dict(checkpoint)
    MentorNet.eval()

    # Get optimizer, scheduler
    optimizer_S, scheduler_S = get_optim_scheduler(args,StudentNet)
    optimizer_M, scheduler_M = get_optim_scheduler(args,MentorNet)

    # Set burn-in epoch
    args.burn_in_epoch = int(args.epoch*0.2)

    path = './checkpoint/'+args.dataset+'/'+args.StudentNet+'_'+args.MentorNet_type+'_'+str(args.noise_rate)+'_trial_'+args.trial
  
    best_acc=0
    loss_p_prev = 0
    loss_p_second_prev = 0
    for epoch in range(args.epoch):
        loss_p_prev, loss_p_second_prev = train(args, MentorNet, StudentNet, train_dataloader, optimizer_S, scheduler_S, loss_p_prev, loss_p_second_prev, epoch)
        acc = test(args, StudentNet, test_dataloader, optimizer_S, scheduler_S, epoch)
        scheduler_S.step()
        if best_acc<acc:
            best_acc = acc
            if not os.path.isdir('checkpoint/'+args.dataset):
                os.makedirs('checkpoint/'+args.dataset)
            torch.save(StudentNet.state_dict(), path)

def train(args, MentorNet, StudentNet, train_dataloader, optimizer_S, scheduler_S, loss_p_prev, loss_p_second_prev, epoch):
    StudentNet.train()
    train_loss = 0
    p_bar = tqdm(range(train_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (inputs, targets,_,v_label,index) in enumerate(train_dataloader):
        '''
        3'rd argument is only used when training with MentorNet_DD
        '''
        loss, loss_p_prev, loss_p_second_prev, v = MentorMixLoss(args,MentorNet,StudentNet,inputs,targets,v_label,loss_p_prev, loss_p_second_prev, epoch)

        # Update v
        for count, idx in enumerate(index):
            train_dataloader.dataset.v_label[idx] = v[count].long()

        optimizer_S.zero_grad()
        loss.backward()
        optimizer_S.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_dataloader.__len__(),
                    lr=scheduler_S.optimizer.param_groups[0]['lr'],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return loss_p_prev, loss_p_second_prev

def test(args, StudentNet, test_dataloader, optimizer_S, scheduler_S, epoch):
    StudentNet.eval()
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = StudentNet(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    lr=scheduler_S.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    print('Accuracy :'+ '%0.4f'%acc )
    return acc


if __name__ == '__main__':
    main()