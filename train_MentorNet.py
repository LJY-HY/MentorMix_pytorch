import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import argparse
from utils.arguments import *
from utils.utils import *
from dataset.cifar import *
from utils.MentorMixLoss import *
def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_MentorNet_arguments()
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
    args.arch = args.MentorNet
    MentorNet = get_architecture(args)
    args.arch = args.StudentNet
    StudentNet = get_architecture(args)
    
    # Get optimizer, scheduler
    optimizer_M, scheduler_M = get_optim_scheduler(args,MentorNet)
    optimizer_S, scheduler_S = get_optim_scheduler(args,StudentNet)

    # Set Loss function
    BCE_loss = nn.BCELoss(reduction = 'mean')
    CE_loss = nn.CrossEntropyLoss(reduction = 'mean')

    if args.dataset == 'cifar10':
        '''
        This is not 'typo'.
        MentorNet trained with CIFAR-10 should not be used for Noisy CIFAR-10. 
        Rather it can be used as MentorNet for StudentNet training with CIFAR-100.
        vice versa
        '''
        path_MentorNet = './checkpoint/cifar100/MentorNet'
        MentorNet_filename = path_MentorNet+'/MentorNet_'
    elif args.dataset == 'cifar100':
        path_MentorNet = './checkpoint/cifar10/MentorNet'
        MentorNet_filename = path_MentorNet+'/MentorNet_'
    MentorNet_filename = MentorNet_filename+args.MentorNet_type+'_'+str(args.noise_rate)+'_trial_'+args.trial

    loss_p_prev = 0
    for epoch in range(args.epoch):
        loss_p_prev = train(args, MentorNet, StudentNet, train_dataloader, optimizer_M, optimizer_S, scheduler_M, scheduler_S, BCE_loss, CE_loss, loss_p_prev, epoch)
        acc = test(args, MentorNet, StudentNet, test_dataloader, optimizer_M, optimizer_S, scheduler_M, scheduler_S, epoch)
        scheduler_M.step()
        scheduler_S.step()
        if not os.path.isdir(path_MentorNet):
            os.makedirs(path_MentorNet)
        torch.save(MentorNet.state_dict(), MentorNet_filename)

def train(args, MentorNet, StudentNet, train_dataloader, optimizer_M, optimizer_S, scheduler_M, scheduler_S, BCE_loss, CE_loss, loss_p_prev, epoch):
    MentorNet.train()
    StudentNet.train()
    MentorNet_loss = 0
    StudentNet_loss = 0
    p_bar = tqdm(range(train_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (inputs, targets, v_true, v_label, index) in enumerate(train_dataloader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        v_label = v_label.to(args.device)
        bsz = inputs.shape[0]

        with torch.no_grad():
            outputs = StudentNet(inputs)
            loss = F.cross_entropy(outputs, targets,reduction='none')
            loss_p = args.ema*loss_p_prev + (1-args.ema)*sorted(loss)[int(bsz*args.gamma_p-1)]
            loss_diff = loss-loss_p
        
        if args.MentorNet_type == 'PD':
            assert args.noise_rate==0.          
            v_true = (loss_diff<0).long().to(args.device)   # closed-form optimal solution
            
            if epoch < int(args.epoch*0.2):
                v_true = torch.bernoulli(torch.ones_like(loss_diff)/2).to(args.device)

        '''
        Train MentorNet.
        calculate the gradient of the MentorNet.
        '''
        v = MentorNet(v_label,args.epoch, epoch,loss,loss_diff)
        loss = BCE_loss(v,v_true.type(torch.FloatTensor).to(args.device))
        MentorNet_loss+=loss.item()

        optimizer_M.zero_grad()
        loss.backward()
        optimizer_M.step()

        for count, idx in enumerate(index):
            train_dataloader.dataset.v_label[idx] = v_true[count].long()
                
        '''
        Train StudentNet
        calculate the gradient of the StudentNet
        '''
        v = v.detach()
        outputs = StudentNet(inputs)
        loss_S = F.cross_entropy(outputs,targets,reduction='none')
        loss_S = loss_S*v
        loss_S = loss_S.mean()
        StudentNet_loss += loss_S.item()

        optimizer_S.zero_grad()
        loss_S.backward()
        optimizer_S.step()

        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. S_loss: {StudentNet_loss:.4f}. l_p: {threshold_loss:.3f}".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_dataloader.__len__(),
                    lr=scheduler_S.optimizer.param_groups[0]['lr'],
                    StudentNet_loss = StudentNet_loss/(batch_idx+1),
                    threshold_loss= loss_p,)
                    )
        p_bar.update()
    p_bar.close()

    return loss_p
    
def test(args, MentorNet, StudentNet, test_dataloader, optimizer_M, optimizer_S, scheduler_M, scheduler_S, epoch):
    MentorNet.eval()
    StudentNet.eval()
    acc = 0
    test_loss = 0
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