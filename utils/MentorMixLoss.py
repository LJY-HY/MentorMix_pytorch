import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cat
import torch.distributions.dirichlet as diri

def MentorMixLoss(args,MentorNet, StudentNet, x_i, y_i,v_true, loss_p_prev, loss_p_second_prev, epoch):
    '''
    v_true is set to 0s in this version.
    inputs : 
        x_i         [bsz,C,H,W]
        outputs_i   [bsz,num_class]
        y_i         [bsz]
    intermediate :
        x_j         [bsz,C,H,W]
        outputs_j   [bsz,num_class]
        y_j         [bsz]
    outputs:
        loss        [float]
        gamma       [float]

    Simple threshold function is used as MentorNet in this repository.
    '''
    XLoss = torch.nn.CrossEntropyLoss(reduction='none')
    # MentorNet 1
    bsz = x_i.shape[0]
    x_i, y_i,v_true = x_i.to(args.device), y_i.to(args.device), v_true.to(args.device)
    with torch.no_grad():
        outputs_i = StudentNet(x_i) 
        loss = F.cross_entropy(outputs_i,y_i,reduction='none')                      
        loss_p = args.ema*loss_p_prev + (1-args.ema)*sorted(loss)[int(bsz*args.gamma_p)]
        loss_diff = loss-loss_p
        v = MentorNet(v_true,args.epoch, epoch,loss,loss_diff)   

        # Burn-in Process(needed?)
        if epoch < int(args.epoch*0.2):
            v = torch.bernoulli(torch.ones_like(loss_diff)/2).to(args.device)

    P_v = cat.Categorical(F.softmax(v,dim=0))           
    indices_j = P_v.sample(y_i.shape)                   
    
    # Prepare Mixup
    x_j = x_i[indices_j]
    y_j = y_i[indices_j]
    
    # MIXUP
    Beta = diri.Dirichlet(torch.tensor([args.alpha for _ in range(2)]))
    lambdas = Beta.sample(y_i.shape).to(args.device)
    lambdas_max = lambdas.max(dim=1)[0]                 
    lambdas = v*lambdas_max + (1-v)*(1-lambdas_max)     
    x_tilde = x_i * lambdas.view(lambdas.size(0),1,1,1) + x_j * (1-lambdas).view(lambdas.size(0),1,1,1)
    outputs_tilde = StudentNet(x_tilde)
    
    # Second Reweight
    with torch.no_grad():
        loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
        loss_p_second = args.ema*loss_p_second_prev + (1-args.ema)*sorted(loss)[int(bsz*args.gamma_p)]
        loss_diff = loss-loss_p_second
        v_mix = MentorNet(v_true,args.epoch, epoch,loss,loss_diff)

        # Burn-in Process(needed?)
        if epoch < int(args.epoch*0.2):
            v_mix = torch.bernoulli(torch.ones_like(loss_diff)/2).to(args.device)

    loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
    loss = loss*v_mix
  
    return loss.mean(), loss_p, loss_p_second, v
    
