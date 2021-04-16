import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cat
import torch.distributions.dirichlet as diri

def MentorMixLoss(args,net, x_i, y_i, gamma_old, epoch):
    '''
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
    batch_size = x_i.shape[0]
    x_i, y_i = x_i.to(args.device), y_i.to(args.device)   
    with torch.no_grad():
        outputs_i = net(x_i) 
        loss = F.cross_entropy(outputs_i,y_i,reduction='none')                      
    loss_value, _ = loss.sort()
    l_p = loss_value[int(batch_size*args.gamma_p)]      
    gamma = gamma_old*args.ema + l_p*(1-args.ema)       
    v = (loss<gamma).float()                            
    P_v = cat.Categorical(F.softmax(v,dim=0))           
    indices_j = P_v.sample(y_i.shape)                   
    
    # Prepare Mixup
    x_j = x_i[indices_j]
    y_j = y_i[indices_j]
    label_i = F.one_hot(y_i, num_classes= args.num_classes)
    label_j = F.one_hot(y_j,num_classes= args.num_classes)
    
    # MIXUP
    Beta = diri.Dirichlet(torch.tensor([args.alpha for _ in range(2)])) # 9
    lambdas = Beta.sample(y_i.shape).to(args.device)
    lambdas_max = lambdas.max(dim=1)[0]                 
    lambdas = v*lambdas_max + (1-v)*(1-lambdas_max)     
    x_tilde = x_i * lambdas.view(lambdas.size(0),1,1,1) + x_j * (1-lambdas).view(lambdas.size(0),1,1,1)
    outputs_tilde = net(x_tilde)
    y_tilde = label_i * lambdas.view(lambdas.size(0),1) + label_j * (1-lambdas).view(lambdas.size(0),1)
    
    if args.second_reweight:
        with torch.no_grad():
            loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
            loss_value, _ = loss.sort()
            l_p = loss_value[int(batch_size*args.gamma_p)]      
            v = (loss<l_p).float()
        loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
        loss = loss*v
    else:
        loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
    return loss.mean(), gamma
    
