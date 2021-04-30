import torch
import torch.nn as nn

class MentorNet_arch(nn.Module):
    def __init__(self,MentorNet_type,label_embedding_size=2, epoch_embedding_size=5, num_label_embedding=2, num_fc_nodes=20):
        super(MentorNet_arch,self).__init__()
        self.MentorNet_type = MentorNet_type
        self.feat_dim = label_embedding_size + epoch_embedding_size + 2     # 2 notates the outputs of the 'bidirectional RNN'
        self.embedding_layer1 = torch.nn.Embedding(num_embeddings = num_label_embedding, embedding_dim=label_embedding_size,padding_idx=0)
        self.embedding_layer2 = torch.nn.Embedding(num_embeddings = 100, embedding_dim = epoch_embedding_size,padding_idx=0)
        self.RNN = torch.nn.RNN(2,1, bidirectional= True)

        self.fc1 = nn.Linear(self.feat_dim,num_fc_nodes)
        self.activation_1 = nn.Tanh()
        self.fc2 = nn.Linear(num_fc_nodes,1)
        self.activation_2 = nn.Sigmoid()

    def forward(self,label,total_epoch, epoch,loss,loss_diff):
        if self.MentorNet_type == 'threshold':
            return self.forward_th(label, epoch,loss,loss_diff)
        elif self.MentorNet_type == 'MentorNet':
            return self.forward_MentorNet(label,total_epoch, epoch,loss,loss_diff)

    def forward_th(self,label,epoch,loss,loss_diff):
        output = (loss_diff<0).float()        
        return output                   
        
    def forward_MentorNet(self, label, total_epoch, epoch, loss, loss_diff):
        '''
        inputs
            label           : [bsz,1]               (0,1)           label in this version only notifies whether it is noisy or not.
                                                                    when training MentorNet, labels are set to 0 or 1 according to the intended noise
                                                                    when using MentorNet, labels are all set to 0s.
            epoch           : [bsz,1]               (0~99)
            loss            : [bsz,1]   
            loss_diff       : [bsz,1]               (loss-loss_p)

        middle_outputs
            label_embedded  : [bsz,label_embedding_size]
            epoch_embedded  : [bsz,epoch_embedding_size]
            RNN_output      : [bsz,2]
            stacked_input   : [bsz,feat_dim]        (feat_dim = label_embedded.shape[1] + epoch_embedded.shape[1] + RNN_output.shape[1])
        outputs
            v               : [bsz,1]
        '''
        bsz = loss.shape[0]
        epoch = int((100*epoch/total_epoch))
        epoch = torch.LongTensor([epoch for i in range(len(loss))]).to(loss.device)
        loss_stacked = torch.stack((loss,loss_diff),dim=1)
        label_embedded = self.embedding_layer1(label)
        epoch_embedded = self.embedding_layer2(epoch)
        hidden = torch.zeros(2,1,1).to(loss.device)
        loss_stacked = loss_stacked.view((bsz,1,2))
        RNN_output ,_ = self.RNN(loss_stacked,hidden)             # hidden_state is ignored
        RNN_output = RNN_output.squeeze(dim=1)
        stacked_input = torch.cat((label_embedded,epoch_embedded,RNN_output),dim=1)

        output = self.activation_1(self.fc1(stacked_input))
        output = self.activation_2(self.fc2(output))
        output = output.squeeze(dim=1)
        return output

def MentorNet(args):
    return MentorNet_arch(args.MentorNet)