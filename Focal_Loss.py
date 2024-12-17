import torch
import  torch.nn as nn
import  torch.nn.functional as f
from Data_Utils import  load_data,scaling
from helper_utils import *
from evaluation_Utils import  *
from torch.utils.tensorboard import  SummaryWriter



class FocalLoss(nn.Module):
    def __init__(self,gamma=2,alpha=.25):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
        self.alpha=alpha

    def forward(self,pred_logits,target):
        bce_loss=f.binary_cross_entropy_with_logits(pred_logits,target,reduction='none')
        prob=pred_logits.sigmoid()
        alpha_t=torch.where(target==1,self.alpha,(1-self.alpha))
        pt=torch.where(target==1,prob,1-prob)
        loss=alpha_t*((1-pt) ** self.gamma)*bce_loss
        return  loss.sum()
class FraudNN(nn.Module):
    def __init__(self):
        super(FraudNN,self).__init__()
        self.hidden1 = nn.Linear(30, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.hidden2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.hidden3 = nn.Linear(128, 16, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=self.sigmoid(self.bn1(self.hidden1(x)))
        x=self.dropout(x)
        x=self.sigmoid(self.bn2(self.hidden2(x)))
        x=self.dropout(x)
        x=self.sigmoid(self.bn3(self.hidden3(x)))
        x=self.output(x)
        return x


if __name__=='__main__':
        config=load_Config('data_config.yml')
        torch.manual_seed(config['random_seed'])
        x_t,y_t,x_v,y_v=load_data(config)
        x_t,x_v=scaling(x_t,x_v)

        model=FraudNN()
        alpha=.7
        gamma=2
        lr=0.001
        criterion=(FocalLoss(alpha=alpha,gamma=gamma))
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
        x_Tensor=torch.tensor(x_t,dtype=torch.float32)
        y_Tensor=torch.tensor(y_t,dtype=torch.float32).reshape(-1,1)
        x_Vensor=torch.tensor(x_v,dtype=torch.float32)
        y_Vensor=torch.tensor(y_v,dtype=torch.float32).reshape(-1,1)

        batch_size=64
        num_epochs=20
        start_epoch=0
        run_name=f"modelwith{gamma}_as_gamma,{alpha}_as_alpha"
        writer=SummaryWriter(log_dir=f'{run_name}adamw_optimizer')

        #train
        for epoch in range(start_epoch,num_epochs):
            model.train()
            epoch_loss=0
            permutation = torch.randperm(x_Tensor.size()[0])
            x_T_shuffle=x_Tensor[permutation].clone()
            y_T_shuffle=y_Tensor[permutation].clone()
            for i in range(0,len(x_Tensor),batch_size):
                x_b=x_T_shuffle[i:i+batch_size]
                y_b=y_T_shuffle[i:i+batch_size]

                optimizer.zero_grad()
                output=model(x_b)
                loss=criterion(output,y_b)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()

            epoch_loss /=len(x_Tensor)/batch_size
            writer.add_scalar('Loss_Train',epoch_loss,epoch)
            for name,param in model.named_parameters():
                writer.add_histogram(name,param,epoch)
            print('Epoch[{}/{}],Loss: {:5f}'.format(epoch+1,num_epochs,epoch_loss))

            #end Training

            model.eval()
            with torch.no_grad():
                val_op=model(x_Vensor)
                val_loss=criterion(val_op,y_Vensor).item()
                y_val_prob=val_op.sigmoid().numpy()
                y_val_pred=(y_val_prob>.5).astype(int)
                report_val = classification_report(y_pred=y_val_pred,y_true=y_v , output_dict=True)
                auc_pr = eval_auc_pr(y_val_prob, y_v)
                eval_pr_curve(y_val_prob, y_v, title=f'FNN val_pr_curve')
                eval_cm(y_val_pred, y_v, f'FNN val_CM')
                writer.add_scalar('Val_Precision',report_val["1"]["precision"], epoch + 1)
                writer.add_scalar('Val_Recall', report_val["1"]["recall"], epoch + 1)
                writer.add_scalar('Val_F1', report_val["1"]["f1-score"], epoch + 1)
                writer.add_scalar('Val_AUC', auc_pr, epoch + 1)
                writer.add_scalar('loss_validation',val_loss,epoch)
            #checkpoints
            if(epoch+1)%15==0:
                save_checkpoint(model,epoch+1,title=run_name)

            writer.close()
