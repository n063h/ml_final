import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()



    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,15) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,15)
        '''
        batch_size,s,_,cls_size= pred_tensor.shape
        nonzero=torch.nonzero(target_tensor)[:,1:3].to(device)
        class_pred=torch.zeros(batch_size,cls_size).to(device)
        class_target=torch.zeros(batch_size,cls_size).to(device)
        for i in range(batch_size):
            j,k=nonzero[i]
            class_target[i]=target_tensor[i][j][k]
            class_pred[i]=pred_tensor[i][j][k]

        class_pred= class_pred.softmax(dim=1)

        class_loss = F.mse_loss(class_pred, class_target, size_average=False)
        #class_loss = F.pairwise_distance(class_pred, class_target, p=2).sum()

        return class_loss/ batch_size

    def forward2(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,15) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,15)
        '''
        batch_size,s,_,cls_size= pred_tensor.shape
        class_pred = torch.zeros(batch_size, cls_size)
        class_target = torch.zeros(batch_size, cls_size)
        zero=torch.zeros(cls_size)
        for i in range(batch_size):
            for j in range(s):
                for k in range(s):
                    if (target_tensor[i][j][k]!=zero).sum()>0 :
                        class_target[i]=target_tensor[i][j][k]
                        class_pred[i]=pred_tensor[i][j][k]

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return ( class_loss) / batch_size

if __name__ == '__main__':
    r1 = torch.randn(10, 6, 6, 15)
    r2 = torch.zeros(10, 6, 6, 15)
    for i in range(10):
        r2[i][2][3][10] = 1
    l=Loss()
    s1=l.forward(r1,r2)
    s2=l.forward2(r1,r2)