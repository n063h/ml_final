import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class featureMapLoss(nn.Module):
    def __init__(self):
        super(featureMapLoss, self).__init__()

    def forward(self, pred_tensor, target_tensor,loss='yolo'):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,15) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,15)
        '''
        batch_size,s,_,cls_size= pred_tensor.shape
        #nonzero[k]=[i,j]表示第k个样本的i,j点是目标中心
        nonzero=torch.nonzero(target_tensor)[:,1:3].to(device)
        class_pred=torch.zeros(batch_size,cls_size).to(device)
        class_target=torch.zeros(batch_size,cls_size).to(device)
        for i in range(batch_size):
            j,k=nonzero[i]
            class_target[i]=target_tensor[i][j][k]
            class_pred[i]=pred_tensor[i][j][k]

        if loss=='yolo':
            class_loss = F.mse_loss(class_pred, class_target, size_average=True)
        elif loss=='softmax_yolo':
            class_pred = class_pred.softmax(dim=1)
            class_loss = F.mse_loss(class_pred, class_target, size_average=True)
        else:
            # 每个样本只留一维值,交叉熵自带softmax
            class_target = class_target.argmax(axis=1)
            class_loss = nn.CrossEntropyLoss(class_pred, class_target)

        return class_loss




if __name__ == '__main__':
    r1 = torch.randn(10, 6, 6, 3)
    r2 = torch.zeros(10, 6, 6, 3)
    for i in range(10):
        r2[i][2][3][1] = 1
    l=featureMapLoss()
    s1=l.forward(r1,r2)
