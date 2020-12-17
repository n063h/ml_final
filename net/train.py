import dataset as d,util,torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from net.loss import Loss
from net.resnet import *
import torchvision
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def t():
    epoch = 50
    lr = 0.01

    train_dataset = d.dataset('./dataset/train_data.txt', transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataset = d.dataset('./dataset/test_data.txt', transform=[transforms.ToTensor()])
    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


    #model = ResNet()
    model=torchvision.models.resnet34(pretrained=True)
    numFit = model.fc.in_features
    model.fc = nn.Linear(numFit, 6*6*15)
    loss_func = Loss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    torch.autograd.set_detect_anomaly(True)
    for e in range(epoch):
        model.train()
        epoch_loss = torch.Tensor([0]).to(device)
        if epoch == 20:
            lr = 0.001
        if epoch == 40:
            lr = 0.0001
        if epoch in [20,30,40]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('\n\nStarting epoch %d / %d' % (e + 1, epoch))
        print('Learning Rate for this epoch: {}'.format(lr))

        for i,(a,b,target,real_size) in enumerate(train_loader):
            a = Variable(a.to(device))
            b = Variable(b.to(device))
            target = Variable(target.to(device))
            a_pred = model(a).to(device)
            b_pred = model(b).to(device)
            pred=(a_pred-b_pred).view(-1,6,6,15)
            loss = loss_func(pred, target).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch %d/%d| Step %d/%d Loss: %.2f"%(e+1,epoch,i+1,len(train_loader),loss))
            epoch_loss = epoch_loss + loss
        print("Epoch %d/%d| MeanLoss: %.2f" % (e + 1, epoch, epoch_loss/len(train_loader)))
        if (e+1)%10==0:
            torch.save(model.state_dict(), './model/epoch"+str(e+1)+".pth')
            # compute_val_map(model)


if __name__ == '__main__':
    t()