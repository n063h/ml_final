import dataset as d,torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from net.loss import *
from net.network import ResNet,Vgg
from torch.autograd import Variable
from util.evaluate import eval
import warnings,sys,os
warnings.filterwarnings("ignore", category=UserWarning)

epoch = 80
lr = 0.01
dict = {'1': 0, '2': 1, '14': 2}

def t(train_loader,test_loader,model,loss_func,optimizer,lr,model_name):
    best_eval_acc=0
    best_eval_epoch = 0
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
        print('\n\nStarting epoch %d / %d' % (e + 1, epoch), flush=True)
        print('Learning Rate for this epoch: {}'.format(lr), flush=True)

        for i,(b,target) in enumerate(train_loader):
            batch_size=len(target)
            b = Variable(b.to(device))
            target = Variable(target.to(device))
            b_pred = model(b).to(device)
            # pred=(a_pred-b_pred).view(-1,6,6,3)
            loss = loss_func(b_pred, target).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("%s :Epoch %d/%d| Step %d/%d Loss: %.2f"%(model_name,e+1,epoch,i+1,len(train_loader),loss), flush=True)
            epoch_loss = epoch_loss + loss
        print("%s :Epoch %d/%d| MeanLoss: %.2f" % (model_name,e + 1, epoch, epoch_loss/len(train_loader)), flush=True)
        #训练时只测试一次
        eval_acc=eval(model,loss_func,test_loader)
        if eval_acc > best_eval_acc:
            best_eval_acc, best_eval_epoch = eval_acc, e
        torch.save(model.state_dict(), './model/'+ model_name + '.pth')
    print("%s :Epoch %d has best eval_acc %.2f" % (model_name,best_eval_epoch+1,best_eval_acc), flush=True)


if __name__ == '__main__':
    train_transformer = [
        transforms.ToPILImage(),
        transforms.Resize((600,600)),
        transforms.RandomResizedCrop(600,scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    test_transformer = [
        transforms.ToPILImage(),
        transforms.Resize((600,600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_dataset = d.dataset('./dataset/train_data.txt', transform=train_transformer)
    count = [train_dataset.labels.count(0), train_dataset.labels.count(1), train_dataset.labels.count(2)]
    weight = torch.Tensor([count[j] for j in train_dataset.labels])/len(train_dataset)
    weight=1/weight
    sampler = WeightedRandomSampler(weight, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False,sampler=sampler)

    test_dataset = d.dataset('./dataset/test_data.txt', transform=test_transformer)
    count = [test_dataset.labels.count(0), test_dataset.labels.count(1), test_dataset.labels.count(2)]
    weight = torch.Tensor([count[j] for j in test_dataset.labels])/len(test_dataset)
    weight=1/weight
    sampler = WeightedRandomSampler(weight, len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,sampler=sampler)

    resnet = ResNet(3)
    vgg=Vgg(3)
    # loss_func = featureMapLoss()
    loss_func = nn.CrossEntropyLoss()
    vgg_loss_func=nn.NLLLoss()

    model=resnet.resnet18(pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet18WithoutPreTrain')

    model=resnet.resnet34(pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet34WithoutPreTrain')

    # model = vgg.vgg11(pretrained=True).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # t(train_loader, test_loader, model, vgg_loss_func, optimizer, lr, 'vgg11WithSqrtLoss')
    #
    # model = vgg.vgg19(pretrained=True).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # t(train_loader, test_loader, model, vgg_loss_func, optimizer, lr, 'vgg19WithSqrtLoss')
    #
    # model = vgg.vgg11_bn(pretrained=True).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # t(train_loader, test_loader, model, vgg_loss_func, optimizer, lr, 'vgg11_bnWithSqrtLoss')
    #
    # model = vgg.vgg19_bn(pretrained=True).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # t(train_loader, test_loader, model, vgg_loss_func, optimizer, lr, 'vgg19_bnWithSqrtLoss')

    model=resnet.resnet50(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet50WithPreTrain')

    model=resnet.resnet101(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet101WithPreTrain')

    model=resnet.resnet152(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet152WithPreTrain')

    model=resnet.resnet50(pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet50WithoutPreTrain')

    model=resnet.resnet101(pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet101WithoutPreTrain')

    model=resnet.resnet152(pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    t(train_loader,test_loader,model,loss_func,optimizer,lr,'resnet152WithoutPreTrain')

