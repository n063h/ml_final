import dataset as d,torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from net.loss import *
import matplotlib.pyplot as plt
from net.network import ResNet,ResNetWithTwoInput
from torch.autograd import Variable
from util.evaluate import *
from dataset.json2txt import *
import warnings,sys,os
warnings.filterwarnings("ignore", category=UserWarning)

epoch = 80
lr = 0.01
dict = {'1': 0, '2': 1, '14': 2}

def t(train_loader,test_loader,model,loss_func,optimizer,lr,model_name,train_type="onlyB"):
    best_eval_acc=0
    best_eval_epoch = 0
    torch.autograd.set_detect_anomaly(True)
    train_loss_list=[]
    test_acc_list = []
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
        if train_type=='onlyB':
            for i, (a, b, box, label, target) in enumerate(train_loader):
                b = Variable(b.to(device))
                # 历史遗留原因,这里的target是label
                target = Variable(label.to(device))
                b_pred = model(b).to(device)
                loss = loss_func(b_pred, target).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("%s :Epoch %d/%d| Step %d/%d Loss: %.2f"%(model_name,e+1,epoch,i+1,len(train_loader),loss), flush=True)
                epoch_loss = epoch_loss + loss
        elif train_type=='yolo':
            for i, (a, b, box, label, target) in enumerate(train_loader):
                b = Variable(b.to(device))
                # 历史遗留原因,这里的target是label
                target = Variable(label.to(device))
                b_pred = model(b).to(device)
                loss = loss_func(b_pred, target).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("%s :Epoch %d/%d| Step %d/%d Loss: %.2f"%(model_name,e+1,epoch,i+1,len(train_loader),loss), flush=True)
                epoch_loss = epoch_loss + loss
        elif train_type=='BSubA':
            for i, (b,label) in enumerate(train_loader):
                b = Variable(b.to(device))
                # 历史遗留原因,这里的target是label
                target = Variable(label.to(device))
                b_pred = model(b).to(device)
                loss = loss_func(b_pred, target).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("%s :Epoch %d/%d| Step %d/%d Loss: %.2f"%(model_name,e+1,epoch,i+1,len(train_loader),loss), flush=True)
                epoch_loss = epoch_loss + loss
        elif train_type=='BAddBSubA':
            for i, (b,label) in enumerate(train_loader):
                b = Variable(b.to(device))
                # 历史遗留原因,这里的target是label
                target = Variable(label.to(device))
                b_pred = model(b).to(device)
                loss = loss_func(b_pred, target).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("%s :Epoch %d/%d| Step %d/%d Loss: %.2f"%(model_name,e+1,epoch,i+1,len(train_loader),loss), flush=True)
                epoch_loss = epoch_loss + loss
        elif train_type=='FBSubFa':
            for i, (a, b, box, label, target) in enumerate(train_loader):
                a = Variable(a.to(device))
                b = Variable(b.to(device))
                # 历史遗留原因,这里的target是label
                target = Variable(label.to(device))
                b_pred = model(a,b).to(device)
                loss = loss_func(b_pred, target).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("%s :Epoch %d/%d| Step %d/%d Loss: %.2f"%(model_name,e+1,epoch,i+1,len(train_loader),loss), flush=True)
                epoch_loss = epoch_loss + loss

        epoch_mean_loss=float(epoch_loss/train_loader.batch_size)
        print("%s :Epoch %d/%d| train MeanLoss: %.2f" % (model_name,e + 1, epoch, epoch_mean_loss), flush=True)
        eval_acc=eval(model,test_loader,train_type)
        if eval_acc > best_eval_acc:
            best_eval_acc, best_eval_epoch = eval_acc, e
            if e>10:
                torch.save(model.state_dict(), './model/'+ model_name + '.pth')
        train_loss_list.append(epoch_mean_loss)
        test_acc_list.append(eval_acc)
    print("%s :Epoch %d has best eval_acc %.2f" % (model_name,best_eval_epoch+1,best_eval_acc), flush=True)
    try:
        draw_loss(train_loss_list,model_name,'train_loss')
        draw_acc(test_acc_list, model_name, 'test_acc')
    except:
        print(model_name, 'test_acc draw error')





def draw_loss(y,model_name,type):
    x = range(epoch)
    plt.plot(x, y, '.-',label=model_name)
    plt_title = model_name
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel(type)
    plt.savefig('%s_%s.png'%(model_name,type))

def draw_acc(y,model_name,type):
    x = range(epoch)
    plt.plot(x, y, '.-',label=model_name)
    plt_title = model_name
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel(type)
    plt.savefig('%s_%s.png'%(model_name,type))

def get_sampler(dataset):
    count = [dataset.labels.count(0), dataset.labels.count(1), dataset.labels.count(2)]
    weight = torch.Tensor([count[j] for j in dataset.labels]) / len(dataset)
    weight = 1 / weight
    sampler = WeightedRandomSampler(weight, len(dataset))
    return sampler

def random_txt(data):
    random.shuffle(data)
    write('./dataset/train_data.txt', data[:700])
    write('./dataset/test_data.txt', data[700:])
    return

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

    data=read('./dataset/fabric_data_new')
    resnet = ResNet(3)

    yolo_loss_func = featureMapLoss()
    vgg_loss_func = nn.NLLLoss()
    cross_loss_func = nn.CrossEntropyLoss()






##B+B-A test :BAddBSubA
    try:
        random_txt(data)
        train_dataset, test_dataset = d.BAddBSubADataset('./dataset/train_data.txt',
                                                         transform=train_transformer), d.dataset(
            './dataset/test_data.txt', transform=test_transformer)
        train_sampler = get_sampler(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        model = resnet.resnet34(pretrained=True).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet34_BAddBSubA', 'BAddBSubA')

        model = resnet.resnet101(pretrained=True).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet101_BAddBSubA', 'BAddBSubA')
    except:
        pass

## feature_B-featureA
    try:
        random_txt(data)
        train_dataset, test_dataset = d.ABBoxDataset('./dataset/train_data.txt',
                                                         transform=train_transformer), d.dataset(
            './dataset/test_data.txt', transform=test_transformer)
        train_sampler = get_sampler(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        model=ResNetWithTwoInput(model_name='resnet18')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet18_FBSubFa', 'FBSubFa')

        model=ResNetWithTwoInput(model_name='resnet34')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet34_FBSubFa', 'FBSubFa')

        model=ResNetWithTwoInput(model_name='resnet101')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet101_FBSubFa', 'FBSubFa')
    except Exception:
        pass



    # os.system('/root/shutdown')


# ##backbone test :onlyB 借用ABBox的数据集
#     try:
#         random_txt(data)
#         train_dataset, test_dataset = d.ABBoxDataset('./dataset/train_data.txt',
#                                                      transform=train_transformer), d.ABBoxDataset(
#             './dataset/test_data.txt', transform=test_transformer)
#         train_sampler = get_sampler(train_dataset)
#
#         train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=train_sampler)
#         test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
#         model=resnet.resnet18(pretrained=True).to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#         t(train_loader,test_loader,model,cross_loss_func,optimizer,lr,'resnet18_onlyB','onlyB')
#
#
#         model=resnet.resnet34(pretrained=True).to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#         t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet34_onlyB','onlyB')
#
#         #更深网络参数太多,需要减小batch_size
#         train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False,sampler=train_sampler)
#         test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
#
#         model=resnet.resnet50(pretrained=True).to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#         t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet50_onlyB','onlyB')
#
#         model=resnet.resnet101(pretrained=True).to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#         t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet101_onlyB','onlyB')
#
#     except:
#         pass
#
# ##B-A test :BSubA
#     try:
#         random_txt(data)
#         train_dataset, test_dataset = d.BSubADataset('./dataset/train_data.txt', transform=train_transformer), d.dataset(
#             './dataset/test_data.txt', transform=test_transformer)
#         train_sampler = get_sampler(train_dataset)
#
#         train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=train_sampler)
#         test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
#         model = resnet.resnet18(pretrained=True).to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#         t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet18_BSubA','BSubA')
#
#         model = resnet.resnet34(pretrained=True).to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#         t(train_loader, test_loader, model, cross_loss_func, optimizer, lr, 'resnet34_BSubA','BSubA')
#     except:
#         pass
