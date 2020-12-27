import dataset as d,torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net.loss import *
from net.network import ResNet,ResnetFeatureMap,ResnetFPN,ResNetWithTwoInput
import warnings,traceback
import numpy as np
from dataset.json2txt import *
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(model,test_loader,train_type):

    total_pred=[]
    total_target = []
    if train_type == 'onlyB':
        for i, (b,label) in enumerate(test_loader):
            model.eval()
            b =b.to(device)
            target = label.to(device)
            b_pred = model(b).to(device)
            pred_res=b_pred.argmax(dim=1)
            total_pred.append(pred_res.cpu().numpy())
            total_target.append(target.cpu().numpy())
            model.train()
    elif train_type == 'yolo':
        for i, (a,b,box,label,target) in enumerate(test_loader):
            model.eval()
            batch_size,s,_,cls_size=target.shape
            b = b.to(device)
            target = target.to(device)
            b_pred = model(b).to(device)

            nonzero = torch.nonzero(target)[:, 1:3].to(device)
            class_pred = torch.zeros(batch_size, cls_size).to(device)
            class_target = torch.zeros(batch_size, cls_size).to(device)
            for i in range(batch_size):
                j, k = nonzero[i]
                class_target[i] = target[i][j][k]
                class_pred[i] = b_pred[i][j][k]

            class_pred = class_pred.argmax(dim=1)
            class_target = class_target.argmax(dim=1)

            total_pred.append(class_pred.cpu().numpy())
            total_target.append(class_target.cpu().numpy())
            model.train()
    elif train_type == 'boxAsB':
        for i, (a,b,box,label,target) in enumerate(test_loader):
            model.eval()
            b = box.to(device)
            target = label.to(device)
            b_pred = model(b).to(device)
            pred_res = b_pred.argmax(dim=1)
            total_pred.append(pred_res.cpu().numpy())
            total_target.append(target.cpu().numpy())
            model.train()
    elif train_type == 'FBSubFa':
        for i, (a,b,box,label,target) in enumerate(test_loader):
            model.eval()
            a = a.to(device)
            b = b.to(device)
            target = label.to(device)
            b_pred = model(a,b).to(device)
            pred_res = b_pred.argmax(dim=1)
            total_pred.append(pred_res.cpu().numpy())
            total_target.append(target.cpu().numpy())
            model.train()
    elif train_type == 'ABBox':
        for i, (a,b,box,label,target) in enumerate(test_loader):
            model.eval()
            b = b.to(device)
            target = label.to(device)
            b_pred = model(b).to(device)
            pred_res = b_pred.argmax(dim=1)
            total_pred.append(pred_res.cpu().numpy())
            total_target.append(target.cpu().numpy())
            model.train()

    total_target=np.concatenate(total_target)
    total_pred = np.concatenate(total_pred)
    acc=accuracy_score(total_target, total_pred)
    f1 = f1_score(total_target, total_pred, average='macro')
    p = precision_score(total_target, total_pred, average='macro')
    r = recall_score(total_target, total_pred, average='macro')
    print('test acc: ',acc,' f1: ', f1, ' precision: ', p, ' recall: ', r)
    return acc

def random_txt(data):
    random.shuffle(data)
    write('./dataset/train_data.txt', data[:700])
    write('./dataset/test_data.txt', data[700:])
    return


if __name__ == '__main__':
    test_transformer = [
        transforms.ToPILImage(),
        transforms.Resize((600,600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    data = read('./dataset/fabric_data_new')
    resnet = ResNet(3)

    ten_res = torch.zeros(10).to(device)
# ## BSubA
#     for i in range(10):
#         random_txt(data)
#         test_dataset = d.BSubADataset('./dataset/test_data.txt', transform=test_transformer)
#         test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#         model = resnet.resnet34().to(device)
#         model.load_state_dict(torch.load('./model/resnet34_BSubA.pth', map_location=torch.device(device)))
#         ten_res[i]=eval(model, test_loader, 'onlyB')
#     print('%s 10 times test acc: %.4f\n'%('resnet34_BSubA',ten_res.mean()))
#
# ## BAddBSubA
#     for i in range(10):
#         random_txt(data)
#         test_dataset = d.BAddBSubADataset('./dataset/test_data.txt', transform=test_transformer)
#         test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#         model = resnet.resnet34().to(device)
#         model.load_state_dict(torch.load('./model/resnet34_BAddBSubA.pth',map_location=torch.device(device)))
#         ten_res[i]=eval(model, test_loader, 'onlyB')
#     print('%s 10 times test acc: %.4f\n' % ('resnet34_BAddBSubA', ten_res.mean()))

## FPN onlyB
    # for i in range(10):
    #     random_txt(data)
    #     test_dataset = d.ABBoxDataset('./dataset/test_data.txt', transform=test_transformer)
    #     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    #     model = ResnetFPN(model_name='resnet34').to(device)
    #     model.load_state_dict(torch.load('./model/resnet34_FPNOnlyB.pth', map_location=torch.device(device)))
    #     ten_res[i]=eval(model, test_loader, 'ABBox')
    # print('%s 10 times test acc: %.4f\n' % ('resnet34_FPNOnlyB', ten_res.mean()))

    for i in range(10):
        random_txt(data)
        test_dataset = d.ABBoxDataset('./dataset/test_data.txt', transform=test_transformer)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        model = ResnetFPN(model_name='resnet101').to(device)
        model.load_state_dict(torch.load('./model/resnet101_FPNOnlyB.pth', map_location=torch.device(device)))
        ten_res[i]=eval(model, test_loader, 'ABBox')
    print('%s 10 times test acc: %.4f\n' % ('resnet101_FPNOnlyB', ten_res.mean()))

## FPN BSubA
    for i in range(10):
        random_txt(data)
        test_dataset = d.BSubADataset('./dataset/test_data.txt', transform=test_transformer)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        model = ResnetFPN(model_name='resnet34').to(device)
        model.load_state_dict(torch.load('./model/resnet34_FPNBSubA.pth', map_location=torch.device(device)))
        ten_res[i]=eval(model, test_loader, 'onlyB')
    print('%s 10 times test acc: %.4f\n' % ('resnet34_FPNBSubA', ten_res.mean()))

## FPN boxAsB
    for i in range(10):
        random_txt(data)
        test_dataset = d.ABBoxDataset('./dataset/test_data.txt', transform=test_transformer)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        model = ResnetFPN(model_name='resnet34').to(device)
        model.load_state_dict(torch.load('./model/resnet34_FPNBboxAsB.pth', map_location=torch.device(device)))
        ten_res[i]=eval(model, test_loader, 'boxAsB')
    print('%s 10 times test acc: %.4f\n' % ('resnet34_FPNBboxAsB', ten_res.mean()))