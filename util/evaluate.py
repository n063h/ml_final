import dataset as d,torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net.loss import *
from net.network import ResNet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_right(pred_tensor, target_tensor):
    '''
    pred_tensor: (tensor) size(batchsize,3 )
    target_tensor: (tensor) size(batchsize,)
    '''
    cnt=0
    batch_size, cls_size = pred_tensor.shape
    for i in range(batch_size):
        class_target= target_tensor[i]
        class_pred= pred_tensor[i]
        cnt+=1 if class_target==class_pred.argmax() else 0

    return cnt




def eval(model,loss_func,test_loader,once=False):

    total_cnt=0
    total=0
    for i, (b,target) in enumerate(test_loader):
        model.eval()
        batch_size = len(target)
        b =b.to(device)
        target = target.to(device)
        b_pred = model(b).to(device)
        loss = loss_func(b_pred, target).to(device)
        cnt=count_right(b_pred,target)
        total_cnt+=cnt
        total+=len(target)
        #print("Step %d/%d Loss: %.2f acc: %d/%d" % (i + 1, len(test_loader), loss,cnt,len(target)), flush=True)
        model.train()
        if once==True:return cnt/len(target)
    print("total acc: %d/%d" % (total_cnt,total), flush=True)

    return total_cnt/total



if __name__ == '__main__':
    test_dataset = d.dataset('./dataset/test_data.txt', transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    resnet = ResNet(6 * 6 * 15)
    model = resnet.resnet34(load_path='./model/resnet34_ep50.pth').to(device)
    loss_func = Loss()
    eval(model,loss_func,test_loader)