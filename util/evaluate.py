import dataset as d,torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net.loss import Loss
from net.network import ResNet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_right(pred_tensor, target_tensor):
    '''
    pred_tensor: (tensor) size(batchsize,S,S,15) [x,y,w,h,c]
    target_tensor: (tensor) size(batchsize,S,S,15)
    '''
    cnt=0
    batch_size, s, _, cls_size = pred_tensor.shape
    nonzero = torch.nonzero(target_tensor)[:, 1:3].to(device)
    for i in range(batch_size):
        j, k = nonzero[i]
        class_target= target_tensor[i][j][k]
        class_pred= pred_tensor[i][j][k]
        cnt+=1 if class_target.argmax()==class_pred.argmax() else 0

    return cnt




def eval(model,loss_func,test_loader,once=False):
    model.eval()
    total_cnt=0
    for i, (a, b, target, real_size) in enumerate(test_loader):
        a = a.to(device)
        b = b.to(device)
        target = target.to(device)
        a_pred = model(a).to(device)
        b_pred = model(b).to(device)
        pred = (a_pred - b_pred).view(-1, 6, 6, 15)
        loss = loss_func(pred, target).to(device)
        cnt=count_right(pred,target)
        total_cnt+=cnt
        print("Step %d/%d Loss: %.2f acc: %d/%d" % (i + 1, len(test_loader), loss,cnt,len(target)), flush=True)
        if once==True:return cnt/len(target)
    print("total acc: %d/%d" % (total_cnt,len(test_dataset)), flush=True)
    return total_cnt/len(test_dataset)



if __name__ == '__main__':
    test_dataset = d.dataset('./dataset/test_data.txt', transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    resnet = ResNet(6 * 6 * 15)
    model = resnet.resnet34(load_path='./model/resnet34_ep50.pth').to(device)
    loss_func = Loss()
    eval(model,loss_func,test_loader)