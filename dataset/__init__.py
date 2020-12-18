from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms






class dataset(Dataset):
    def __init__(self,txt_path,transform):
        self.transform = transform
        # 目标正方形
        self.img_size=600
        self.grid_num=6
        self.grid_size=int(self.img_size/self.grid_num)
        self.class_num=15
        with open(txt_path) as f:
            lines= f.readlines()
        self.labels =[]
        self.boxes=[]
        self.a_paths=[]
        self.b_paths = []
        self.real_size = []
        for line in lines:
            blocks=line.split(' ')
            self.a_paths.append(blocks[0])
            self.b_paths.append(blocks[1])
            self.labels.append(int(blocks[2]))
            img = np.array(Image.open(blocks[0]))
            # open 出来是 h,w,c
            real_size=[img.shape[1],img.shape[0]]
            self.real_size.append(real_size)
            xy=blocks[3].split('+')
            xmin = float(xy[0])
            xmax = float(xy[1])
            ymin=float(xy[2])
            ymax =float(xy[3])
            #x, y, w, h=self.convert_box(xmin,xmax,ymin,ymax,real_size)
            self.boxes.append([xmin,xmax,ymin,ymax])


    def get_img(self,img_path):
        img = np.array(Image.open(img_path))
        img = img.transpose((1, 0, 2))
        #image = np.resize(img, (self.img_size, self.img_size, 3))
        for t in self.transform:
            image = t(image)
        return image


    def __getitem__(self,index):
        a=self.get_img(self.a_paths[index])
        b = self.get_img(self.b_paths[index])
        box=self.boxes[index]
        label=self.labels[index]
        real_size=self.real_size[index]
        target = self.format(box, label,real_size)  # 6*6*15
        return a,b,target,real_size

    def convert_box(self,xmin,xmax,ymin,ymax,real_size):
        x=(xmin+xmax)/(2*real_size[0])
        y = (ymin + ymax) / (2 *real_size[1])
        #wh没有范围限制,没必要用真实比例
        w=(xmax-xmin)/self.grid_size
        h=(ymax-ymin)/self.grid_size
        return x,y,w,h

    def format(self,box,label,real_size):
        """
        :param box: torch.size([4]) or torch.size([feature_num])
        :param label: torch.size([1])  or torch.size([box_num_in_img])
        :return: torch.size([6,6,15]) or torch.size([grid_num,grid_num,class_num])
        """
        target = torch.zeros(self.grid_num,self.grid_num,self.class_num)

        box_grid_ind_x = int(np.ceil((box[0]+box[1])/(2*real_size[0])*self.grid_num)) - 1
        box_grid_ind_y = int(np.ceil((box[2]+box[3])/(2*real_size[1])*self.grid_num)) - 1
        target[box_grid_ind_x][box_grid_ind_y][label] = 1
        return target

    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    train_dataset = voc_dataset('./dataset/train_data.txt',transform = transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    test_dataset = voc_dataset('./dataset/test_data.txt', transform=transforms.ToTensor())
    test_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    for i, (a,b,l,r) in enumerate(train_loader):
        print(i)
    b = [123]
