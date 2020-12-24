from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch,cv2
from PIL import Image
import torchvision.transforms as transforms






class dataset(Dataset):
    def __init__(self,txt_path,transform):
        self.transform = transform
        # 目标正方形
        self.img_size=600
        self.grid_num=6
        self.grid_size=int(self.img_size/self.grid_num)
        self.class_num=3
        self.dict={'1':0,"2":1,"14":2}
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
            self.labels.append(self.dict[blocks[2]])
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


    def get_img(self,b):
        b = cv2.imread(b)
        img = cv2.resize(b, (self.img_size, self.img_size))
        for t in self.transform:
            img = t(img)
        return img

    def get_sub_img(self,a,b):
        a = cv2.imread(a)
        b=cv2.imread(b)
        a=cv2.resize(a,(self.img_size,self.img_size))
        b = cv2.resize(b, (self.img_size, self.img_size))
        img=b-a
        for t in self.transform:
            img = t(img)
        return img

    def get_sub_add_img(self,a,b):
        a = cv2.imread(a)
        b=cv2.imread(b)
        a=cv2.resize(a,(self.img_size,self.img_size))
        b = cv2.resize(b, (self.img_size, self.img_size))
        sub=b-a
        img=b+sub
        for t in self.transform:
            img = t(img)
        return img

    def __getitem__(self,index):
        #a=self.get_sub_img(self.a_paths[index],self.b_paths[index])
        #b=self.get_img(self.b_paths[index])
        b = self.get_img(self.b_paths[index])
        label=self.labels[index]
        return b,label

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
        :return: torch.size([6,6,3]) or torch.size([grid_num,grid_num,class_num])
        """
        target = torch.zeros(self.grid_num,self.grid_num,self.class_num)

        box_grid_ind_x = int(np.ceil((box[0]+box[1])/(2*real_size[0])*self.grid_num)) - 1
        box_grid_ind_y = int(np.ceil((box[2]+box[3])/(2*real_size[1])*self.grid_num)) - 1
        target[box_grid_ind_x][box_grid_ind_y][label] = 1
        return target

    def __len__(self):
        return len(self.labels)

class BSubADataset(Dataset):
    def __init__(self,txt_path,transform):
        self.transform = transform
        # 目标正方形
        self.img_size=600
        self.grid_num=6
        self.grid_size=int(self.img_size/self.grid_num)
        self.class_num=3
        self.dict={'1':0,"2":1,"14":2}
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
            self.labels.append(self.dict[blocks[2]])
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

    def get_b_sub_a_img(self,a,b):
        a = cv2.imread(a)
        b=cv2.imread(b)
        a=cv2.resize(a,(self.img_size,self.img_size))
        b = cv2.resize(b, (self.img_size, self.img_size))
        img=b-a
        for t in self.transform:
            img = t(img)
        return img

    def __getitem__(self,index):
        b = self.get_b_sub_a_img(self.a_paths[index], self.b_paths[index])
        label=self.labels[index]
        return b,label

    def __len__(self):
        return len(self.labels)

class BAddBSubADataset(Dataset):
    def __init__(self,txt_path,transform):
        self.transform = transform
        # 目标正方形
        self.img_size=600
        self.grid_num=6
        self.grid_size=int(self.img_size/self.grid_num)
        self.class_num=3
        self.dict={'1':0,"2":1,"14":2}
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
            self.labels.append(self.dict[blocks[2]])
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

    def get_b_add_b_sub_a_img(self,a,b):
        a = cv2.imread(a)
        b=cv2.imread(b)
        a=cv2.resize(a,(self.img_size,self.img_size))
        b = cv2.resize(b, (self.img_size, self.img_size))
        sub=b-a
        img=b+sub
        for t in self.transform:
            img = t(img)
        return img

    def __getitem__(self,index):
        b = self.get_b_add_b_sub_a_img(self.a_paths[index], self.b_paths[index])
        label=self.labels[index]
        return b,label

    def __len__(self):
        return len(self.labels)

class ABBoxDataset(Dataset):
    def __init__(self,txt_path,transform):
        self.transform = transform
        # 目标正方形
        self.img_size=600
        self.grid_num=6
        self.grid_size=int(self.img_size/self.grid_num)
        self.class_num=3
        self.dict={'1':0,"2":1,"14":2}
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
            self.labels.append(self.dict[blocks[2]])
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


    def get_img(self,b):
        b = cv2.imread(b)
        img = cv2.resize(b, (self.img_size, self.img_size))
        for t in self.transform:
            img = t(img)
        return img

    def __getitem__(self,index):
        a=self.get_img(self.a_paths[index])
        b=self.get_img(self.b_paths[index])
        box=self.extract_box(index)
        label=self.labels[index]
        target=self.get_target_matrix(index)
        return a,b,box,label,target



    def extract_box(self,index):
        b =cv2.imread(self.b_paths[index])
        xmin,xmax,ymin,ymax=self.boxes[index]
        #h,w,c
        img = b[int(ymin):int(ymax), int(xmin):int(xmax)]
        for t in self.transform:
            img = t(img)
        return img

    def get_target_matrix(self,index):
        target = torch.zeros(self.grid_num, self.grid_num, self.class_num)

        box=self.boxes[index]
        real_size=self.real_size[index]
        label = self.labels[index]

        box_grid_ind_x = int(np.ceil((box[0] + box[1]) / (2 * real_size[0]) * self.grid_num)) - 1
        box_grid_ind_y = int(np.ceil((box[2] + box[3]) / (2 * real_size[1]) * self.grid_num)) - 1
        target[box_grid_ind_x][box_grid_ind_y][label] = 1
        return target


    def __len__(self):
        return len(self.labels)

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

    train_dataset = ABBoxDataset('./dataset/train_data.txt',transform = train_transformer)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    for i, (a,b,box,label,target) in enumerate(train_loader):
        print(i)
    b = [123]
