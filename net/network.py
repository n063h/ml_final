import torch.nn as nn
import math,copy
import torch
import torch.nn.functional as F
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class ResNet(nn.Module):

    def __init__(self,output_num=6*6*15):
        super(ResNet, self).__init__()
        self.output_num=output_num

    def change_output(self,model,load_path=None,):
        numFit = model.fc.in_features
        model.fc = nn.Linear(numFit, self.output_num)
        if load_path != None:
            model.load_state_dict(torch.load(load_path,map_location=torch.device(device)))
        return model

    def resnet18(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet18(pretrained=pretrained).to(device)
        model=self.change_output(model,load_path)
        return model



    def resnet34(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet34(pretrained=pretrained).to(device)
        model = self.change_output(model, load_path)
        return model



    def resnet50(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet50(pretrained=pretrained).to(device)
        model = self.change_output(model, load_path)
        return model


    def resnet101(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet101(pretrained=pretrained).to(device)
        model = self.change_output(model, load_path)
        return model




class ResNetWithTwoInput(nn.Module):

    def __init__(self,model_name,output_num=3,pretrained=False,load_path=None):
        super(ResNetWithTwoInput, self).__init__()
        if model_name=="resnet18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        if model_name=="resnet34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        if model_name=="resnet101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        if load_path != None:
            model.load_state_dict(torch.load(load_path,map_location=torch.device(device)))
        model=model.to(device)
        self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool=model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        numFit = model.fc.in_features
        self.fc = nn.Linear(numFit, output_num).to(device)

    def feature_extractor(self,img):
        f=self.conv1(img)
        f=self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        return f

    def forward(self,a,b):
        feature_a=self.feature_extractor(a).to(device)
        feature_b = self.feature_extractor(b).to(device)
        feature=(feature_b-feature_a).to(device)
        feature=feature.view(a.shape[0],-1)
        output=self.fc(feature)
        return output.to(device)

class ResnetFeatureMap(nn.Module):

    def __init__(self,model_name,output_num=6*6*3,pretrained=True,load_path=None):
        super(ResnetFeatureMap, self).__init__()
        if model_name=="resnet18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        if model_name=="resnet34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        if model_name=="resnet101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        if load_path != None:
            model.load_state_dict(torch.load(load_path,map_location=torch.device(device)))
        model = model.to(device)
        self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool=model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        numFit = model.fc.in_features
        self.fc = nn.Linear(numFit, output_num).to(device)

    def feature_extractor(self,img):
        f=self.conv1(img)
        f=self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        return f

    def forward(self,b):
        feature_b = self.feature_extractor(b).to(device)
        feature_b=feature_b.view(b.shape[0],-1)
        output=self.fc(feature_b).to(device)
        output=output.view(-1,6,6,3)
        return output

class ResnetFPN(nn.Module):

    def __init__(self,model_name,output_num=3,pretrained=True):
        super(ResnetFPN, self).__init__()
        if model_name=="resnet18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        if model_name=="resnet34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        if model_name=="resnet101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        model=model.to(device)
        self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool=model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        numFit = model.fc.in_features
        #通道都是256,故全局平均池化后输出也都是256
        self.fc4 = nn.Linear(256, output_num).to(device)
        self.fc3 = nn.Linear(256, output_num).to(device)
        self.fc2 = nn.Linear(256, output_num).to(device)

        self.latlayer5 = nn.Conv2d(numFit, 256, 1, 1, 0)
        self.latlayer4 = nn.Conv2d(numFit//2, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(numFit//4, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(numFit//8, 256, 1, 1, 0)

        self.smooth4 = nn.Conv2d(256,256,3,1,1)
        self.smooth3 = nn.Conv2d(256,256,3,1,1)
        self.smooth2 = nn.Conv2d(256,256,3,1,1)


    def upsample_add(self,x,y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def pre_layer_forward(self,img):
        f=self.conv1(img)
        f=self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        return f


    def forward(self,b):
        c1=self.pre_layer_forward(b)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4) #512
        # c5 = self.avgpool(c4)
        p5=self.latlayer5(c5)
        p4=self.upsample_add(p5,self.latlayer4(c4))
        p3 = self.upsample_add(p4, self.latlayer3(c3))
        p2 = self.upsample_add(p3,self.latlayer2(c2))

        p4=self.smooth4(p4)
        p4=self.avgpool(p4)
        p3 = self.smooth4(p3)
        p3 = self.avgpool(p3)
        p2 = self.smooth4(p2)
        p2 = self.avgpool(p2)

        o4=self.fc4(p4.view(b.shape[0],-1))
        o3 = self.fc3(p3.view(b.shape[0],-1))
        o2 = self.fc2(p2.view(b.shape[0],-1))
        output=(o4+o3+o2)/3
        return output.to(device)