import torch.nn as nn
import math,copy
import torch
import torch.nn.functional as F
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

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
        model = torchvision.models.resnet18(pretrained=pretrained)
        model=self.change_output(model,load_path)
        return model



    def resnet34(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet34(pretrained=pretrained)
        model = self.change_output(model, load_path)
        return model



    def resnet50(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet50(pretrained=pretrained)
        model = self.change_output(model, load_path)
        return model


    def resnet101(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet101(pretrained=pretrained)
        model = self.change_output(model, load_path)
        return model

class ResNetFeatureMap(nn.Module):

    def __init__(self,output_num=6*6*15):
        super(ResNetFeatureMap, self).__init__()
        self.output_num=output_num

    def extract_feature_map(self,model,load_path=None,):
        model.fc = nn.Sequential()
        if load_path != None:
            model.load_state_dict(torch.load(load_path,map_location=torch.device(device)))
        return model

    def resnet18(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet18(pretrained=pretrained)
        model=self.extract_feature_map(model,load_path)
        return model



    def resnet34(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet34(pretrained=pretrained)
        model = self.extract_feature_map(model, load_path)
        return model



    def resnet50(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet50(pretrained=pretrained)
        model = self.extract_feature_map(model, load_path)
        return model


    def resnet101(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet101(pretrained=pretrained)
        model = self.extract_feature_map(model, load_path)
        return model


class Vgg(nn.Module):
    def __init__(self,output_num=3):
        super(Vgg, self).__init__()
        self.output_num=output_num

    def change_output(self,model,load_path=None,):
        numFit = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(numFit, self.output_num)
        model.classifier._modules['7'] = torch.nn.LogSoftmax(dim=1)
        if load_path != None:
            model.load_state_dict(torch.load(load_path,map_location=torch.device(device)))
        return model

    def vgg11(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.vgg11(pretrained=pretrained)
        model=self.change_output(model,load_path)
        return model

    def vgg19(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.vgg19(pretrained=pretrained)
        model=self.change_output(model,load_path)
        return model

    def vgg11_bn(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.vgg11_bn(pretrained=pretrained)
        model=self.change_output(model,load_path)
        return model

    def vgg19_bn(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.vgg19_bn(pretrained=pretrained)
        model=self.change_output(model,load_path)
        return model



class ResNet18TwoInput(nn.Module):

    def __init__(self,output_num=3,pretrained=False):
        super(ResNet18TwoInput, self).__init__()

        model = torchvision.models.resnet18(pretrained=pretrained)
        self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool=model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        numFit = model.fc.in_features
        self.fc = nn.Linear(numFit, output_num)

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
        feature_a=self.feature_extractor(a)
        feature_b = self.feature_extractor(b)
        feature=feature_b-feature_a
        feature=feature.view(a.shape[0],-1)
        output=self.fc(feature)
        return output

class ResNet34TwoInput(nn.Module):

    def __init__(self,output_num=3,pretrained=False):
        super(ResNet34TwoInput, self).__init__()

        model = torchvision.models.resnet34(pretrained=pretrained)
        self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool=model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        numFit = model.fc.in_features
        self.fc = nn.Linear(numFit, output_num)

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
        feature_a=self.feature_extractor(a)
        feature_b = self.feature_extractor(b)
        feature=feature_b-feature_a
        feature=feature.view(a.shape[0],-1)
        output=self.fc(feature)
        return output
