import torch.nn as nn
import math
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
            model.load_state_dict(torch.load(load_path),map_location=device)
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


    def resnet152(self,load_path=None,pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = torchvision.models.resnet152(pretrained=pretrained)
        model = self.change_output(model, load_path)
        return model

class Vgg(nn.Module):
    def __init__(self,output_num=3):
        super(Vgg, self).__init__()
        self.output_num=output_num

    def change_output(self,model,load_path=None,):
        numFit = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(numFit, self.output_num)
        if load_path != None:
            model.load_state_dict(torch.load(load_path),map_location=device)
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

