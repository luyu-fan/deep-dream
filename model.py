'''
@File: model.py
@Author: Luyufan
@Date: 2020/3/31
@Desc: Some Common Module
'''
import torch,numpy as np
import torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

class DeepDreamNet(nn.Module):
    def __init__(self):
        super(DeepDreamNet,self).__init__()
        _extractor = models.resnet50(pretrained=True)
        self.conv1 = _extractor.conv1
        self.bn1 = _extractor.bn1
        self.relu = _extractor.relu
        self.maxpool = _extractor.maxpool
        self.layer1 = _extractor.layer1
        self.layer2 = _extractor.layer2
        self.layer3 = _extractor.layer3
        self.layer4 = _extractor.layer4

    def fixParams(self):
        for layer in self.modules():
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_feature = self.layer1(x)
        layer2_feature = self.layer2(layer1_feature)
        layer3_feature = self.layer3(layer2_feature)

        return layer3_feature

class GaussianSampler(nn.Module):
    """gaussian sampling filter"""
    def __init__(self,down = True):
        super(GaussianSampler,self).__init__()
        self.down = down
        gaussian_kernel = np.outer(np.float32([1, 4, 6, 4, 1]), np.float32([1, 4, 6, 4, 1]))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = torch.from_numpy(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        self.gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=False)

    def forward(self, img):
        img.requires_grad = False
        x1 = img[0, 0, :, :].unsqueeze(0).unsqueeze(0)
        x2 = img[0, 1, :, :].unsqueeze(0).unsqueeze(0)
        x3 = img[0, 2, :, :].unsqueeze(0).unsqueeze(0)
        if self.down:
            x1_out = F.conv2d(x1, weight=self.gaussian_kernel, bias=None, stride=2, padding=2)
            x2_out = F.conv2d(x2, weight=self.gaussian_kernel, bias=None, stride=2, padding=2)
            x3_out = F.conv2d(x3, weight=self.gaussian_kernel, bias=None, stride=2, padding=2)
            out = torch.cat((x1_out, x2_out, x3_out), dim=1)
            return out
        else:
            x1_out = F.conv_transpose2d(x1, weight=self.gaussian_kernel, bias=None, stride=2,padding=2)
            x2_out = F.conv_transpose2d(x2, weight=self.gaussian_kernel, bias=None, stride=2,padding=2)
            x3_out = F.conv_transpose2d(x3, weight=self.gaussian_kernel, bias=None, stride=2,padding=2)
            out = torch.cat((x1_out, x2_out, x3_out), dim=1)
            return out
