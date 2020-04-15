import torch
import torch.nn as nn
import torchvision.models as models

class FDNet(nn.Module):
    def __init__(self, out_features=2):
        super(FDNet, self).__init__()
        mnet = models.mobilenet_v2(pretrained=True)
        for param in mnet.parameters():
            param.requires_grad_(False)
            
        # Parameters of newly constructed modules have requires_grad=True by default
        in_features = mnet.classifier[1].in_features
        mnet.classifier = nn.Sequential(
                                nn.Dropout(p=0.2, inplace=False),
                                nn.Linear(in_features, out_features))
        self.mnet = mnet
        
    def forward(self, images):
        features = self.mnet(images)
        
        return features

class FDNet2(nn.Module):
    def __init__(self, out_features=2):
        super(FDNet2, self).__init__()
        mnet = models.mobilenet_v2(pretrained=True)
        for name, param in mnet.named_parameters():
            if("bn" not in name):
                param.requires_grad_(False)
            
        # Parameters of newly constructed modules have requires_grad=True by default
        in_features = mnet.classifier[1].in_features
        mnet.classifier = nn.Sequential(
                                nn.Dropout(p=0.2, inplace=False),
                                nn.Linear(in_features,500),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(500, out_features))
        self.mnet = mnet
        
    def forward(self, images):
        features = self.mnet(images)
        
        return features
