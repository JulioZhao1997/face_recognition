from .layer.l2norm import l2norm

import torchvision.models as models
import torch.nn as nn

def make_resnet(num_layers, num_embeddings, num_class):
    if num_layers == 18:
        return resnet18(num_embeddings, num_class)
    if num_layers == 50:
        return resnet50(num_embeddings, num_class)

class resnet18(nn.Module):
    def __init__(self, num_embeddings, num_class):
        super(resnet18, self).__init__()
        self.backbone = list(models.resnet18(pretrained = False).children())[:-2]
        self.bn1 = nn.BatchNorm2d(512)
        self.dp = nn.Dropout2d(p = 0.5)
        self.fc = nn.Linear(512 * 4 * 4, num_embeddings)
        self.bn2 = nn.BatchNorm1d(num_embeddings)
        self.gt_linear = l2norm(num_embeddings, num_class)
        self.fc2 = nn.Linear(num_embeddings, num_class)


    def forward(self, x):
        for module in self.backbone:
            x = module(x)
        #BN-Dropout-FC-BN
        x = self.bn1(x)
        x = self.dp(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn2(x)
        #x = self.gt_linear(x)
        x = self.fc2(x)
        return x

class resnet50(nn.Module):
    def __init__(self, num_embeddings, num_class):
        super(resnet50, self).__init__()
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained = False).children())[:-2])
        self.bn1 = nn.BatchNorm2d(2048)
        self.dp = nn.Dropout2d(p = 0.5)
        self.fc = nn.Linear(2048 * 4 * 4, num_embeddings)
        self.bn2 = nn.BatchNorm1d(num_embeddings)
        self.gt_linear = l2norm(num_embeddings, num_class)
        self.fc2 = nn.Linear(num_embeddings, num_class)


    def forward(self, x):
        x = self.backbone(x)
        #BN-Dropout-FC-BN
        x = self.bn1(x)
        x = self.dp(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.gt_linear(x)
        #x = self.fc2(x)
        return x


