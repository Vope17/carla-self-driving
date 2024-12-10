import torch
import torch.nn as nn
import torchvision.models as models

class MyResNet18(nn.Module):
    def __init__(self, num_classes=1, weights=None):
        super(MyResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=weights)

        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)
