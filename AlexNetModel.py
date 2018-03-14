import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from alexnet import alexnet

class AlexNetModel(nn.Module):
    def __init__(self, num_classes=None):
        super(AlexNetModel, self).__init__()
        self.base = alexnet(pretrained=False)

        planes = 4096
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal(self.fc.weight, std=0.001)
            init.constant(self.fc.bias, 0.1)

    def forward(self, x):
        feat = self.base(x)

        if hasattr(self, 'fc'):
            logits = self.fc(feat)
            return feat, logits

        return feat