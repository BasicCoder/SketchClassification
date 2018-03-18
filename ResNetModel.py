import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet18, resnet50

class ResNetModel(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNetModel, self).__init__()
        self.base = resnet18(pretrained=True)

        planes = 512

        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal(self.fc.weight, std=0.001)
            init.constant(self.fc.bias, 0.1)

    def forward(self, x):
        # shape [N, C, H, W]
        feat = self.base(x)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        global_feat = global_feat.view(global_feat.size(0), -1)

        if hasattr(self, 'fc'):
            logits = self.fc(global_feat)
            return global_feat, logits

        # return global_feat, local_feat
        return global_feat




