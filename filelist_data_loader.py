#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

from PIL import Image
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import json
import torch.utils.data
import torchvision.transforms as transforms


def default_image_loader(path):
    # return plt.imread(path)
    return Image.open(path).convert('RGB')

class SketchImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filelist_filename, mode="train", transform=None, loader=default_image_loader):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

