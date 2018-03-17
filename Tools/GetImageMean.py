import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from progressbar import Bar, ProgressBar
from scipy.misc import imread, imresize

def is_image(ext):
    ext = ext.lower()
    if ext == '.jpg':
        return True
    elif ext == '.png':
        return True
    elif ext == '.jpeg':
        return True
    elif ext == '.bmp':
        return True
    else:
        return False

def get_all_image_names(rootdir, image_names_list=[]):
    for file in os.listdir(rootdir):
        filepath = osp.join(rootdir, file)
        if osp.isdir(filepath):
            get_all_image_names(filepath, image_names_list)
        elif osp.isfile(filepath):
            ext = osp.splitext(filepath)[1]
            if is_image(ext):
                image_names_list.append(filepath)
    image_names_list = sorted(image_names_list)
    return image_names_list

def GetImageMean(rootdir, size=(256, 256)):
    R_channel = []
    G_channel = []
    B_channel = []
    image_names_list = get_all_image_names(rootdir)
    progress = ProgressBar(max_value= len(image_names_list))
    for i, name in enumerate(image_names_list):
        img = imread(name)
        img = imresize(img, size)
        if(img.shape[-1] == 3 or img.shape[-1] == 4):
            R_channel.append(img[:, :, 0])
            G_channel.append(img[:, :, 1])
            B_channel.append(img[:, :, 2])
        else:
            R_channel.append(img[:, :])

        progress.update(i)
    progress.finish()

    # num = len(image_names_list) * size[0] * size[1]

    if (img.shape[-1] == 3 or img.shape[-1] == 4):
        R_mean = np.mean(np.asarray(R_channel))
        G_mean = np.mean(np.asarray(G_channel))
        B_mean = np.mean(np.asarray(B_channel))

        R_std = np.std(np.asarray(R_channel))
        G_std = np.std(np.asarray(G_channel))
        B_std = np.std(np.asarray(B_channel))
        return R_mean, G_mean, B_mean, R_std, G_std, B_std
    else:
        R_mean = np.mean(np.asarray(R_channel))
        R_std = np.std(np.asarray(R_channel))
        return R_mean, R_std


if __name__ == "__main__":
    rootdir = r"/home/bc/Work/Database/TU-Berlin sketch dataset/png"
    mean = GetImageMean(rootdir)
    print(mean)