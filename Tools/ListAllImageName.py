import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
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
                image_names_list.append(osp.join(osp.split(rootdir)[1], file))
    image_names_list = sorted(image_names_list)
    return image_names_list

def save_image_list(image_names_list, save_filename):
    f = open(save_filename, 'w')
    image_names_list = [line+'\n' for line in image_names_list]
    f.writelines(image_names_list)
    f.close()

if __name__ == "__main__":
    data_root=r"/home/bc/Work/Database/TU-Berlin sketch dataset/png"
    save_filename=r"./train.txt"
    image_names_list = get_all_image_names(data_root)
    save_image_list(image_names_list, save_filename)


