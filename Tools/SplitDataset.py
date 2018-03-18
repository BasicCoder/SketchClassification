#!/usr/bin/env python

import sys
import os
import os.path as osp
import shutil

def copyfile(srcfile, dstfile):
    if not osp.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname = osp.split(dstfile)
        if not osp.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile, dstfile)

def movetodir(record_file, Data_root, dataset_path):
    with open(record_file, 'r') as f:
        for line in f:
            src_img_path = osp.join(Data_root, line.rstrip())
            dst_img_path = osp.join(dataset_path, line.rstrip())
            copyfile(src_img_path, dst_img_path)

def main():
    Data_root = "/home/bc/Work/Database/TU-Berlin sketch dataset/png"
    Train_record_file = "/home/bc/Work/Database/TU-Berlin sketch dataset/png/train_list.txt"
    Val_record_file = "/home/bc/Work/Database/TU-Berlin sketch dataset/png/val_list.txt"


    Dataset_train_path = osp.join(Data_root, "../train_val/train")
    if not osp.exists(Dataset_train_path):
        os.makedirs(Dataset_train_path)
    
    Dataset_val_path = osp.join(Data_root, "../train_val/val")
    if not osp.exists(Dataset_val_path):
        os.makedirs(Dataset_val_path)

    movetodir(Train_record_file, Data_root, Dataset_train_path)
    movetodir(Val_record_file, Data_root, Dataset_val_path)


if __name__ == "__main__":
    main()
