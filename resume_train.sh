#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python Train.py \
    --batch-size 128 \
    --resume ./runs/NetModel/checkpoint.pth.tar