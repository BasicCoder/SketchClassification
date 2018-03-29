#!/usr/bin/env bash

DATA=dataset/images
echo "Create train.txt..."
rm -rf $DATA/train.txt
ls bike | sed "s:^:bike/:" | sed "s:$: 1:" >> train.txt
ls cat | sed "s:^:cat/:" | sed "s:$: 2:" >> train.txt