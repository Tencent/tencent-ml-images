#!/usr/bin/bash
set -x

PYTHON=/path/to/you/python
RESNET=101
DATA_FORMAT='NCHW'
GPUID=0
CKPT="./ckpts"

$PYTHON extract_feature.py \
  --resnet_size=$RESNET \
  --data_format=$DATA_FORMAT \
  --visiable_gpu=${GPUID} \
  --pretrain_ckpt=$CKPT \
  --result=test.txt \
  --images=imglist.txt 
	
