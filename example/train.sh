
set -x 

# Parameters for the training
PYTHON=/usr/bin/python
DATASET_DIR=./data/ml-images
WITH_BBOX=FALSE
IMG_SIZE=224
CLASSNUM=11166
RESNET=101
MASK_THRES=0.7
NEG_SELECT=0.1
BATCHSIZE=1
SNAPSHOT=4400
BATCHNORM_DECAY=0.997
BATCHNORM_EPS=1e-5
LR=0.08
LR_DECAY_STEP=110000
LR_DECAY_FACTOR=0.1
WEIGHT_DECAY=0.0001
WARMUP=35200
LR_WARMUP=0.01
LR_WARMUP_DECAY_STEP=4400
LR_WARMUP_DECAY_FACTOR=1.297
MAXIER=440000
DATA_FORMAT='NCHW'
LOG_INTERVAL=100
LOG_DIR="./out/log"
if [[ ! -d $LOG_DIR ]]; then
  mkdir -p $LOG_DIR
fi

$PYTHON train.py \
	--data_dir=${DATASET_DIR} \
	--model_dir=./out/checkpoint/imagenet/resnet_model_${NODE_NUM}node_${GPU_NUM}gpu \
	--tmp_model_dir=./out/tmp/imagenet/resnet_model_${NODE_NUM}node_${GPU_NUM}gpu \
    --image_size=${IMG_SIZE} \
	--class_num=${CLASSNUM} \
	--resnet_size=${RESNET} \
    --mask_thres=${MASK_THRES} \
    --neg_select=${NEG_SELECT} \
	--batch_size=${BATCHSIZE} \
    --with_bbox=${WITH_BBOX} \
    --batch_norm_decay=${BATCHNORM_DECAY} \
    --batch_norm_epsilon=${BATCHNORM_EPS} \
	--lr=${LR} \
	--lr_decay_step=${LR_DECAY_STEP} \
	--lr_decay_factor=${LR_DECAY_FACTOR} \
    --weight_decay=${WEIGHT_DECAY} \
	--max_iter=${MAXIER} \
	--snapshot=${SNAPSHOT} \
	--warmup=${WARMUP} \
	--lr_warmup=${LR_WARMUP} \
	--lr_warmup_decay_step=${LR_WARMUP_DECAY_STEP} \
	--lr_warmup_decay_factor=${LR_WARMUP_DECAY_FACTOR} \
	--log_interval=${LOG_INTERVAL} \
	--data_format=${DATA_FORMAT} 2>&1 | tee ${LOG_DIR}/Node${NODE_NUM}_GPU${GPU_NUM}.log 
