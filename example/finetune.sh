set -x

python2 finetune.py \
      --mode=train \
      --class_num=1000 \
      --data_dir=./data/imagenet \
      --num_gpus=4 \
      --batch_size=64 \
      --max_iter=600000 \
      --lr=0.1 \
      --lr_decay_step=150000 \
      --lr_decay_factor=0.1 \
      --weight_decay_rate=0.0001 \
      --optimizer='mom' \
      --batch_norm_elipson=1e-5 \
      --resnet_size=101 \
      --prof_interval=500 \
      --log_interval=5000 \
      --snapshot=5000 \
      --model_dir=./out/checkpoint/ \
      --log_dir=./out/log/ \
      --image_size=224 \
      --FixBlock2=True \
      --restore=True \
      --data_format='NCHW' \
      --pretrain_ckpt="./out/checkpoint/model.ckpt"
