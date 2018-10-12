#!/usr/bin/python
"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

"""Global Options
"""
tf.app.flags.DEFINE_string('mode', 'train',
   "run coder in tain or validation mode")
tf.app.flags.DEFINE_integer('max_to_keep', 200,
   "save checkpoint here")


"""Data Options
"""
tf.app.flags.DEFINE_string('data_dir', './data/train/',
   "Path to the data TFRecord of Example protos. Should save in train and val")
tf.app.flags.DEFINE_integer('batch_size', 512,
   "Number of images to process in a batch.")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
   "Number of preprocessing threads per tower. Please make this a multiple of 4")
tf.app.flags.DEFINE_integer('file_shuffle_buffer', 1500,
   "buffer size for file names")
tf.app.flags.DEFINE_integer('shuffle_buffer', 2048,
   "buffer size for samples")
tf.app.flags.DEFINE_boolean('with_bbox', True, 
   "whether use bbox in train set")

"""Model Options
"""
tf.app.flags.DEFINE_integer('class_num', 1000,
  "distinct class number")
tf.app.flags.DEFINE_integer('resnet_size', 101,
  "resnet block layer number [ 18, 34, 50, 101, 152, 200 ]")
tf.app.flags.DEFINE_string('data_format', 'channels_first',
  "data format for the input and output data [ channels_first | channels_last ]")
tf.app.flags.DEFINE_integer('image_size', 224,
   "default image size for model input layer")
tf.app.flags.DEFINE_integer('image_channels', 3,
   "default image channels for model input layer")
tf.app.flags.DEFINE_float('batch_norm_decay', 0.997,
   "use for batch normal moving avg")
tf.app.flags.DEFINE_float('batch_norm_epsilon', 1e-5,
   "use for batch normal layer, for avoid divide by zero")
tf.app.flags.DEFINE_float('mask_thres', 0.7,
   "mask thres for balance pos neg")
tf.app.flags.DEFINE_float('neg_select', 0.3,
   "how many class within only negtive samples in a batch select to learn")

"""Train Options
"""
tf.app.flags.DEFINE_boolean('restore', False,
   "whether to restore weights from pretrained checkpoint.")
tf.app.flags.DEFINE_integer('num_gpus', 1,
   "How many GPUs to use.")
tf.app.flags.DEFINE_string('optimizer','mom',
   "optimation algorthm")
tf.app.flags.DEFINE_float('opt_momentum', 0.9,
   "moment during learing")
tf.app.flags.DEFINE_float('lr', 0.1,
   "Initial learning rate.")
tf.app.flags.DEFINE_integer('lr_decay_step', 0,
   "Iterations after which learning rate decays.")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1,
   "Learning rate decay factor.")
tf.app.flags.DEFINE_float('weight_decay', 0.0001,
   "Tainable Weight l2 loss factor.")
tf.app.flags.DEFINE_integer('warmup', 0,
   "Steps when stop warmup, need when use distributed learning")
tf.app.flags.DEFINE_float('lr_warmup', 0.1,
   "Initial warmup learning rate, need when use distributed learning")
tf.app.flags.DEFINE_integer('lr_warmup_decay_step', 0,
   "Iterations after which learning rate decays, need when use distributed learning")
tf.app.flags.DEFINE_float('lr_warmup_decay_factor', 1.414,
   "Warmup learning rate decay factor, need when use distributed learning")
tf.app.flags.DEFINE_integer('max_iter', 1000000,
   "max iter number for stopping.-1 forever")
tf.app.flags.DEFINE_integer('test_interval', 0,
   "iterations interval for evluate model")
tf.app.flags.DEFINE_integer('test_iter', 0,
   "iterations for evluate model")
tf.app.flags.DEFINE_integer('prof_interval', 10,
   "iterations for print training time cost")
tf.app.flags.DEFINE_integer('log_interval', 0,
  "iterations for print summery log")
tf.app.flags.DEFINE_string('log_dir', './out/log/',
   "Directory where to write event logs")
tf.app.flags.DEFINE_string('model_dir', './out/checkpoint/',
   "path for saving learned tf model")
tf.app.flags.DEFINE_string('tmp_model_dir', './out/tmp/checkpoint/',
   "The directory where the temporary model will be stored")
tf.app.flags.DEFINE_integer('snapshot', 0,
   "Iteration for saving model snapshot")
tf.app.flags.DEFINE_integer('epoch_iter', 0,
   "Iteration for epoch ")
tf.app.flags.DEFINE_float('drop_rate', 0.5, 
   "DropOut rate")
tf.app.flags.DEFINE_integer('random_seed', 1234,
   "Random sedd for neigitive class selected")
tf.app.flags.DEFINE_string('pretrain_ckpt', '',
   'pretrain checkpoint file')
tf.app.flags.DEFINE_boolean('FixBlock2', False,
   'whether to fix the first two block, used for fintuning')


"""eval options
"""
tf.app.flags.DEFINE_integer('visiable_gpu', 0,
   "wihch gpu can use")
tf.app.flags.DEFINE_string('piclist', '',
   "eval picture list")
tf.app.flags.DEFINE_integer('interval', 32,
   "eval chekpoint interval")
tf.app.flags.DEFINE_integer('start', 0,
   "the start index of ckpts")
