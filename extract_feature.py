#!/usr/bin/python
"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

"""Use pre-trained model extract image feature
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2 as cv
import tensorflow as tf
from models import resnet as resnet
from flags import FLAGS

tf.app.flags.DEFINE_string("result", "",
  "file name to save features")
tf.app.flags.DEFINE_string("images", "",
  "contains image path per line per image")

"""Crop Image To 224*224
 Args:
   img: an 3-D numpy array (H,W,C)
   type: crop method support [ center | 10crop ]
"""
def preprocess(img, type="center"):
  # resize image with smallest side to be 256
  rawH = float(img.shape[0])
  rawW = float(img.shape[1])
  newH = 256.0
  newW = 256.0
  if rawH <= rawW:
    newW = (rawW/rawH) * newH
  else:
    newH = (rawH/rawW) * newW
  img = cv.resize(img, (int(newW), int(newH)))
  imgs = None
  if type=='center':
    imgs = np.zeros((1, 224, 224, 3))
    imgs[0,...] = img[int((newH-224)/2):int((newH-224)/2)+224,
              int((newW-224)/2):int((newW-224)/2)+224]
  elif type=='10crop':
    imgs = np.zeros((10, 224, 224, 3))
    offset = [(0, 0),
              (0, int(newW-224)),
              (int(newH-224), 0),
              (int(newH-224), int(newW-224)),
              (int((newH-224)/2), int((newW-224)/2))]
    for i in range(0, 5):
      imgs[i,...] = img[offset[i][0]:offset[i][0]+224, 
                        offset[i][1]:offset[i][1]+224]
    img = cv.flip(img, 1)
    for i in range(0, 5):
      imgs[i+5,...] = img[offset[i][0]:offset[i][0]+224, 
                          offset[i][1]:offset[i][1]+224]
  else:
    raise ValueError("Type not support")
  imgs = ((imgs/255.0) - 0.5) * 2.0
  imgs = imgs[...,::-1]
  return imgs

# build model
images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
net = resnet.ResNet(images, is_training=False)
net.build_model()

logits = net.logit
feat = net.feat

# restore model
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(FLAGS.visiable_gpu)
config.log_device_placement=False
sess = tf.Session(config=config)

# load trained model
saver.restore(sess, FLAGS.pretrain_ckpt)

# inference on net
types='center'
ffeat = open(FLAGS.result, 'w')
with open(FLAGS.images, 'r') as lines:
  for line in lines:
    sp = line.rstrip('\n').split(' ')
    raw_img = cv.imread(sp[0])
    if type(raw_img)==None or raw_img.data==None :
      print("open pic " + sp[0] + " failed")
      continue
    imgs = preprocess(raw_img, types)
    feats = sess.run(feat, {images:imgs})
    feats = np.squeeze(feats[0])
    if types=='10crop':
      feats = np.mean(feats, axis=0)
    print('feature-length:{}, feature={}'.format(len(feats), feats))
    ffeat.write(sp[0] + "\t" + sp[1] + "\t" + " ".join([str(x) for x in list(feats)]) + '\n')
ffeat.close()
