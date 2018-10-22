#!/usr/bin/python
"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

"""Use the saved checkpoint to run single-label image classification"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2 as cv
import tensorflow as tf
from models import resnet as resnet
from flags import FLAGS

tf.app.flags.DEFINE_string("result", "label_pred.txt",
  "file name to save predictions")
tf.app.flags.DEFINE_string("images", "",
  "contains image path per line per image")
tf.app.flags.DEFINE_integer("top_k_pred", 5,
  "the top-k predictions")
tf.app.flags.DEFINE_string("dictionary", "",
  "the class dictionary of imagenet-2012")

def _load_dictionary(dict_file):
    dictionary = dict()
    with open(dict_file, 'r') as lines:
        for line in lines:
            sp = line.rstrip('\n').split('\t')
            idx, name = sp[0], sp[1]
            dictionary[idx] = name
    return dictionary

def preprocess(img):
    rawH = float(img.shape[0])
    rawW = float(img.shape[1])
    newH = 256.0
    newW = 256.0
    test_crop = 224.0 

    if rawH <= rawW:
        newW = (rawW/rawH) * newH
    else:
        newH = (rawH/rawW) * newW
    img = cv.resize(img, (int(newW), int(newH)))
    img = img[int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]
    img = ((img/255.0) - 0.5) * 2.0
    img = img[...,::-1]
    return img

# build model
images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
net = resnet.ResNet(images, is_training=False)
net.build_model()

logit = net.logit
prob = tf.nn.softmax(logit)
prob_topk, pred_topk = tf.nn.top_k(prob, k=FLAGS.top_k_pred)

# restore model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(FLAGS.visiable_gpu)
config.log_device_placement=False
sess = tf.Session(config=config)
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, FLAGS.model_dir)

dictionary = _load_dictionary(FLAGS.dictionary)

# inference
types= 'center'#'10crop'
orig_stdout = sys.stdout
f = open(FLAGS.result, 'w')
sys.stdout = f
with open(FLAGS.images, 'r') as lines:
  for line in lines:
    sp = line.rstrip('\n').split('\t')
    raw_img = cv.imread(sp[0])
    if type(raw_img)==None or raw_img.data==None :
      print("open pic " + sp[0] + " failed")
      continue
    #imgs = preprocess(raw_img, types)
    img = preprocess(raw_img)
    logits, probs_topk, preds_topk = sess.run([logit, prob_topk, pred_topk], 
        {images:np.expand_dims(img, axis=0)})
    probs_topk = np.squeeze(probs_topk)
    preds_topk = np.squeeze(preds_topk)
    names_topk = [dictionary[str(i)] for i in preds_topk]
    print('+++ the predictions of {} is:'.format(sp[0]))
    for i, pred in enumerate(preds_topk):
        print('%d %s: %.3f' % (pred, names_topk[i], probs_topk[i]))    
sys.stdout = orig_stdout
f.close()
