#!/usr/bin/python

"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from __future__ import print_function

import sys
import os
import time
from datetime import datetime
import tensorflow as tf

from models import resnet as resnet
from data_processing import dataset as file_db
from data_processing import image_preprocessing as image_preprocess
from flags import FLAGS

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def assign_weights_from_cp(cpk_path, sess, scope):
    '''
    restore from ckpt
    '''
    reader = tf.train.NewCheckpointReader(cpk_path)
    temp = reader.debug_string().decode('utf8')
    lines = temp.split("\n")
    i = 0
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        with tf.variable_scope(scope, reuse=True):
            try:
                if key.find(r'global_step')!=-1 or key.find(r'Momentum')!=-1 or key.find(r'logits')!=-1:
                    print("do not need restore from ckpt key:%s" % key)
                    continue
                var = tf.get_variable(key)
                sess.run(var.assign(reader.get_tensor(key)))
                print("restore from ckpt key:%s" % key)
            except ValueError:
                print("can not restore from ckpt key:%s" % key)

def record_parser_fn(value, is_training):
    """Parse an image record from `value`."""
    keys_to_features = {
          'width': tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
          'height': tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
          'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          'label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'name': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    image = tf.image.decode_image(tf.reshape(parsed['image'], shape=[]),
      FLAGS.image_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    bbox = tf.concat(axis=0, values=[ [[]], [[]], [[]], [[]] ])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
    image = image_preprocess.preprocess_image(
        image=image,
        output_height=FLAGS.image_size,
        output_width=FLAGS.image_size,
        object_cover=0.0, 
        area_cover=0.05,
        is_training=is_training,
        bbox=bbox)

    label = tf.cast(tf.reshape(parsed['label'], shape=[]),dtype=tf.int32)
    label = tf.one_hot(label, FLAGS.class_num)    

    return image, label

def tower_model(images, labels):
    model = resnet.ResNet(images, is_training=(FLAGS.mode == tf.estimator.ModeKeys.TRAIN))
    model.build_model()

    # waring: Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=model.logit, onehot_labels=labels)
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We Add the batch norm variables into L2 normal because
    # in large scale data training this will improve the generalization power of model.
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                'bn' not in v.name]) + 0.1 * FLAGS.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bn' in v.name])

    return model, loss

def average_gradients(tower_grads):
    """ Calculate the average gradient of shared variables across all towers. """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for grad, var in grad_and_vars:
            grads.append(tf.expand_dims(grad, 0))
        # Average over the 'tower' dimension.
        gradient = tf.reduce_mean(tf.concat(axis=0, values=grads), 0)
        v = grad_and_vars[0][1]
        grad_and_var = (gradient, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(train_dataset, is_training=True):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # set global_step and learning_rate
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(
            FLAGS.lr,
            global_step,
            FLAGS.lr_decay_step,
            FLAGS.lr_decay_factor,
            staircase=True)

        # optimizer, default is momentum
        if FLAGS.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif FLAGS.optimizer == "mom":
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=FLAGS.opt_momentum)
        else:
            raise ValueError("Do not support optimizer '%s'" % FLAGS.optimizer)

        # Get images and labels for training and split the batch across GPUs.
        """Input function which provides batches for train or eval."""
        worker_num = 1
        num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
        batch_size = FLAGS.batch_size * FLAGS.num_gpus
        print('batch_size={}'.format(batch_size))
        dataset = tf.data.Dataset.from_tensor_slices(train_dataset.data_files())
        dataset = dataset.shuffle(buffer_size=FLAGS.file_shuffle_buffer, seed=worker_num)
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda value: record_parser_fn(value, is_training), 
                    num_parallel_calls=num_preprocess_threads)
        dataset = dataset.prefetch(batch_size)

        if is_training:
          # When choosing shuffle buffer sizes, larger sizes result in better
          # randomness, while smaller sizes have better performance.
          # dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER, seed=worker_id)
          dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer)
          # We call repeat after shuffling, rather than before, to prevent separate
          # epochs from blending together.
          dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        images_splits = tf.split(images, FLAGS.num_gpus, 0)
        labels_splits = tf.split(labels, FLAGS.num_gpus, 0)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Calculate the gradients for each model tower
        Loss = None
        tower_grads = []

        #building graphs
        with tf.variable_scope(tf.get_variable_scope()):
            print("Building graph ...", file=sys.stderr)
            for i in xrange(FLAGS.num_gpus):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("%s_%d" % ("tower", i)) as scope:
                        # Build graph
                        model, Loss = tower_model(images_splits[i], labels_splits[i])
                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Get finetune variables
                        finetune_vars = []
                        if FLAGS.FixBlock2:
                            finetune_vars = [v for v in tf.trainable_variables()
                                             if v.name.find(r"stages_2") != -1 or
                                             v.name.find(r"stages_3") != -1 or
                                             v.name.find(r"global_pool") != -1 or
                                             v.name.find(r"logits") != -1]
                        else:
                            finetune_vars = tf.trainable_variables()

                        # Only the summaries from the final tower are retained
                        summary = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
                        grads = optimizer.compute_gradients(Loss, var_list=finetune_vars)
                        tower_grads.append(grads)

                        print("Build Graph (%s/%s)" % (i+1, FLAGS.num_gpus), file=sys.stderr)
        summaries.append(summary)
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Build train op,
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        ##############
        batchnorm_updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        # Build Session : may conf carefully
        sess = tf.Session(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=False))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
        summary_op = tf.summary.merge(summaries)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Initialize Model
        if FLAGS.restore:
            print("Restoring checkpoint from %s" % FLAGS.pretrain_ckpt, file=sys.stderr)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            #restore from existing ckpts
            assign_weights_from_cp(FLAGS.pretrain_ckpt, sess, tf.get_variable_scope())

        else:
            print("Run global_variables_initializer ..", file=sys.stderr)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        sys.stdout.write("---------------Trainging Begin---------------\n")

        batch_duration = 0.0
        # Initial queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Start train iter
        step = sess.run(global_step)
        i=0
        while i <= FLAGS.max_iter:
            # profile log
            if i > 0 and i % FLAGS.prof_interval == 0:
                print("%s: step %d, iteration %d, %.2f sec/batch" %
                      (datetime.now(), step, i, batch_duration))

            # log
            if i > 0 and i % FLAGS.log_interval == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, i)

            # checkpoint
            if i > 0 and i % FLAGS.snapshot == 0:
                if not os.path.exists(FLAGS.model_dir):
                    os.mkdir(FLAGS.model_dir)
                ckpt_path = os.path.join(FLAGS.model_dir, "resnet.ckpt")
                saver.save(sess, ckpt_path, global_step=global_step)

            # train
            batch_start = time.time()
            _, step, loss = sess.run([train_op, global_step, Loss])
            batch_duration = time.time() - batch_start
            i = i + 1
            print("%s: step %d, iteration %d, train loss %.2f " % (datetime.now(), step, i, loss))
        coord.request_stop()

def main(_):
    train_dataset = file_db.Dataset(os.path.join(FLAGS.data_dir, 'train'))
    train(train_dataset, is_training=(FLAGS.mode == tf.estimator.ModeKeys.TRAIN))

if __name__ == "__main__":
    tf.app.run()
