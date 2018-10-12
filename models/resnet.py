"""ResNet model
Related papers:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../')
from flags import FLAGS
import tensorflow as tf

class ResNet(object):
  def __init__(self, images, is_training):
    """Net constructor
    Args:
      images: 4-D Tensor of images with Shape [batch_size, image_size, image_size, 3]
      is_training: bool, used in batch normalization
    Return:
      A wrapper For building model
    """
    self.is_training = is_training
    self.filters =  [256, 512, 1024, 2048] # feature map size for each stages
    self.strides =  [2,   2,   2,    2]    # conv strides for each stages's first block
    if FLAGS.resnet_size == 50:            # resnet size paramters
      self.stages = [3,   4,   6,    3]
    elif FLAGS.resnet_size == 101:
      self.stages = [3,   4,   23,   3]
    elif FLAGS.resnet_size == 152:
      self.stages = [3,   8,   36,   3]
    else:
      raise ValueError('resnet_size %d Not implement:' % FLAGS.resnet_size)
    self.data_format = FLAGS.data_format
    self.num_classes = FLAGS.class_num
    self.images = images
    if self.data_format == "NCHW":  
      self.images = tf.transpose(images, [0, 3, 1, 2])


  def build_model(self):
    # Initial net
    with tf.variable_scope('init'):
      x = self.images
      x = self._pre_padding_conv('init_conv', x, 7, 64, 2)

    # 4 stages 
    for i in range(0, len(self.stages)):
      with tf.variable_scope('stages_%d_block_%d' % (i,0)):
        x = self._bottleneck_residual(
              x, 
              self.filters[i], 
              self.strides[i], 
              'conv',
              self.is_training)
      for j in range(1, self.stages[i]):
        with tf.variable_scope('stages_%d_block_%d' % (i,j)):
          x = self._bottleneck_residual(
                x, 
                self.filters[i], 
                1,
                'identity', 
                self.is_training)
    
    # class wise avg pool
    with tf.variable_scope('global_pool'):
      x = self._batch_norm('bn', x, self.is_training)
      x = self._relu(x)
      x = self._global_avg_pool(x)
    
    # extract features
    self.feat=x
    
    # logits
    with tf.variable_scope("logits"):
      self.logit = self._fully_connected(x, out_dim=self.num_classes)

    return self.logit

  def _bottleneck_residual(self, x, out_channel, strides, _type, is_training):
    """Residual Block
     Args:
       x : A 4-D tensor
       out_channels : out feature map size of residual block
       strides : conv strides of block
       _type: short cut type, 'conv' or 'identity'
       is_training :  A Boolean for whether the model is in training or inference mdoel
    """
    # short cut
    orig_x = x
    if _type=='conv':
      orig_x = self._batch_norm('conv1_b1_bn', orig_x, is_training)
      orig_x = self._relu(orig_x)
      orig_x = self._pre_padding_conv('conv1_b1', orig_x, 1, out_channel, strides)

    # bottleneck_residual_block
    x = self._batch_norm('conv1_b2_bn', x, is_training)
    x = self._relu(x)
    x = self._pre_padding_conv('conv1_b2', x, 1, out_channel/4, 1)
    x = self._batch_norm('conv2_b2_bn', x, is_training)
    x = self._relu(x)
    x = self._pre_padding_conv('conv2_b2', x, 3, out_channel/4, strides)
    x = self._batch_norm('conv3_b2_bn', x, is_training)
    x = self._relu(x)
    x = self._pre_padding_conv('conv3_b2', x, 1, out_channel, 1)

    # sum
    return x + orig_x

  def _batch_norm(self, name, x, is_training=True):
    """Batch normalization.
     Considering the performance, we use batch_normalization in contrib/layers/python/layers/layers.py
     instead of tf.nn.batch_normalization and set fused=True
     Args:
       x: input tensor
       is_training: Whether to return the output in training mode or in inference mode, use the argment
                    in finetune
    """
    with tf.variable_scope(name):
      return tf.layers.batch_normalization(
             inputs=x,
             axis=1 if self.data_format == 'NCHW' else 3,
             momentum = FLAGS.batch_norm_decay,
             epsilon = FLAGS.batch_norm_epsilon,
             center=True,
             scale=True,
             training=is_training,
             fused=True
             )

  def _pre_padding(self, x, kernel_size):
    """Padding Based On Kernel_size"""
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if self.data_format == 'NCHW':
      x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return x 

  def _pre_padding_conv(self, name, x, kernel_size, out_channels, strides, bias=False):
    """Convolution
    As the way of padding in conv is depended on input size and kernel size, which is very different with caffe
    So we will do pre-padding to Align the padding operation.
     Args:
       x : A 4-D tensor 
       kernel_size : size of kernel, here we just use square conv kernel
       out_channels : out feature map size
       strides : conv stride
       bias : bias may always be false
    """
    if strides > 1:
      x = self._pre_padding(x, kernel_size)
    with tf.variable_scope(name):
      return tf.layers.conv2d(
             inputs = x,
             filters = out_channels,
             kernel_size=kernel_size,
             strides=strides,
             padding=('SAME' if strides == 1 else 'VALID'), 
             use_bias=bias,
             kernel_initializer=tf.variance_scaling_initializer(),
             data_format= 'channels_first' if self.data_format == 'NCHW' else 'channels_last')

  def _relu(self, x, leakiness=0.0):
    """
    Relu. With optical leakiness support
    Note: if leakiness set zero, we will use tf.nn.relu for concern about performance
     Args:
       x : A 4-D tensor
       leakiness : slope when x < 0
    """
    if leakiness==0.0:
      return tf.nn.relu(x)
    else:
      return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _global_avg_pool(self, x):
    """
    Global Average Pool, for concern about performance we use tf.reduce_mean 
    instead of tf.layers.average_pooling2d
     Args:
       x: 4-D Tensor
    """
    assert x.get_shape().ndims == 4
    axes = [2, 3] if self.data_format == 'NCHW' else [1, 2]
    return tf.reduce_mean(x, axes, keep_dims=True)

  def _fully_connected(self, x, out_dim):
    """
    As tf.layers.dense need 2-D tensor, reshape it first
    Args:
      x : 4-D Tensor
      out_dim : dimensionality of the output space.
    """
    assert x.get_shape().ndims == 4
    axes = 1 if self.data_format == 'NCHW' else -1
    x = tf.reshape(x, shape=[-1, x.get_shape()[axes]])
    return tf.layers.dense(x, units = out_dim)
