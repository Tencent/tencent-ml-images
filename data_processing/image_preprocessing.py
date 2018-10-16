"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""


"""Read and preprocess image data.
  Image processing occurs on a single image at a time. Image are read and
  preprocessed in parallel across multiple threads. The resulting images
  are concatenated together to form a single batch for training or evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

def rotate_image(image, thread_id=0, scope=None):
  """Rotate image
  thread_id comes from {0, 1, 2, 3} uniformly, 
  we will apply rotation on 1/4 images of the trainning set
  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    rotated image
  """
  with tf.name_scope(name=scope, default_name='rotate_image'):
    angle = tf.random_uniform([], minval=-45*math.pi/180, maxval=45*math.pi/180, dtype=tf.float32, name="angle")
    distorted_image = tf.cond(
                        tf.equal(thread_id, tf.constant(0, dtype=tf.int32)),
                        lambda: tf.contrib.image.rotate(image, angle),
                        lambda: image
                      )
    return distorted_image

def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.
  thread_id comes from {0, 1, 2, 3} uniformly, 
  and we will apply color distortion when thresd_id = 0 or 1, 
  thus, only 1/2 images of the trainning set will be distorted
  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(name=scope, default_name='distort_color'):
    def color_ordering_0(image):
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      return image
    
    def color_ordering_1(image):
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      return image
    
    image = tf.cond(
              tf.equal(thread_id, tf.constant(0, dtype=tf.int32)),
              lambda: color_ordering_0(image),
              lambda: image
            )
    image = tf.cond(
              tf.equal(thread_id, tf.constant(1, dtype=tf.int32)),
              lambda: color_ordering_1(image),
              lambda: image
            )
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def distort_image(image, height, width, object_cover, area_cover, bbox, thread_id=0, scope=None):
  """Distort one image for training a network.
  Args:
    image: Tensor containing single image
    height: integer, image height
    width: integer, image width
    object_cover: float
    area_cover: float
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.name_scope(name=scope, default_name='distort_image'):
    # Crop the image to the specified bounding box.
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=object_cover,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[area_cover, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # Resize the image to net input shape
    distorted_image = tf.image.resize_images(distorted_image, [height, width])
    distorted_image.set_shape([height, width, 3])

    # Flip image, we just apply horizontal flip on 1/2 images of the trainning set
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Rotate image
    distorted_image = rotate_image(distorted_image, thread_id)

    # Distored image color
    distorted_image = distort_color(distorted_image, thread_id)
    return distorted_image

def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.
  Args:
    image: Tensor containing single image
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(values=[image, height, width], name=scope, default_name='eval_image'):
    # Crop the central region of the image with an area containing 80% of the original image.
    image = tf.image.central_crop(image, central_fraction=0.80)
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    return image

def image_preprocessing(image, output_height, output_width, object_cover, area_cover, train, bbox):
  """Decode and preprocess one image for evaluation or training.
  Args:
    image: Tensor containing single image
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    output_height: integer
    output_width: integer
    train: boolean
  Returns:
    3-D float Tensor containing an appropriately scaled image
  Raises:
    ValueError: if user does not provide bounding box
  """
  if train:
    thread_id = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32, name="thread_id")
    image = distort_image(image, output_height, output_width, object_cover, area_cover, bbox, thread_id)
  else:
    image = eval_image(image, output_height, output_width)

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  image = tf.reshape(image, shape=[output_height, output_width, 3])
  return image

def preprocess_image(image, output_height, output_width, object_cover, area_cover, is_training=False, bbox=None):
  return image_preprocessing(image, output_height, output_width, object_cover, area_cover, is_training, bbox)
