#!/usr/bin/python
import sys
import os
import tensorflow as tf
import numpy as np
import imghdr
import threading
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-idx","--indexs", type=str, default="", help="dirs contains train index files")
parser.add_argument("-tfs", "--tfrecords", type=str, default="", help="dirs contains train tfrecords")
parser.add_argument("-im", "--images", type=str, default="", help="the path contains the raw images")
parser.add_argument("-cls", "--num_class", type=int, default=0, help="class label number")
parser.add_argument("-one", "--one_hot", type=bool, default=True, help="indicates the format of label fields in tfrecords")
parser.add_argument("-sidx", "--start_index", type=int, default=0, help="the start number of train tfrecord files")
args = parser.parse_args()

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()
    
    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})
  
  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  return (imghdr.what(filename)=='png')

def _is_jpeg(filename):
  return (imghdr.what(filename)=='jpeg')

def _process_image(filename, coder):
  """Process a single image file."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()
    if not _is_jpeg(filename):
      if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
      else:
        try:
            image = coder.decode_jpeg(image_data)
            assert len(image.shape) == 3
            height = image.shape[0]
            width = image.shape[1]
            assert image.shape[2] == 3
            return image_data, height, width
        except:
            print('Cannot converted type %s' % imghdr.what(filename))
            return [], 0, 0

    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _save_one(train_txt, tfrecord_name, label_num, one_hot):
  writer = tf.python_io.TFRecordWriter(tfrecord_name)
  with tf.Session() as  sess:
    coder = ImageCoder()
    with open(train_txt, 'r') as lines:
      for line in lines:
        sp = line.rstrip("\n").split()
        imgf = os.path.join(args.images, sp[0])
        print(imgf)
        img, height, width = _process_image(imgf, coder)
        if height*width==0:
          continue

        if one_hot:
          label = np.zeros([label_num,], dtype=np.float32)
          for i in range(1, len(sp)):
            if len(sp[i].split(":"))==2:
              label[int(sp[i].split(":")[0])] = float(sp[i].split(":")[1])
            else:
              label[int(sp[i].split(":")[0])] = 1.0
          example = tf.train.Example(features=tf.train.Features(feature={
            'width': _int64_feature(width),
            'height': _int64_feature(height),
            'image': _bytes_feature(tf.compat.as_bytes(img)),
            'label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
            'name': _bytes_feature(sp[0])
          }))
          writer.write(example.SerializeToString())
  
        else:
          label = int(sp[1])
          example = tf.train.Example(features=tf.train.Features(feature={
            'width': _int64_feature(width),
            'height': _int64_feature(height),
            'image': _bytes_feature(tf.compat.as_bytes(img)),
            'label': _int64_feature(label),
            'name': _bytes_feature(sp[0])
          }))
          writer.write(example.SerializeToString())
  writer.close()

def _save():
  files = os.listdir(args.indexs)
  coord = tf.train.Coordinator()
  threads = []

  i = args.start_index
  for idxf in files:
    threads.append( 
      threading.Thread(target=_save_one, 
        args=(os.path.join(args.indexs, idxf), 
              os.path.join(args.tfrecords, str(i) + ".tfrecords"), 
              args.num_class, args.one_hot)
      )
    )
    i = i+1

  i=0
  thread = []
  for t in threads:
    if i==32:
      for ct in thread: 
        ct.start()
      coord.join(thread)
      i = 0
      thread = [t]
    else:
      thread.append(t)
      i += 1

  for ct in thread:
    ct.start()
  coord.join(thread)

if __name__=='__main__':
  _save()
