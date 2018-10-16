"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""


"""Small library that points to a data set.
Methods of Data class:
        data_files: Returns a python list of all (sharded) data set files.
        reader: Return a reader for a single entry from the data set.
"""

import os
import tensorflow as tf
from datetime import datetime

class Dataset(object):

	def __init__(self, data_dir, worker_hosts = [], task_id = 0, use_split = False, record_pattern='*tfrecords'):
		"""Initialize dataset the path to the data."""
		self.data_dir = data_dir
		self.worker_hosts = worker_hosts
		self.task_id = task_id
		self.use_split = use_split
		self.record_pattern = record_pattern

	def data_filter(self, file_name):
		idx = int(file_name.split('/')[-1].split('.tfrecords')[0])
		return (idx % len(self.worker_hosts) == self.task_id)

	def data_files(self):
		"""Returns a python list of all (sharded) data files.
		Returns:
			python list of all (sharded) data set files.
		Raises:
			ValueError: if there are not data_files in the data dir
		"""
		tf_record_pattern = os.path.join(self.data_dir, self.record_pattern)
		data_files = tf.gfile.Glob(tf_record_pattern)
		data_files = filter(self.data_filter, data_files) if self.use_split else data_files
		if not data_files:
			print('No files found for in data dir %s' % (self.data_dir))
			exit(-1)
		tf.logging.info('[%s] Worker[%d/%d] Files[%d] TrainDir[%s]' %
			(datetime.now(), self.task_id, len(self.worker_hosts), len(data_files), self.data_dir))
		return data_files

	def reader(self):
		"""Return a reader for a single entry from the data set.
		See io_ops.py for details of Reader class.
		Returns:
			Reader object that reads the data set.
		"""
		return tf.TFRecordReader()
