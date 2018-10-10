# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for building the input for the prediction model."""

# TODO: Can remove the parts of this code that enable use of state/action.
# TODO: Do this if you lose overview of the code (its a mess!)

import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import src.utils.utils as utils
import src.data_loading as dl

FLAGS = flags.FLAGS

# Original image dimensions
#ORIGINAL_WIDTH = 640
#ORIGINAL_HEIGHT = 512
#COLOR_CHAN = 3
ORIGINAL_WIDTH = 80
ORIGINAL_HEIGHT = 80
COLOR_CHAN = 1

# Default image dimensions.
IMG_WIDTH = 80
IMG_HEIGHT = 80

# Dimension of the state and action.
STATE_DIM = 5


def build_tfrecord_input(split_string='train', file_nums=[1,2,3,4], training=None, feed_labels=False,
                         return_queue=False, shuffle=True, sequential=False):
  """Create input tfrecord tensors.

     Choose between queue output = dict containing images and labels, or
                    queue output = [images, zeros, zeros].
     Default parameters are so that it can be used for training with prediction_train.py.

  Args:
    split_string: Only used if FLAGS.custom_data.
                    'train', 'val' or 'test' - use only files within path that have this string in their names.
    file_nums:    Only used if FLAGS.custom_data.
                  A list. Only use files ending in 'i.npy', where i is one of the numbers in file_nums.
    training:     Only used if *not* FLAGS.custom_data. Specifies training or validation data.
    feed_labels:  Ignored if not FLAGS.custom_data OR if FLAGS.use_state.
                     Set this to False for compatibility with prediction_train.py. (For training the model.)
                     If True, the first return value is a dictionary containing images (key: 'image_seq')
                     as well as labels. Else, first return value is the image output of the queue.
    return_queue: Ignored if not FLAGS.custom_data.
                     If True, return the queue itself as a fourth return value
                     (so that the queue size, etc. can be accessed.)
    sequential:   Ignored if not FLAGS.custom_data: If True, use only one thread.
  Flags used:
    FLAGS.data_dir  I believe the tfrecords in there should be one file per sequence.
    FLAGS.custom_data: If True, expect files containing '*train*' and '*val*' within data directory.
                             If False, load first 'train_val_split'-percent of the files if train,
                              else the rest.
    FLAGS.custom_data: to use own bouncing_objects or bouncing_circles datasets.
    FLAGS.sequence_length: the lenth of sequences in the rf record files
    FLAGS.use_state
    IMG_HEIGHT =!= IMG_WIDTH
    ORIGINAL_HEIGHT, ORIGINAL_WIDTH: can have any height to width ratio
    COLOR_CHAN
  Returns:
    list of tensors corresponding to images, actions, and states. The images
    tensor is 5D, batch x time x height x width x channels. The state and
    action tensors are 3D, batch x time x dimension.
    The images values are between 0 and 1.
  Raises:
    RuntimeError: if no files found.
  """

  num_threads = min(FLAGS.batch_size, 4)
  if sequential:
      num_threads = 1

  if not FLAGS.custom_data:
    filenames = gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
    if not filenames:
      raise RuntimeError('No data files found.')
    index = int(np.floor(FLAGS.train_val_split * len(filenames)))
    if training:
      filenames = filenames[:index]
    else:
      filenames = filenames[index:]
  else:
    filenames = gfile.Glob(os.path.join(FLAGS.data_dir, '*'+split_string+'*.npy'))
            # todo: reduce validation files to only one, for comparability with other models
    filenames = [fn for fn in filenames if np.any([fn.endswith(str(i)+'.npy') for i in file_nums]) ]
    filenames_labels = [fn.replace('.npy', '.npz').replace('_frames_', '_labels_') for fn in filenames]
    for fn in filenames_labels:
        assert fn in gfile.Glob(os.path.join(FLAGS.data_dir, '*'+split_string+'*.npz')), "Label file not found: \n "+fn
  if not filenames:
    raise RuntimeError('No data files found.')

# load the files and create a queue
  if not FLAGS.custom_data:
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

  else:
      all_data = None
      all_labels = None
      for fn, fnl in zip(filenames, filenames_labels):
          data = np.load(fn)
          labels = dict(np.load(fnl))
          if all_data is None:
              all_data = data
              all_labels = labels
          else:
              all_data = np.concatenate((all_data, data), axis=0)
              all_labels = utils.vstack_array_dicts(all_labels, labels, allow_new_keys=False)
      all_labels = OrderedDict(all_labels)
      shape_arr = [[FLAGS.sequence_length, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN]]
      type_arr = [tf.uint8]
      for key, val in all_labels.items():
          assert len(val) == len(all_data)
          #shape_arr.append((FLAGS.sequence_length,)+ val.shape[1:])
          shape_arr.append(val.shape[1:])
          type_arr.append(val.dtype)
      if shuffle:
          min_after_dequeue = min(100, int(100*FLAGS.batch_size/2.))
          data_queue = tf.RandomShuffleQueue(capacity=100*FLAGS.batch_size, min_after_dequeue=min_after_dequeue, dtypes=type_arr, shapes=shape_arr,
                                         names=['image_seq']+list(all_labels.keys()))
      else:
          data_queue = tf.FIFOQueue(capacity=100*FLAGS.batch_size, dtypes=type_arr, shapes=shape_arr,
                                         names=['image_seq']+list(all_labels.keys()))
      if len(all_data.shape) == 4:
          all_data = all_data[..., np.newaxis]
      assert len(all_data.shape) == 5 and all_data.shape[-1] in [1,2,3,4], "Very weird number of channels found in stored dataset: "+str(data.shape[-1]+". Full shape was: "+str(data.shape))

      n_samples = len(all_data)
      data_queue.n_samples = n_samples # uh-oh
      enqueue_op = data_queue.enqueue_many({'image_seq': all_data, **all_labels})
      qr = tf.train.QueueRunner(data_queue, [enqueue_op] * num_threads)
      tf.train.add_queue_runner(qr)
      serialized_example_dict = data_queue.dequeue()


  image_seq, state_seq, action_seq = [], [], []

  for i in range(FLAGS.sequence_length):
    #image_name = 'move/' + str(i) + '/image/encoded'
    #action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
    #state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'
    if not FLAGS.custom_data:
      #image_name, action_name, state_name = get_feature_names_custom(i)
      #assert not FLAGS.use_state, "Dont want to use states in this project."
    #else:
      image_name, action_name, state_name = get_feature_names_finn(i)
      if FLAGS.use_state:
          features = {image_name: tf.FixedLenFeature([1], tf.string),
                      action_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
                      state_name: tf.FixedLenFeature([STATE_DIM], tf.float32)}
      else:
        features = {image_name: tf.FixedLenFeature([1], tf.string)}
        #if not FLAGS.custom_data:
      features = tf.parse_single_example(serialized_example, features=features)
        #else:
        #    features = tf.train.Example()
        #    features.ParseFromString(serialized_example, features=features)
      image_buffer = tf.reshape(features[image_name], shape=[])
      image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
      image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

    else:
        image = serialized_example_dict['image_seq'][i, ...]
        #image = serialized_example[i, ...]
        image = tf.convert_to_tensor(image, np.uint8)

    if IMG_HEIGHT != IMG_WIDTH:
      raise ValueError('Unequal height and width unsupported')

    crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
    image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    image_seq.append(image)

    if FLAGS.use_state:
      state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
      state_seq.append(state)
      action = tf.reshape(features[action_name], shape=[1, STATE_DIM])
      action_seq.append(action)

  image_seq = tf.concat(image_seq, 0)

  if FLAGS.use_state:
    state_seq = tf.concat(state_seq, 0)
    action_seq = tf.concat(action_seq, 0)
    [image_batch, action_batch, state_batch] = tf.train.batch(
        [image_seq, action_seq, state_seq],
        FLAGS.batch_size,
        num_threads=num_threads, #FLAGS.batch_size,
        capacity=100 * FLAGS.batch_size)
    if return_queue:
       return image_batch, action_batch, state_batch, data_queue
    return image_batch, action_batch, state_batch
  else:
    serialized_example_dict['image_seq'] = image_seq
    zeros_batch = tf.zeros([FLAGS.batch_size, FLAGS.sequence_length, STATE_DIM])
    if feed_labels:  # output of the queue will be a dict
        image_label_batch = tf.train.batch(
            { #'image_seq': image_seq,
                **serialized_example_dict},
            FLAGS.batch_size,
            num_threads=num_threads,  #FLAGS.batch_size,
            capacity=100 * FLAGS.batch_size
        )
    else: # queue output is just the images
        image_label_batch = tf.train.batch(
            [image_seq],
            FLAGS.batch_size,
            num_threads=num_threads,  # FLAGS.batch_size,
            capacity=100 * FLAGS.batch_size)
    if return_queue:
        return image_label_batch, zeros_batch, zeros_batch, data_queue
    return image_label_batch, zeros_batch, zeros_batch

def get_feature_names_finn(i):
    image_name = 'move/' + str(i) + '/image/encoded'
    action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
    state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'
    return image_name, action_name, state_name

def get_feature_names_custom(i):
    image_name = str(i) + '/image_raw'
    return image_name, '', ''
