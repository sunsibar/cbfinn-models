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

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from lstm_ops import basic_conv_lstm_cell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


class CoreModel(object):
    ''' Model 'at the core' of original 'Model' class, to store all the in-between features.
        Actual Model will get this one as an attribute.'''
    #def construct_model(
    def __init__(self, images, actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    use_state=True,
                    num_masks=10,
                    stp=False,
                    cdna=True,
                    dna=False,
                    context_frames=2,
                    schedule='logistic'):
          """Build convolutional lstm video predictor using STP, CDNA, or DNA.

              Args:
                images: tensor of ground truth image sequences
                actions: tensor of action sequences
                states: tensor of ground truth state sequences
                iter_num: tensor of the current training iteration (for sched. sampling)
                k: constant used for scheduled sampling. -1 to feed in own prediction.
                use_state: True to include state and action in prediction
                num_masks: the number of different pixel motion predictions (and
                           the number of masks for each of those predictions)
                stp: True to use Spatial Transformer Predictor (STP)
                cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
                dna: True to use Dynamic Neural Advection (DNA)
                context_frames: number of ground truth frames to pass in before
                                feeding in own predictions
                schedule: the type of scheduling; one of 'logistic' (standard) and 'linear'
              Returns: (no, use self.return_gen to get these two)
                self.gen_images: predicted future image frames
                self.gen_states: predicted future states

              Raises:
                ValueError: if more than one network option specified or more than 1 mask
                specified for DNA model.
          """
          if stp + cdna + dna != 1:
            raise ValueError('More than one, or no network option specified.')
          self.batch_size, self.img_height, self.img_width, self.color_channels = images[0].get_shape()[0:4]
          lstm_func = basic_conv_lstm_cell
          self.schedule = schedule

          self.masks = []
          self.transformed = []

          self.enc0, self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6, self.enc7 = [], [], [], [], [], [], [], []
          self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.hidden5, self.hidden6, self.hidden7 = [], [], [], [], [], [], []
          self.encs = [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6, self.enc7]
          self.hidden_layers = [self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.hidden5, self.hidden6, self.hidden7 ]

          # Generated robot states and images.
          self.gen_states, self.gen_images = [], []
          current_state = states[0]

          if k == -1:
              self.feedself = True
              self.perc_ground_truth = tf.constant(0.)
          else:
              # Scheduled sampling:
              # Calculate number of ground-truth frames to pass in.
              if self.schedule == 'linear':
                  gt_perc_fun = lambda it_num: tf.maximum(0, 100. - it_num / k * 100.)
              elif self.schedule == 'logistic':
                  gt_perc_fun = lambda it_num: (k / (k + tf.exp(it_num / k)))
              else:
                  raise ValueError(
                      "Unknown value for parameter 'schedule': " + self.schedule + "; allowed are strings 'logistic' and 'linear'. ")

              # num_ground_truth = tf.to_int32(
              #    tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
              num_ground_truth = tf.to_int32(tf.round(tf.to_float(self.batch_size) * gt_perc_fun(iter_num)))
              self.perc_ground_truth = gt_perc_fun(iter_num)
              self.feedself = False

          # LSTM state sizes and states.
          self.lstm_size = lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
          lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
          lstm_state5, lstm_state6, lstm_state7 = None, None, None
          self.lstm_state1, self.lstm_state2, self.lstm_state3, self.lstm_state4, self.lstm_state5, self.lstm_state6, self.lstm_state7 = \
              [], [], [], [], [], [], []
          self.lstm_states = [self.lstm_state1, self.lstm_state2, self.lstm_state3, self.lstm_state4, self.lstm_state5, self.lstm_state6, self.lstm_state7]

          for image, action in zip(images[:-1], actions[:-1]):
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            done_warm_start = len(self.gen_images) > context_frames - 1
            with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

              if self.feedself and done_warm_start:
                # Feed in generated image.
                prev_image = self.gen_images[-1]
              elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(image, self.gen_images[-1], self.batch_size,
                                              num_ground_truth)
              else:
                # Always feed in ground_truth
                prev_image = image

              # Predicted state is always fed back in
              state_action = tf.concat([action, current_state], 1)

              enc0 = slim.layers.conv2d(
                  prev_image,
                  32, [5, 5],
                  stride=2,
                  scope='scale1_conv1',
                  normalizer_fn=tf_layers.layer_norm,
                  normalizer_params={'scope': 'layer_norm1'})

              hidden1, lstm_state1 = lstm_func(
                  enc0, lstm_state1, lstm_size[0], scope='state1')
              hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
              hidden2, lstm_state2 = lstm_func(
                  hidden1, lstm_state2, lstm_size[1], scope='state2')
              hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
              enc1 = slim.layers.conv2d(
                  hidden2, hidden2.get_shape()[3], [3, 3], stride=2, scope='conv2')

              hidden3, lstm_state3 = lstm_func(
                  enc1, lstm_state3, lstm_size[2], scope='state3')
              hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
              hidden4, lstm_state4 = lstm_func(
                  hidden3, lstm_state4, lstm_size[3], scope='state4')
              hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm5')
              enc2 = slim.layers.conv2d(
                  hidden4, hidden4.get_shape()[3], [3, 3], stride=2, scope='conv3')

              # Pass in state and action.
              smear = tf.reshape(
                  state_action,
                  [int(self.batch_size), 1, 1, int(state_action.get_shape()[1])])
              smear = tf.tile(
                  smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
              if use_state:
                enc2 = tf.concat([enc2, smear], 3)
              enc3 = slim.layers.conv2d(
                  enc2, hidden4.get_shape()[3], [1, 1], stride=1, scope='conv4')

              hidden5, lstm_state5 = lstm_func(
                  enc3, lstm_state5, lstm_size[4], scope='state5')  # last 8x8
              hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
              enc4 = slim.layers.conv2d_transpose(
                  hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

              hidden6, lstm_state6 = lstm_func(
                  enc4, lstm_state6, lstm_size[5], scope='state6')  # 16x16
              hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')
              # Skip connection.
              hidden6 = tf.concat([hidden6, enc1], 3)  # both 16x16

              enc5 = slim.layers.conv2d_transpose(
                  hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
              hidden7, lstm_state7 = lstm_func(
                  enc5, lstm_state7, lstm_size[6], scope='state7')  # 32x32
              hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

              # Skip connection.
              hidden7 = tf.concat([hidden7, enc0], 3)  # both 32x32

              enc6 = slim.layers.conv2d_transpose(
                  hidden7,
                  hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                  normalizer_fn=tf_layers.layer_norm,
                  normalizer_params={'scope': 'layer_norm9'})

              if dna:
                # Using largest hidden state for predicting untied conv kernels.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, DNA_KERN_SIZE**2, 1, stride=1, scope='convt4')
              else:
                # Using largest hidden state for predicting a new image layer.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, self.color_channels, 1, stride=1, scope='convt4')
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                transformed = [tf.nn.sigmoid(enc7)]

              if stp:
                stp_input0 = tf.reshape(hidden5, [int(self.batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')
                transformed += stp_transformation(prev_image, stp_input1, num_masks)
              elif cdna:
                cdna_input = tf.reshape(hidden5, [int(self.batch_size), -1])
                transformed += cdna_transformation(prev_image, cdna_input, num_masks,
                                                   int(self.color_channels))
              elif dna:
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                  raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

              masks = slim.layers.conv2d_transpose(
                  enc6, num_masks + 1, 1, stride=1, scope='convt7')
              masks = tf.reshape(
                  tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                  [int(self.batch_size), int(self.img_height), int(self.img_width), num_masks + 1])
              mask_list = tf.split(masks, num_masks + 1, 3)
              output = mask_list[0] * prev_image
              for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask
              self.gen_images.append(output)

              current_state = slim.layers.fully_connected(
                  state_action,
                  int(current_state.get_shape()[1]),
                  scope='state_pred',
                  activation_fn=None)

              self.gen_states.append(current_state)
              self.masks.append(masks)
              self.transformed.append(transformed)
              lstm_states = [lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7]
              encs = [enc0, enc1, enc2, enc3, enc4, enc5, enc6, enc7]
              hidden_layers = [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]
              for i, enc in enumerate(encs):
                  self.encs[i].append(enc)
              for i, hidd in enumerate(hidden_layers):
                  self.hidden_layers[i].append(hidd)
              for i, lstm_st in enumerate(lstm_states):
                  self.lstm_states[i].append(lstm_st)

    def return_generated(self):
        return self.gen_images, self.gen_states


## Utility functions
def stp_transformation(prev_image, stp_input, num_masks):
  """Apply spatial transformer predictor (STP) to previous image.

  Args:
    prev_image: previous image to be transformed.
    stp_input: hidden layer to be used for computing STN parameters.
    num_masks: number of masks and hence the number of STP transformations.
  Returns:
    List of images transformed by the predicted STP parameters.
  """
  # Only import spatial transformer if needed.
  import os
  import sys
  sys.path.insert(1, os.path.join(sys.path[0], '..'))
  from transformer.spatial_transformer import transformer

  identity_params = tf.convert_to_tensor(
      np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
  transformed = []
  for i in range(num_masks - 1):
    params = slim.layers.fully_connected(
        stp_input, 6, scope='stp_params' + str(i),
        activation_fn=None) + identity_params
    transformed.append(transformer(prev_image, params, (prev_image.get_shape()[1], prev_image.get_shape()[2])))   # Note: noob added the last argument to transformer here, could be that height and width are reversed

  return transformed


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels):
  """Apply convolutional dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    cdna_input: hidden lyaer to be used for computing CDNA kernels.
    num_masks: the number of masks and hence the number of CDNA transformations.
    color_channels: the number of color channels in the images.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  batch_size = int(cdna_input.get_shape()[0])

  # Predict kernels using linear function of last hidden layer.
  cdna_kerns = slim.layers.fully_connected(
      cdna_input,
      DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
      scope='cdna_params',
      activation_fn=None)

  # Reshape and normalize.
  cdna_kerns = tf.reshape(
      cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
  cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
  norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keepdims=True)
  cdna_kerns /= norm_factor

  cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, color_channels, 1])
  cdna_kerns = tf.split(cdna_kerns, batch_size, 0)
  prev_images = tf.split(prev_image, batch_size, 0)

  # Transform image.
  transformed = []
  for kernel, preimg in zip(cdna_kerns, prev_images):
    kernel = tf.squeeze(kernel)
    if len(kernel.get_shape()) == 3:
      # => either num_masks is 1, or num input channels is 1.
      if num_masks == 1:
        kernel = tf.expand_dims(kernel, -1)
      elif color_channels == 1:
        kernel = tf.expand_dims(kernel, -2)
      else:
          raise RuntimeError("Check where kernels size is coming from, exactly. How could that happen?")
    transformed.append(
        tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))
  transformed = tf.concat(transformed, 0)
  transformed = tf.split(transformed, num_masks, 3)
  return transformed


def dna_transformation(prev_image, dna_input):
  """Apply dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    dna_input: hidden lyaer to be used for computing DNA transformation.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  # Construct translated images.
  prev_image_pad = tf.pad(prev_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
  image_height = int(prev_image.get_shape()[1])
  image_width = int(prev_image.get_shape()[2])

  inputs = []
  for xkern in range(DNA_KERN_SIZE):
    for ykern in range(DNA_KERN_SIZE):
      inputs.append(
          tf.expand_dims(
              tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                       [-1, image_height, image_width, -1]), [3]))
  inputs = tf.concat(inputs, 3)

  # Normalize channels to 1.
  kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
  kernel = tf.expand_dims(
      kernel / tf.reduce_sum(
          kernel, [3], keep_dims=True), [4])
  return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
  """Sample batch with specified mix of ground truth and generated data points.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    num_ground_truth: number of ground-truth examples to include in batch.
  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  idx = tf.random_shuffle(tf.range(int(batch_size)))
  # take num_ground_truth random indices from the batch
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  # the remaining indices of the batch - there's (batch_size - num_ground_truth) of them
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)
  return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                           [ground_truth_examps, generated_examps])
