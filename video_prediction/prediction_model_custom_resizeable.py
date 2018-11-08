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

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from collections import OrderedDict


import sys
sys.path.append('../../../src/')
import src.utils.tf_utils as tf_utils
from src.utils.tf_utils import get_multi_cell_output_size as mcell_out_size
#from src.data_pipeline import DataPipeline
from src.frame_predictor import BaseFramePredictorModel
from src.utils.custom_layers import LayerCell, StridedConvLSTMCell, StridedLeeConv2DLSTMCell


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


class FramePredictorFinn(object):
    ''' Model 'at the core' of original 'Model' class, to store all the in-between features.
        Actual Model will get this one as an attribute.'''
    #def construct_model(
    def __init__(self, images, actions=None,
                    states=None, model_config=None,
                    iter_num=-1.0, k=-1, context_frames=2,
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

        k = k
        use_state = False
        num_masks = model_config['num_masks']
        subtype_key = 'Finn_subtype' if 'Finn_subtype' in model_config.keys() else 'model_subtype'
        #stp = model_config['Finn_subtype'].lower() == 'stp'
        #cdna = model_config['Finn_subtype'].lower() == 'cdna'
        #dna = model_config['Finn_subtype'].lower() == 'dna'
        stp = model_config[subtype_key].lower() == 'stp'
        cdna = model_config[subtype_key].lower() == 'cdna'
        dna = model_config[subtype_key].lower() == 'dna'
        #context_frames = model_config['context_frames']
        #self.schedule = model_config['freerun_schedule']
        self.schedule = schedule
        self.model_config = model_config
        #sequence_length = train_config['max_seq_length']

        # architecture
        #self.use_batch_norm = model_config['use_batch_norm']
        #self.replace_pool_by_conv = model_config['replace_pool_by_conv']
        #self.non_conv_rnn_mode = model_config['non_conv_rnn_mode']
        #self.rnn_mode = model_config['rnn_mode']

        self.filter_size_first = model_config['filter_size_first']
        self.filter_size     = model_config['filter_size']
        self.conv_stride     = model_config['conv_stride']
        self.enc_filter_size = model_config['enc_filter_size']
        self.enc_stride      = model_config['enc_stride']
        self.encoder_layers_sz = model_config['encoder_layers_sz']
        self.decoder_layers_sz = model_config['decoder_layers_sz']  # mirror encoder
        self.pool_where_enc = model_config['pool_where_enc']
        self.unpool_where_dec = model_config['unpool_where_dec']
        # number of features in each encoding layer
        self.bottleneck_layers_sz = model_config['bottleneck_layers_sz']

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        if stp + cdna + dna != 1:
            raise ValueError('More than one, or no network option specified.')
        self.batch_size, self.img_height, self.img_width, self.color_channels = images[0].get_shape()[0:4]
        lstm_func = basic_conv_lstm_cell

        self.masks = []
        self.transformed = []

        n_layers = (len(self.encoder_layers_sz) + len(self.decoder_layers_sz) + 1)
        n_pool_unpool = np.sum(self.pool_where_enc) + np.sum(self.unpool_where_dec) +4 # +4: the first layer downsamples, always; plus enc3, plus enc4, plus enc7 at the end
        self.lstm_size = np.int32(np.array(self.encoder_layers_sz + [self.bottleneck_layers_sz] + self.decoder_layers_sz))
        # variables storing stuff at each timestep
        self.encs = [[] for _ in range(n_pool_unpool)]
        self.hidden_layers = [[] for _ in range(n_layers)]
        self.lstm_states = [[] for _ in range(len(self.lstm_size))]
        # stores the states from the latest round for each layer:
        lstm_states = [None for _ in range(len(self.lstm_size))]
        assert np.sum(self.pool_where_enc) == np.sum(self.unpool_where_dec), "down- and upsampling don't correspond"

        #self.enc0, self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6, self.enc7 = [], [], [], [], [], [], [], []
       # self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.hidden5, self.hidden6, self.hidden7 = [], [], [], [], [], [], []
       # self.encs = [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6, self.enc7]
       # self.hidden_layers = [self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.hidden5, self.hidden6, self.hidden7]

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
                gt_perc_fun = lambda it_num: tf.maximum(0., 1. - it_num / k * 1.)
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

        for image, action in zip(images[:-1], actions[:-1]):
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            # variables re-written for each timestep
            hidden_layers = [[] for _ in range(n_layers)]
            encs = [[] for _ in range(n_pool_unpool)]
            encs_to_concat = []

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

                #  ----  encoder ------------------------------------------| |--bottleneck---| |= t-conv ----- = t-conv ----- decoder ----------------|
                # enc0, hidden1, hidden2, enc1, hidden3, hidden4, enc2, -concat-, enc3, hidden5, enc4, hidden6, enc5, hidden7, enc6, enc7=dna/cdna/stp
                # [32,     32,     32,     32,     64,      64,     64,             =  ,  128,    128,    64,    64,    32,      32,   depends ]

                ## Encoder
                encs[0] = slim.layers.conv2d(
                    prev_image,
                    self.encoder_layers_sz[0], #[5, 5],
                    self.filter_size_first,
                    stride=2,  # first conv downsamples
                    scope='scale1_conv1',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm1'})
                enc_count = 1
                hidden_count = 0
                layer_before = encs[0]
                encs_to_concat.append(encs[0])

                for i, feat_sz in enumerate(self.encoder_layers_sz):
                    hidden, lstm_states[i] = lstm_func(
                        layer_before, lstm_states[i], self.lstm_size[i], scope='state'+str(i+1)
                    )
                    hidden_layers[i] = tf_layers.layer_norm(hidden, scope='layer_norm'+str(i+2))
                    layer_before = hidden_layers[i]
                    hidden_count += 1
                    if self.pool_where_enc[i]:
                        encs[enc_count] = slim.layers.conv2d(
                                encs[enc_count-1], encs[enc_count-1].get_shape()[3], self.enc_filter_size,
                                stride=self.enc_stride, scope='conv'+str(enc_count))
                        layer_before = encs[enc_count]
                        encs_to_concat.append(encs[enc_count])
                        enc_count += 1
                # drop lowest-size encoder, don't need to concat that anywhere
                encs_to_concat = encs_to_concat[:-1]

                # ~~~ Concat "actions" etc (just zeros), followed by original enc3

                # Pass in state and action.
                smear = tf.reshape(
                    state_action,
                    [int(self.batch_size), 1, 1, int(state_action.get_shape()[1])])
                smear = tf.tile(
                    smear, [1, int(encs[enc_count-1].get_shape()[1]), int(encs[enc_count-1].get_shape()[2]), 1])
                if use_state:
                    encs[enc_count-1] = tf.concat([encs[enc_count-1], smear], 3)
                encs[enc_count] = slim.layers.conv2d(
                    encs[enc_count-1], hidden_layers[hidden_count-1].get_shape()[3], [1, 1], stride=1, scope='conv'+str(hidden_count+2))
                enc_count += 1

                # ~~~ Bottleneck
                assert self.lstm_size[hidden_count] == self.bottleneck_layers_sz
                assert hidden_count == len(self.encoder_layers_sz)
                bottleneck_hidden, lstm_states[hidden_count] = lstm_func(
                    encs[enc_count-1], lstm_states[hidden_count], self.lstm_size[hidden_count], scope='state'+str(hidden_count+1))  # last 8x8
                bottleneck_hidden = tf_layers.layer_norm(bottleneck_hidden, scope='layer_norm'+str(hidden_count+3))
                hidden_layers[hidden_count] = bottleneck_hidden
                hidden_count += 1
                # originally enc4, first transpose convolution
                encs[enc_count] = slim.layers.conv2d_transpose(
                    hidden_layers[hidden_count-1], hidden_layers[hidden_count-1].get_shape()[3],
                    self.enc_filter_size, stride=self.enc_stride, scope='convt1')
                enc_count += 1
                decoder_count = 1

                # ~~~ Decoder
                assert hidden_count == len(self.encoder_layers_sz)+1

                for i, feat_sz in enumerate(self.decoder_layers_sz):
                    hidden, lstm_states[hidden_count] = lstm_func(
                        encs[enc_count-1], lstm_states[hidden_count], self.lstm_size[hidden_count], scope='state'+str(hidden_count+2))
                    hidden_layers[hidden_count] = tf_layers.layer_norm(hidden, scope='layer_norm'+str(hidden_count+3))
                    hidden_count += 1

                    # todo: Here's one difference to the original; the original only had layer_norm in the second decoder layer (enc6,
                    # todo:   directly before the final 'warping'/whatever
                    if self.unpool_where_dec[i]:
                        hidden_layers[hidden_count-1] = tf.concat([hidden, encs_to_concat[-1-i]], 3)
                        encs[enc_count] = slim.layers.conv2d_transpose(
                            hidden_layers[hidden_count-1], hidden_layers[hidden_count-1].get_shape()[3],
                            self.enc_filter_size[0], stride=self.enc_stride, scope='convt'+str(decoder_count+1),
                            normalizer_fn=tf_layers.layer_norm,
                            normalizer_params={'scope': 'layer_norm'+str(enc_count+2)})
                        last_enc = encs[enc_count]
                        decoder_count += 1
                        enc_count += 1

                # ~~~ Final layers
                if dna:
                    # Using largest hidden state for predicting untied conv kernels.
                    very_last_enc = slim.layers.conv2d_transpose(
                        last_enc, DNA_KERN_SIZE ** 2, 1, stride=1, scope='convt_last')
                else:
                    # Using largest hidden state for predicting a new image layer.
                    very_last_enc = slim.layers.conv2d_transpose(
                        last_enc, self.color_channels, 1, stride=1, scope='convt_last')
                    # This allows the network to also generate one image from scratch,
                    # which is useful when regions of the image become unoccluded.
                    transformed = [tf.nn.sigmoid(very_last_enc)]
                encs[enc_count] = very_last_enc #
                if stp:
                    stp_input0 = tf.reshape(bottleneck_hidden, [int(self.batch_size), -1])
                    stp_input1 = slim.layers.fully_connected(
                        stp_input0, 100, scope='fc_stp')
                    transformed += stp_transformation(prev_image, stp_input1, num_masks)
                elif cdna:
                    cdna_input = tf.reshape(bottleneck_hidden, [int(self.batch_size), -1])
                    transformed += cdna_transformation(prev_image, cdna_input, num_masks,
                                                       int(self.color_channels))
                elif dna:
                    # Only one mask is supported (more should be unnecessary).
                    if num_masks != 1:
                        raise ValueError('Only one mask is supported for DNA model.')
                    transformed = [dna_transformation(prev_image, very_last_enc)]

                masks = slim.layers.conv2d_transpose(
                    last_enc, num_masks + 1, 1, stride=1, scope='convt7')
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
                #lstm_states = [lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7]
                #encs = [enc0, enc1, enc2, enc3, enc4, enc5, enc6, enc7]
                #hidden_layers = [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]
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
