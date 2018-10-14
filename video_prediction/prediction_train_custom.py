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

"""Code for training the prediction model."""

import numpy as np
import tensorflow as tf
import logging
import datetime
import time

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from prediction_input_custom import build_tfrecord_input
from prediction_model import construct_model

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
#from src.utils.utils import set_logger
import src.utils.utils as utils

from src.frame_predictor_Finn2015_config import train_config, model_config

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = train_config['summarize_every'] #1000

# How often to run a batch through the validation model.
VAL_INTERVAL = train_config['validate_every']  # 200

# How often to save a model checkpoint
SAVE_INTERVAL = train_config['save_checkpoints_every_epoch'] #2000# 2000

# tf record data location, OR if custom data: .npy data location:
#DATA_DIR = '/home/noobuntu/Sema2018/data/robots_pushing/push/push_train'    #'push/push_testnovel' # 'push/push_train'   # '../../../../data/bouncing_circles/short_sequences/static_simple_1_bcs'
#DATA_DIR = '../../../../data/gen/debug_bouncing_circles/static_simple_2_bcs/tfrecords'  # <- for VM on windows
#DATA_DIR = '../../../../data/gen/bouncing_circles/short_sequences/static_simple_1_bcs'
DATA_DIR = train_config['data_dir']
    # '../../../../data/bouncing_circles/short_sequences/static_simple_1_bcs'
#DATA_DIR = '../../../../data/robots_pushing/push/push_train' # 'push/push_train'   # '../../../../data/bouncing_circles/short_sequences/static_simple_1_bcs'


# local output directory

timestamp =  datetime.datetime.now().strftime("%y-%b-%d_%Hh%M-%S")
OUT_DIR = os.path.join(train_config['output_dir'] , timestamp) #'./train_out/nowforreal/'+timestamp

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', train_config['n_epochs'] , 'number of training iterations.')   # 50000
flags.DEFINE_string('pretrained_model', '', #'./train_out/nowforreal/18-Sep-27_00h40-36/model26002',  # /home/noobuntu/Sema2018/reps2018/models/finn_models/video_prediction/trained/nowforreal/model190.index
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', train_config['max_seq_length'],
                     'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', train_config['context_frames'], '# of frames before predictions.')
flags.DEFINE_integer('use_state', 0,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', model_config['model_subtype'],
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', model_config['num_masks'],
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', train_config['schedsamp_k'],
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
# Scheduled sampling:
# After initial context frames, for each timestep, the training batch will be composed of num_ground_truth
#   gt input frames (at that time) and (batch_size - num_ground_truth) frames predicted from the last timestep.
#   Which samples to take gt vs prediction frames from is sampled randomly at each timestep (I think).
#   The number of ground truth frames after conditioning is calculated from k as:
#         num_ground_truth = round((batch_size) * (k / (k + exp(iter_num / k)))))
#
#flags.DEFINE_float('train_val_split', 0.95,
#                   'The percentage of files to use for the training set,'
#                   ' vs. the validation set. Unused if data is given in '
#                   'two separate folders, "train" and "val".')

flags.DEFINE_integer('batch_size', train_config['batch_size'], 'batch size for training')
flags.DEFINE_float('learning_rate', train_config['learning_rate'], # 0.001,
                   'the base learning rate of the generator')
flags.DEFINE_integer('custom_data', 1, ' If True (1), uses tf-record feature naming '
                     'for the bouncing_objects dataset, and loosk for the '
                     'data in separate /train and /val directories')


## Helper functions
def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):

  def __init__(self,
               images=None,
               actions=None,
               states=None,
               sequence_length=None,
               reuse_scope=None):

    if sequence_length is None:
      sequence_length = FLAGS.sequence_length

    self.prefix = prefix = tf.placeholder(tf.string, [])
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []

    # Split into timesteps.
    actions = tf.split(actions, actions.get_shape()[1], 1)
    actions = [tf.reshape(act, list(act.get_shape()[0:1])+list(act.get_shape()[2:])) for act in actions]
    #actions = [tf.squeeze(act) for act in actions]
    states = tf.split(states, states.get_shape()[1], 1)
    states = [tf.reshape(st, list(st.get_shape()[0:1])+list(st.get_shape()[2:])) for st in states]
    #states = [tf.squeeze(st) for st in states]
    images = tf.split(images, images.get_shape()[1], 1)
    images = [tf.reshape(img, list(img.get_shape()[0:1])+list(img.get_shape()[2:])) for img in images]
    # ^squeeze only the second dimension (split dimension)
    #images = [tf.squeeze(img) for img in images]

    if reuse_scope is None:
      gen_images, gen_states = construct_model(
          images,
          actions,
          states,
          iter_num=self.iter_num,
          k=FLAGS.schedsamp_k,
          use_state=FLAGS.use_state,
          num_masks=FLAGS.num_masks,
          cdna=FLAGS.model == 'CDNA',
          dna=FLAGS.model == 'DNA',
          stp=FLAGS.model == 'STP',
          context_frames=FLAGS.context_frames)
    else:  # If it's a validation or test model.
      with tf.variable_scope(reuse_scope, reuse=True):
        gen_images, gen_states = construct_model(
            images,
            actions,
            states,
            iter_num=self.iter_num,
            k=FLAGS.schedsamp_k,
            use_state=FLAGS.use_state,
            num_masks=FLAGS.num_masks,
            cdna=FLAGS.model == 'CDNA',
            dna=FLAGS.model == 'DNA',
            stp=FLAGS.model == 'STP',
            context_frames=FLAGS.context_frames)

    self.gen_images = gen_images
    gt_perc_fun = lambda iter_num: (FLAGS.schedsamp_k / (FLAGS.schedsamp_k + tf.exp(iter_num / FLAGS.schedsamp_k))) \
        if FLAGS.schedsamp_k != -1 else 0
    self.perc_ground_truth = gt_perc_fun(self.iter_num)
    self.count_parameters()

    # L2 loss, PSNR for eval.
    loss, psnr_all = 0.0, 0.0
    for i, x, gx in zip(
        list(range(len(gen_images))), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      recon_cost = mean_squared_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i
      summaries.append(
          tf.summary.scalar(name=prefix.name + '_recon_cost' + str(i), tensor=recon_cost))
      summaries.append(tf.summary.scalar(name=prefix.name + '_psnr' + str(i), tensor=psnr_i))
      loss += recon_cost

    for i, state, gen_state in zip(
        list(range(len(gen_states))), states[FLAGS.context_frames:],
        gen_states[FLAGS.context_frames - 1:]):
      state_cost = mean_squared_error(state, gen_state) * 1e-4
      summaries.append(
          tf.summary.scalar(name=prefix.name + '_state_cost' + str(i), tensor=state_cost))
      loss += state_cost
    summaries.append(tf.summary.scalar(name=prefix.name + '_psnr_all', tensor=psnr_all))
    self.psnr_all = psnr_all

    self.loss = loss = loss / np.float32(len(images) - FLAGS.context_frames)

    summaries.append(tf.summary.scalar(name=prefix.name + '_loss', tensor=loss))

    self.lr = tf.placeholder_with_default(FLAGS.learning_rate, ())

    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    self.summ_op = tf.summary.merge(summaries)

  def count_parameters(self):
      """ Counts the number of trainable parameters in this model """
      self.n_parameters = 0
      for v in tf.trainable_variables():
          params = 1
          for s in v.get_shape():
              params *= s.value
          self.n_parameters += params
      return self.n_parameters


# - - - - - - -- - - - - - - - - - - - -

def main(unused_argv):

  assert FLAGS.batch_size <= 16, "Servers (at INI) have 8GB; a batch size of 16 is the maximum for this model."

  print('Constructing models and inputs.')
  with tf.variable_scope('model', reuse=None) as training_scope:
      #if FLAGS.custom_data:
      images, actions, states = build_tfrecord_input(split_string='train', file_nums=train_config['train_file_nums'], )
      #else:
      #    images, actions, states = build_tfrecord_input(training=True)
      model = Model(images, actions, states, FLAGS.sequence_length)

  with tf.variable_scope('val_model', reuse=None):
      #if FLAGS.custom_data:
      val_images, val_actions, val_states = build_tfrecord_input(split_string='val', file_nums=train_config['val_file_nums'])
      #else:
      #    val_images, val_actions, val_states = build_tfrecord_input(training=False)
      val_model = Model(val_images, val_actions, val_states,
                            FLAGS.sequence_length, training_scope)

  print('Constructing saver.')
  # Make saver.
  saver = tf.train.Saver( tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
  saver_best = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=train_config['n_keep_checkpoints'])

  utils.set_logger("./logs/")
  # Make training session.
  sess = tf.InteractiveSession()
  summary_writer = tf.summary.FileWriter(
      FLAGS.event_log_dir, graph=sess.graph, flush_secs=10)

  if FLAGS.pretrained_model:
    saver.restore(sess, FLAGS.pretrained_model)

  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(sess, coord=coord)
  sess.run(tf.global_variables_initializer())
  start_time = time.time()
  lowest_loss = np.inf
  val_loss = np.inf
  train_time_lowest = np.inf

  tf.logging.info('FLAGS.num_interations: ' + str(FLAGS.num_iterations))
  tf.logging.info('time, iteration number, cost, lr, percent gt')
  #logging.info('iteration number, cost')

  # Run training.
  for itr in range(FLAGS.num_iterations):
    # Generate new batch of data.
    feed_dict = {model.prefix: 'train',
                 model.iter_num: np.float32(itr),
                 model.lr: FLAGS.learning_rate}
    cost, _, summary_str, p_gt, lr = sess.run([model.loss, model.train_op, model.summ_op, model.perc_ground_truth, model.lr], feed_dict)

    # Print info: iteration #, cost.
    time_delta = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    tf.logging.info(time_delta + ' itr: ' +str(itr) + ' cost: ' + str(cost) + ' lr: '+str(lr) + ' %gt: '+str(p_gt))
    #logging.info(str(itr) + ' ' + str(cost))


    if (itr) % VAL_INTERVAL == 2:
      # Run through validation set.
      feed_dict = {val_model.lr: 0.0,
                   val_model.prefix: 'val',
                   val_model.iter_num: np.float32(itr)}
      _, val_summary_str, val_loss = sess.run([val_model.train_op, val_model.summ_op, val_model.loss],
                                     feed_dict)
      summary_writer.add_summary(val_summary_str, itr)

    if (itr) % SAVE_INTERVAL == 2:
      tf.logging.info('Saving model.')
      saver.save(sess, FLAGS.output_dir + '/model' + str(itr))
    if val_loss < lowest_loss and itr >= 10000: # 10000: should depend on the value of schedsamp-k, but am too lazy to do
                                # that right now. Ignore good values in the area where a lot of ground truth data is used.
        best_save_path = os.path.join(FLAGS.output_dir, 'best_weights')
        best_save_path = saver_best.save(sess, best_save_path, global_step=itr + 1)
        logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
        lowest_loss = val_loss
        train_time_lowest = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    if (itr) % SUMMARY_INTERVAL:
      summary_writer.add_summary(summary_str, itr)

  tf.logging.info('Saving model.')
  saver.save(sess, FLAGS.output_dir + '/model')

  # dump: time taken, #params, best validation error
  infodict = {'train_time': str(datetime.timedelta(seconds=int(time.time() - start_time))),
              'train_time_lowest': train_time_lowest, 'lowest_val_loss': str(lowest_loss),
              'num_trainable_params': str(model.n_parameters)}
  utils.export_config_json(infodict, os.path.join(train_config['model_dir'], 'train_info.json'))

  tf.logging.info('Training complete')
  #tf.logging.flush() --> NotImplementedError


if __name__ == '__main__':
  app.run()
