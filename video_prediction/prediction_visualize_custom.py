import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt

from prediction_input_custom import build_tfrecord_input
from prediction_model import construct_model

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.utils import set_logger, ensure_dir
import src.utils.tf_utils as tf_utils

weights_path = './trained/nowforreal'
ckpt_id = None #'model2'
freerunning = True
n_visualize = 10
#DATA_DIR = '/home/noobuntu/Sema2018/data/robots_pushing/push/push_train'    #'push/push_testnovel' # 'push/push_train'   # '../../../../data/bouncing_circles/short_sequences/static_simple_1_bcs'
#DATA_DIR = '../../../../data/gen/debug_bouncing_circles/static_simple_2_bcs/tfrecords'  # <- for VM on windows
DATA_DIR = '../../../../data/gen/bouncing_circles/short_sequences/static_simple_1_bcs'
#DATA_DIR = '../../../../data/bouncing_circles/short_sequences/static_simple_1_bcs'
#DATA_DIR = '../../../../data/robots_pushing/push/push_train' # 'push/push_train'

# local output directory
OUT_DIR = './vis/'+weights_path.strip('/.')


# todo: this would be so much nicer if the parameters were stored as config somewhere. Ah wait I can do that!
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
#flags.DEFINE_integer('num_iterations', 4005, 'number of training iterations.')
#flags.DEFINE_string('pretrained_model', '',
#                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 20,
                     'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 0,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA',
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 2,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 0.00001,  # sth very close to zero, so that we don't use any gt data after conditioning
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

flags.DEFINE_integer('batch_size', 1, 'batch size for evaluation')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')
flags.DEFINE_integer('custom_data', 1, ' If True (1), uses tf-record feature naming '
                     'for the bouncing_objects dataset, and loosk for the '
                     'data in separate /train and /val directories')



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
    # L2 loss
    self.recon_costs = []
    for i, x, gx in zip(
        list(range(len(gen_images))), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      self.recon_costs.append(mean_squared_error(x, gx))






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == '__main__':

    # * *  load val-2 split data * * * * * * * * * *
    images, actions, states = build_tfrecord_input(split_string='val', file_nums=[2])
    # * *  build the model * * * * * * * * * *
    #gen_images, gen_states = construct_model(
    #    images,
    #    actions,
    ###    states,
     #   iter_num=-1,
     #   k=0.00001,
     #   use_state=FLAGS.use_state,
     #   num_masks=FLAGS.num_masks,
     #   cdna=FLAGS.model == 'CDNA',
     #   dna=FLAGS.model == 'DNA',
     #   stp=FLAGS.model == 'STP',
     #   context_frames=FLAGS.context_frames)
    model = Model(images, actions, states, FLAGS.sequence_length)
    gen_images = model.gen_images
    # * *  take n_visualize random samples (automatically) and predict it * * * * * * * * * *
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
    ensure_dir(OUT_DIR)
    set_logger(OUT_DIR+'vis_log.txt')
    # Make training session.
    sess = tf.InteractiveSession()

    if ckpt_id is not None:
        ckpt_id = tf_utils.ckpt_starting_with(ckpt_id, weights_path)
        ckpt_path = os.path.join(os.path.abspath(weights_path), ckpt_id)
    else:
        ckpt_path = tf.train.latest_checkpoint(weights_path)
    saver.restore(sess, ckpt_path)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # get n_visualize predicted sequences (and target!)
    #  --  need inputs, targets, predictions. then generate plots.
    num_iter = int(np.ceil(n_visualize / FLAGS.batch_size))
    for itr in range(num_iter):
        # Generate new batch of data.
        feed_dict = {}
        inputs, prediction = sess.run([images, gen_images], feed_dict)
        plt.imshow(prediction[0][-1], cmap='gray')

        targets = inputs[1:]
        #inputs = inputs[:FLAGS.context_frames]
        targets_freer = inputs[FLAGS.context_frames:]
        prediction_freer = gen_images[FLAGS.context_frames - 1:]
