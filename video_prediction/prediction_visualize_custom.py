import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import matplotlib.pyplot as plt

from prediction_input_custom import build_tfrecord_input
from prediction_model import construct_model

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.utils import set_logger, ensure_dir
import src.utils.tf_utils as tf_utils
from src.train import visualize_sequence_predicted

#weights_path = './train_out/nowforreal'
#weights_path = './trained/nowforreal'
weights_path = './trained/nowforreal/18-Sep-25_23h16-47'
ckpt_id = 'model4002' #'model2'
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
    with tf.variable_scope('model', reuse=None):
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

    #print_tensors_in_checkpoint_file(ckpt_path, tensor_name='', all_tensors=True)

    saver.restore(sess, ckpt_path)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # get n_visualize predicted sequences (and target!)
    #  --  need inputs, targets, predictions. then generate plots.

    sequence_predictions = []
    sequence_inputs = []
    sequence_targets = []
    num_iter = int(np.ceil(n_visualize / FLAGS.batch_size))
    for itr in range(num_iter):
        # Generate new batch of data.
        feed_dict = {model.prefix: 'infer',
                     model.iter_num: np.float32(100)}
        inputs, prediction, costs = sess.run([images, gen_images, model.recon_costs], feed_dict)
        # --> inputs: batch_size x seq_len x h x w x c.
        #     prediction: list with 19 frames (?batch_size, h, w, c).
        plt.imshow(prediction[-1][0][...,-1], cmap='gray')
        prediction = np.stack(prediction, axis=0).transpose([1,0,2,3,4])
        # --> batch_size x seq_length x h x w x c
        targets = inputs[:, 1:]
        inputs = inputs[:, :-1]
        #inputs = inputs[:FLAGS.context_frames]
        targets_freer = inputs[FLAGS.context_frames:]
        prediction_freer = gen_images[FLAGS.context_frames - 1:]
        sequence_predictions.append(prediction)
        sequence_targets.append(targets)
        sequence_inputs.append(inputs)

    sequence_predictions = np.concatenate(sequence_predictions, axis=0)
    sequence_inputs = np.concatenate(sequence_inputs, axis=0)
    sequence_targets = np.concatenate(sequence_targets, axis=0)

    visualize_sequence_predicted(sequence_inputs, sequence_targets, sequence_predictions, max_n=n_visualize, seq_lengths=FLAGS.sequence_length, store=True, rgb=False,
                                 output_dir=OUT_DIR+'_'+ckpt_id)