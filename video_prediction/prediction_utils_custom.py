import tensorflow as tf



#from prediction_model import construct_model  # if this ever gives an import error, maybe try to move construct_model into .._utils_.. as well
from prediction_model_custom import CoreModel
from prediction_model_custom_resizeable import FramePredictorFinn
from prediction_model_custom_autoencoder_like import FramePredictorAutoencoderLike


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
  '''This class is not used during training. It's more or less a copy of the class used
       during training, except without the attributes only needed for training. '''

  def __init__(self, FLAGS,
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
      self.core_model = CoreModel(
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
        self.core_model = CoreModel(
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
    gen_images, gen_states = self.core_model.return_generated()
    self.prediction = self.gen_images = gen_images
    enc_dict =  {'enc'+str(i): tf.stack(self.core_model.encs[i], axis=1) for i in range(len(self.core_model.encs))}
    hidd_dict = {'hidden'+str(i+1): tf.stack(self.core_model.hidden_layers[i], axis=1) for i in range(len(self.core_model.hidden_layers))}
    self.layers = {**hidd_dict, **enc_dict, 'inputs': tf.stack(images, axis=1)}
    self.layer_names = self.layers.keys()
    self.bottleneck_layers = self.layers # todo?
    self.bottleneck_layer_names = self.bottleneck_layers.keys()
    # L2 loss
    self.recon_costs = []
    for i, x, gx in zip(
        list(range(len(gen_images))), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      self.recon_costs.append(mean_squared_error(x, gx))

  def count_parameters(self):
      """ Counts the number of trainable parameters in this model """
      self.n_parameters_ = 0
      for v in tf.trainable_variables():
          params = 1
          for s in v.get_shape():
              params *= s.value
          self.n_parameters_ += params
      return self.n_parameters_

  @property
  def n_parameters(self):
      if not hasattr(self, 'n_parameters_'):
          self.count_parameters()
      return self.n_parameters_




class ModelFinnCustom(object):
  '''This class is not used during training. It's more or less a copy of the class used
       during training, except without the attributes only needed for training. '''

  def __init__(self, train_config,
               model_config,
               images=None,
               actions=None,
               states=None,
               reuse_scope=None):

    sequence_length = train_config['max_seq_length']

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
      self.core_model = FramePredictorFinn(
          images,
          actions,
          states,
          model_config=model_config,
          iter_num=self.iter_num,
          k=train_config['schedsamp_k'],
          context_frames=train_config['context_frames'],
          schedule=train_config['freerun_schedule'])
    else:  # If it's a validation or test model.
      with tf.variable_scope(reuse_scope, reuse=True):
        self.core_model = FramePredictorFinn(
            images,
            actions,
            states,
            model_config=model_config,
            iter_num=self.iter_num,
            k=train_config['schedsamp_k'],
            context_frames=train_config['context_frames'],
            schedule=train_config['freerun_schedule'])
    gen_images, gen_states = self.core_model.return_generated()
    self.prediction = self.gen_images = gen_images
    enc_dict =  {'enc'+str(i): tf.stack(self.core_model.encs[i], axis=1) for i in range(len(self.core_model.encs))}
    hidd_dict = {'hidden'+str(i+1): tf.stack(self.core_model.hidden_layers[i], axis=1) for i in range(len(self.core_model.hidden_layers))}
    self.layers = {**hidd_dict, **enc_dict, 'inputs': tf.stack(images, axis=1)}
    if 'Finn_subtype' in model_config:
        if model_config['Finn_subtype'] == 'CDNA':
            self.layers = {**self.layers, 'kernels': tf.stack(self.core_model.kernels, axis=1)}
    self.layer_names = self.layers.keys()
    self.bottleneck_layers = self.layers # todo?
    self.bottleneck_layer_names = self.bottleneck_layers.keys()
    # L2 loss
    self.recon_costs = []
    for i, x, gx in zip(
        list(range(len(gen_images))), images[train_config['context_frames'] :],
        gen_images[train_config['context_frames'] - 1:]):
      self.recon_costs.append(mean_squared_error(x, gx))

  def count_parameters(self):
      """ Counts the number of trainable parameters in this model """
      self.n_parameters_ = 0
      for v in tf.trainable_variables():
          params = 1
          for s in v.get_shape():
              params *= s.value
          self.n_parameters_ += params
      return self.n_parameters_

  @property
  def n_parameters(self):
      if not hasattr(self, 'n_parameters_'):
          self.count_parameters()
      return self.n_parameters_




class ModelAutoencoderLike(object):
  '''This class is not used during training. It's more or less a copy of the class used
       during training, except without the attributes only needed for training. '''

  def __init__(self, train_config,
               model_config,
               images=None,
               actions=None,
               states=None,
               reuse_scope=None):

    sequence_length = train_config['max_seq_length']

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
      self.core_model = FramePredictorAutoencoderLike(
          images,
          actions,
          states,
          model_config=model_config,
          iter_num=self.iter_num,
          k=train_config['schedsamp_k'],
          context_frames=train_config['context_frames'],
          schedule=train_config['freerun_schedule'])
    else:  # If it's a validation or test model.
      with tf.variable_scope(reuse_scope, reuse=True):
        self.core_model = FramePredictorAutoencoderLike(
            images,
            actions,
            states,
            model_config=model_config,
            iter_num=self.iter_num,
            k=train_config['schedsamp_k'],
            context_frames=train_config['context_frames'],
            schedule=train_config['freerun_schedule'])
    gen_images, gen_states = self.core_model.return_generated()
    self.prediction = self.gen_images = gen_images
    enc_dict =  {'enc'+str(i): tf.stack(self.core_model.encs[i], axis=1) for i in range(len(self.core_model.encs))}
    hidd_dict = {'hidden'+str(i+1): tf.stack(self.core_model.hidden_layers[i], axis=1) for i in range(len(self.core_model.hidden_layers))}
    self.layers = {**hidd_dict, **enc_dict, 'inputs': tf.stack(images, axis=1)}
    self.layer_names = self.layers.keys()
    self.bottleneck_layers = self.layers # todo?
    self.bottleneck_layer_names = self.bottleneck_layers.keys()
    # L2 loss
    self.recon_costs = []
    for i, x, gx in zip(
        list(range(len(gen_images))), images[train_config['context_frames'] :],
        gen_images[train_config['context_frames'] - 1:]):
      self.recon_costs.append(mean_squared_error(x, gx))

  def count_parameters(self):
      """ Counts the number of trainable parameters in this model """
      self.n_parameters_ = 0
      for v in tf.trainable_variables():
          params = 1
          for s in v.get_shape():
              params *= s.value
          self.n_parameters_ += params
      return self.n_parameters_

  @property
  def n_parameters(self):
      if not hasattr(self, 'n_parameters_'):
          self.count_parameters()
      return self.n_parameters_
