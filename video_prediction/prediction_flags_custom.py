import tensorflow as tf
from tensorflow.python.platform import flags






def generate_flags(DATA_DIR, OUT_DIR, lr, batch_size=32, freerunning=False, num_masks=2):


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

    flags.DEFINE_integer('num_masks', num_masks,
                         'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
    schedsamp_k = 0.00001 if freerunning else 900
    flags.DEFINE_float('schedsamp_k', schedsamp_k,  # sth very close to zero, so that we don't use any gt data after conditioning
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

    flags.DEFINE_integer('batch_size', batch_size, 'batch size for evaluation')
    flags.DEFINE_float('learning_rate', lr,
                       'the base learning rate of the generator')
    flags.DEFINE_integer('custom_data', 1, ' If True (1), uses tf-record feature naming '
                         'for the bouncing_objects dataset, and loosk for the '
                         'data in separate /train and /val directories')

    return FLAGS