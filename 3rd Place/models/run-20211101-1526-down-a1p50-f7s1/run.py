#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Import')

import os
import sys
sys.path.append('../../lib')
import glob
import math
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import tensorflow as tf
print('tf:', tf.__version__)
import tensorflow_addons as tfa
print('tfa:', tfa.__version__)
from sklearn.metrics import mean_absolute_error, accuracy_score
import efficientnet.tfkeras as efn

from vecxoz_utils import init_tpu
from vecxoz_utils import init_tfdata
from vecxoz_utils import KeepLastCKPT
from vecxoz_utils import compute_cv_scores
from vecxoz_utils import create_submission
from vecxoz_utils import ArgumentParserExtended

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Parse CMD')

parser = ArgumentParserExtended()

parser.add_str('--tpu_ip_or_name',    default=None,          help='TPU name or GRPC address e.g. node-1 or grpc://10.70.50.202:8470 or None')
parser.add_str('--data_tfrec_dir',    default='../../data/tfrec_from_images_downsampled_step_1', help='Data directory containig TFRecord files')
parser.add_str('--data_dir',          default='../../data',  help='Data directory containig CSV files')
parser.add_str('--data_preds_dir',    default='preds',       help='Directory to save predictions')
parser.add_str('--mixed_precision',   default=None,          help='Mixed precision. E.g.: mixed_float16, mixed_bfloat16, or None')
parser.add_str('--job',               default='train_val_test_score_subm', help='Job to perform. A combination of words: train, val, test, score, subm. E.g. train_val_test')
parser.add_str('--metric_name',       default='rmse',        help='Metric name')
parser.add_str('--monitor',           default='val_loss',    help='Value to monitor during training to apply checkpoint callback, etc.')
parser.add_int('--n_folds',           default=5,             help='Number of folds')
parser.add_int('--initial_fold',      default=0,             help='Initial fold (from 0 to n_folds-1)')
parser.add_int('--n_channels',        default=3,             help='Number of image channels')
parser.add_int('--n_frames',          default=11,            help='Total number of frames (images) to read from TFRecord')
parser.add_int('--volume_size',       default=7,             help='Number of 3-channel images in a sequence (volume)')
parser.add_int('--auto',              default=-1,            help='Constant value of tf.data.experimental.AUTOTUNE. It is used to manage number of parallel calls, etc.')
parser.add_int('--dim',               default=512,           help='Image size in pixels. This final size which will be passed to a model')
parser.add_int('--dim_before_aug',    default=512,           help='Image size in pixels before augmentation. This may differ form final size passed to a model')
parser.add_bool('--image_inverted',   default=False,         help='Whether to invert the image i.e. 255 - image')
parser.add_int('--n_examples_train',  default=12183,         help='Number of training examples. This value is used to define an epoch')
parser.add_int('--n_epochs',          default=40,            help='Number of epochs to train')
parser.add_int('--batch_size',        default=96,            help='Batch size')
parser.add_bool('--drop_remainder',   default=False,         help='Whether to drop remainder')
parser.add_float('--lr',              default=1e-3,          help='Learning rate')
parser.add_str('--weights',           default='imagenet',    help='Name of pretrained weights. E.g.: imagenet, noisy-student, imagenet21k+imagenet2012, imagenet21k, or None')
parser.add_float('--aug_percentage',  default=0.5,           help='Probablity of outputting augmented image regardless of total number of augmentations')
parser.add_int('--aug_number',        default=9,             help='Number of train-time augmentations. 0 means no augmentation')
parser.add_int('--tta_number',        default=0,             help='Number of test-time augmentations. 0 means no augmentation. In result there will be (tta_number + 1) predictions')
parser.add_int('--n_classes',         default=1,             help='Number of classes for classification task. For regression task must be 1')
parser.add_int('--buffer_size',       default=256,           help='Shuffle buffer size for tf.data. For small RAM or large data use small buffer e.g. 128 or None to disable')
parser.add_bool('--use_cache',        default=False,         help='Whether to use cache for tf.data')

args = parser.parse_args() # []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Settings')

# Interim value for random number generator used with aug
args.aug_maxval = round(args.aug_number / args.aug_percentage)
# Wildcard for test TFRecord file
args.data_tfrec_test = os.path.join(args.data_tfrec_dir, 'test.tfrecord*')
# Model function
args.model_func = efn.EfficientNetB0
# Create subdir for predictions
if not os.path.exists(args.data_preds_dir):
    os.makedirs(args.data_preds_dir)

print(parser.args_repr(args, False))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

feature_description = {
    'image': tf.io.FixedLenFeature([args.n_frames], tf.string),
    'label': tf.io.FixedLenFeature([], tf.float32),
}

def fn(image):
    image = tf.image.decode_jpeg(image, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.image.resize(image, [args.dim, args.dim])
    image = tf.cast(image, dtype=tf.uint8)
    return image

def parse_example(example_proto):
    d = tf.io.parse_single_example(example_proto, feature_description)
    #
    image = tf.map_fn(fn, d['image'], dtype=tf.uint8)   # [11, 512, 512, 3]
    image = image[2:-2]
    #
    label = tf.cast(d['label'], tf.float32)
    #
    image = tf.reshape(image, [args.volume_size, args.dim, args.dim, args.n_channels])
    #
    return image, label


def aug_0(image, label):
    return image, label
def aug_1(image, label):
    return tf.image.flip_left_right(image), label
def aug_2(image, label):
    return tf.image.flip_up_down(image), label
def aug_3(image, label):
    return tfa.image.rotate(image, math.pi/4*2), label #  90
def aug_4(image, label):
    return tfa.image.rotate(image, math.pi/4*4), label # 180
def aug_5(image, label):
    return tfa.image.rotate(image, math.pi/4*6), label # 270
def aug_6(image, label):
    return tf.image.central_crop(tfa.image.rotate(image, math.pi/4*1), 0.7), label #  45
def aug_7(image, label):
    return tf.image.central_crop(tfa.image.rotate(image, math.pi/4*3), 0.7), label # 135
def aug_8(image, label):
    return tf.image.central_crop(tfa.image.rotate(image, math.pi/4*5), 0.7), label # 225
def aug_9(image, label):
    return tf.image.central_crop(tfa.image.rotate(image, math.pi/4*7), 0.7), label # 315


# args.tta_number = 0 means NO tta i.e. tta_func_list = [aug_0]
tta_func_list = [aug_0, aug_1, aug_2, aug_3, aug_4, aug_5, aug_6, aug_7, aug_8, aug_9][:args.tta_number + 1]


def aug(image, label):
    """
    Runs single transformation per call with `aug_percentage` probability
    Specific transformation probability is `aug_percentage` / `aug_number`
    """
    #
    if args.aug_number != 0:
        aug_id = tf.random.uniform([], minval=0, maxval=args.aug_maxval, dtype=tf.int32)
    #
    if args.aug_number == 0:
        pass
    elif aug_id == 0:
        image, label = aug_1(image, label)
    elif aug_id == 1:
        image, label = aug_2(image, label)
    elif aug_id == 2:
        image, label = aug_3(image, label)
    elif aug_id == 3:
        image, label = aug_4(image, label)
    elif aug_id == 4:
        image, label = aug_5(image, label)
    elif aug_id == 5:
        image, label = aug_6(image, label)
    elif aug_id == 6:
        image, label = aug_7(image, label)
    elif aug_id == 7:
        image, label = aug_8(image, label)
    elif aug_id == 8:
        image, label = aug_9(image, label)
    #
    return image, label


def norm(image, label):
    #
    image = tf.image.resize(image, [args.dim, args.dim])
    image = tf.reshape(image, [args.volume_size, args.dim, args.dim, args.n_channels])
    #
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    #
    if args.image_inverted:
        image = 255.0 - image
    #
    image = image / 255.0
    #
    return image, label

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_model(print_summary=True):
    #
    inp = tf.keras.layers.Input(shape=(args.volume_size, 
                                       args.dim, 
                                       args.dim, 
                                       args.n_channels), dtype=tf.float32)
    effnet = args.model_func(input_shape=(args.dim, args.dim, args.n_channels), 
                             weights=args.weights, include_top=False)
    pool_2d = tf.keras.layers.GlobalAveragePooling2D()
    #
    x = tf.keras.layers.TimeDistributed(effnet)(inp)
    x = tf.keras.layers.TimeDistributed(pool_2d)(x)
    x = tf.keras.layers.LSTM(1024, return_sequences=False)(x)
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    out = tf.keras.layers.Dense(args.n_classes, activation='linear')(x)
    #
    model = tf.keras.models.Model(inp, out, name='model')
    #
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), 
                  loss='mae',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name=args.metric_name)])
    if print_summary:
        model.summary()
    return model

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if args.job != 'score' and args.job != 'subm' and args.job != 'score_subm':
    for fold_id in range(args.initial_fold, args.n_folds):
        print('\n*****')
        print('Fold:', fold_id)
        print('*****\n')
        #--------------------------------------------------------------------------
        print('Clear session')
        tf.keras.backend.clear_session()
        #--------------------------------------------------------------------------
        if args.mixed_precision is not None:
            print('Init Mixed Precision:', args.mixed_precision)
            policy = tf.keras.mixed_precision.experimental.Policy(args.mixed_precision)
            tf.keras.mixed_precision.experimental.set_policy(policy)
        else:
            print('Using default PRECISION:', tf.keras.backend.floatx())
        #--------------------------------------------------------------------------
        print('FULL BATCH SHAPE: %d x %d x %d x %d x %d' % (args.batch_size, 
                                                            args.volume_size, 
                                                            args.dim, 
                                                            args.dim, 
                                                            args.n_channels))
        #--------------------------------------------------------------------------
        print('Init TPU')
        tpu, topology, strategy = init_tpu(args.tpu_ip_or_name)
        #--------------------------------------------------------------------------
        print('Init model')
        with strategy.scope():
            model = init_model(print_summary=True)
        #--------------------------------------------------------------------------
        if 'train' in args.job:
            # Globs
            all_fold_ids = np.array(range(args.n_folds))
            train_fold_ids = all_fold_ids[all_fold_ids != fold_id]
            train_glob = os.path.join(args.data_tfrec_dir, 'fold.[%d%d%d%d].tfrecord*' % tuple(train_fold_ids))
            val_glob   = os.path.join(args.data_tfrec_dir, 'fold.[%d].tfrecord*' % fold_id)
            print('TRAIN GLOB:', train_glob)
            print('VAL   GLOB:', val_glob)
            print('TEST  GLOB:', args.data_tfrec_test)
            #--------------------------------------------------------------------------
            print('Init datasets')
            train_ds = init_tfdata(train_glob, 
                                   is_train=True, 
                                   batch_size=args.batch_size, 
                                   auto=args.auto, 
                                   parse_example=parse_example, 
                                   aug=aug, 
                                   norm=norm,
                                   buffer_size=args.buffer_size,
                                   drop_remainder=args.drop_remainder,
                                   use_cache=args.use_cache)
            val_ds = init_tfdata(val_glob, 
                                 is_train=False,  
                                 batch_size=args.batch_size, 
                                 auto=args.auto,
                                 parse_example=parse_example,
                                 norm=norm,
                                 buffer_size=args.buffer_size,
                                 drop_remainder=args.drop_remainder,
                                 use_cache=args.use_cache)
            #--------------------------------------------------------------------------
            print('Init callbacks')
            call_ckpt = tf.keras.callbacks.ModelCheckpoint('model-best-f%d-e{epoch:03d}-{val_loss:.4f}-{val_%s:.4f}.h5' % (fold_id, args.metric_name),
                                                           monitor=args.monitor,
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           verbose=1)
            call_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.monitor, 
                                                                  factor=0.5, 
                                                                  patience=3, 
                                                                  min_delta=1e-4,
                                                                  min_lr=1e-8,
                                                                  verbose=1,
                                                                  mode='auto')
            call_keep_last = KeepLastCKPT(wildcard = './model-best-f%d-e*.h5' % fold_id)
            #--------------------------------------------------------------------------
            print('Fit (fold %d)' % fold_id)
            h = model.fit(
                train_ds,
                steps_per_epoch=args.n_examples_train // args.batch_size,
                epochs=args.n_epochs,
                validation_data=val_ds,
                callbacks=[call_ckpt,
                           call_reduce_lr,
                           call_keep_last]
            )
        #--------------------------------------------------------------------------
        if 'val' in args.job or 'test' in args.job:
            # Load best model for fold
            m = sorted(glob.glob('model-best-f%d*.h5' % fold_id))[-1]
            print('Load model (fold %d): %s' % (fold_id, m))
            model.load_weights(m)
            #--------------------------------------------------------------------------
            # TTA
            #--------------------------------------------------------------------------
            for tta_id in range(len(tta_func_list)):
                print('Init datasets for prediction (fold %d, tta %d)' % (fold_id, tta_id))
                #--------------------------------------------------------------------------
                if 'val' in args.job:
                    val_ds = init_tfdata(val_glob, 
                                     is_train=False,  
                                     batch_size=args.batch_size, 
                                     auto=1,
                                     parse_example=parse_example,
                                     tta=tta_func_list[tta_id],
                                     norm=norm,
                                     buffer_size=args.buffer_size,
                                     drop_remainder=args.drop_remainder,
                                     use_cache=args.use_cache)
                    print('Predict VAL (fold %d, tta %d)' % (fold_id, tta_id))
                    y_pred_val = model.predict(val_ds, verbose=1)
                    np.save(os.path.join(args.data_preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id)), y_pred_val)
                #--------------------------------------------------------------------------
                if 'test' in args.job:
                    test_ds = init_tfdata(args.data_tfrec_test, 
                                      is_train=False,  
                                      batch_size=args.batch_size, 
                                      auto=1,
                                      parse_example=parse_example,
                                      tta=tta_func_list[tta_id],
                                      norm=norm,
                                      buffer_size=args.buffer_size,
                                      drop_remainder=args.drop_remainder,
                                      use_cache=args.use_cache)
                    print('Predict TEST (fold %d, tta %d)' % (fold_id, tta_id))
                    y_pred_test = model.predict(test_ds, verbose=1)
                    np.save(os.path.join(args.data_preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id)), y_pred_test)
                #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compute val scores
#------------------------------------------------------------------------------

if 'score' in args.job:
    print('VAL scores')
    scores = compute_cv_scores(args.data_dir, args.data_preds_dir, args.n_folds, args.tta_number, print_scores=True)

#------------------------------------------------------------------------------
# Create submission
#------------------------------------------------------------------------------

if 'subm' in args.job:
    print('Create submission CSV')
    written_file_name = create_submission(args.data_dir, args.data_preds_dir, args.n_folds, args.tta_number)
    print('Submission was saved to:', written_file_name)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------




