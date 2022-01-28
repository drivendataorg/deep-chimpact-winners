#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import math
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ArgumentParserExtended(ArgumentParser):
    """
    The main purpose of this class is to standardize and simplify definition of arguments
    and allow processing of True, False, and None values.
    There are 4 types of arguments (bool, int, float, str). All accept None.
    
    Usage:

    parser = ArgumentParserExtended()
    
    parser.add_str('--str', default='/home/user/data')
    parser.add_int('--int', default=220)
    parser.add_float('--float', default=3.58)
    parser.add_bool('--bool', default=True)
    
    args = parser.parse_args()
    print(parser.args_repr(args, True))
    """

    def __init__(self, *args, **kwargs):
        super(ArgumentParserExtended, self).__init__(*args, **kwargs)

    def bool_none_type(self, x):
        if x == 'True':
            return True
        elif x == 'False':
            return False
        elif x == 'None':
            return None
        else:
            raise ValueError('Unexpected literal for bool type')

    def int_none_type(self, x):
        return None if x == 'None' else int(x)

    def float_none_type(self, x):
        return None if x == 'None' else float(x)

    def str_none_type(self, x):
        return None if x == 'None' else str(x)

    def add_str(self, name, default=None, choices=None, help='str or None'):
        """
        Returns str or None
        """
        _ = self.add_argument(name, type=self.str_none_type, default=default, choices=choices, help=help)

    def add_int(self, name, default=None, choices=None, help='int or None'):
        """
        Returns int or None
        'hello' or 'none' or 1.2 will cause an error
        """
        _ = self.add_argument(name, type=self.int_none_type, default=default, choices=choices, help=help)

    def add_float(self, name, default=None, choices=None, help='float or None'):
        """
        Returns float or None
        'hello' or 'none' will cause an error
        """
        _ = self.add_argument(name, type=self.float_none_type, default=default, choices=choices, help=help)

    def add_bool(self, name, default=None, help='bool'):
        """
        Returns True, False, or None
        Anything except 'True' or 'False' or 'None' will cause an error

        `choices` are checked after type conversion of argument passed in fact
            i.e. `choices` value must be True instead of 'True'
        Default value is NOT checked using `choices`
        Default value is NOT converted using `type`
        """
        _ = self.add_argument(name, type=self.bool_none_type, default=default, choices=[True, False, None], help=help)

    @staticmethod
    def args_repr(args, print_types=False):
        ret = ''
        props = vars(args)
        keys = sorted([key for key in props])
        vals = [str(props[key]) for key in props]
        max_len_key = len(max(keys, key=len))
        max_len_val = len(max(vals, key=len))
        if print_types:
            for key in keys:
                ret += '%-*s  %-*s  %s\n' % (max_len_key, key, max_len_val, props[key], type(props[key]))
        else:   
            for key in keys:
                ret += '%-*s  %s\n' % (max_len_key, key, props[key])
        return ret.rstrip()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tpu(tpu_ip_or_name=None):
    """
    Initializes `TPUStrategy` or appropriate alternative.
    IMPORTANT: Run this init before init of `tf.data`

    tpu_ip_or_name : str or None
        e.g. 'grpc://10.70.50.202:8470' or 'node-1'

    Usage:
    
    tpu, topology, strategy = init_tpu('node-1')
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_ip_or_name)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('--> Master:      ', tpu.master())
        print('--> Num replicas:', strategy.num_replicas_in_sync)
        return tpu, topology, strategy
    except:
        print('--> TPU was not found!')
        # strategy = tf.distribute.get_strategy() # CPU or single GPU
        strategy = tf.distribute.MirroredStrategy() # GPU or multi-GPU
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # clusters of multi-GPU
        print('--> Num replicas:', strategy.num_replicas_in_sync)
        return None, None, strategy

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tfdata(files_glob, is_train, batch_size, auto, parse_example, 
                mod=None, aug=None, tta=None, norm=None, 
                buffer_size=2048, use_cache=True, drop_remainder=False):
    """
    Creates tf.data.TFRecordDataset with appropriate train/test parameters 
    depending on `is_train` argument.
    
    files_glob : str
        Glob wildcard for TFRecord files
    is_train : bool
        if is_train == True:
            deterministic = False
            shuffle TFRec files
            apply AUG
            do NOT apply TTA
            repeat
            shuffle examples
            do NOT cache
    batch_size : int
    auto : int
    parse_example, mod, aug, tta, norm : callable
        Processing functions
    buffer_size : int or None
        Shuffle buffer size
    use_cache : bool
        Whether to cache data
    drop_remainder : bool
        Whether to drop remainder (incomplete batch)

    Example:
        train_ds = init_tfdata(train_glob, 
                               is_train=True,  
                               batch_size=args.batch_size, 
                               auto=args.auto,
                               parse_example=parse_example, 
                               aug=aug, 
                               norm=norm)
        val_ds = init_tfdata(val_glob, 
                             is_train=False,  
                             batch_size=args.batch_size, 
                             auto=args.auto,
                             parse_example=parse_example,
                             norm=norm)
    """
    options = tf.data.Options()
    options.experimental_deterministic = not is_train
    files = tf.data.Dataset.list_files(files_glob, shuffle=is_train).with_options(options)
    #
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto)
    ds = ds.with_options(options)
    ds = ds.map(parse_example, num_parallel_calls=auto)
    if mod is not None:
        ds = ds.map(mod, num_parallel_calls=auto)
    if is_train and aug is not None:
        ds = ds.map(aug, num_parallel_calls=auto)
    if not is_train and tta is not None:
        ds = ds.map(tta, num_parallel_calls=auto)
    if norm is not None:
        ds = ds.map(norm, num_parallel_calls=auto)
    if is_train:
        ds = ds.repeat()
        if buffer_size is not None:
            ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(auto)
    if not is_train and use_cache:
        ds = ds.cache()
    #
    return ds

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class KeepLastCKPT(tf.keras.callbacks.Callback):
    """
    Sort all ckpt files matching the wildcard and remove all except last.
    If there is only one ckpt file it will not be removed.
    If save_best_only=True in ModelCheckpoint and 
        naming is consistent e.g. "model-best-f0-e001-25.3676.h5"
        then KeepLastCKPT will keep OVERALL best ckpt
    
    NOTE:
    Methods `on_epoch_end` and `on_test_end` are called before last ckpt is created.
    Order of callbacks in the list passed to `model.fit` does not affect this behavor.
    """
    #
    def __init__(self, wildcard):
        super(KeepLastCKPT, self).__init__()
        self.wildcard = wildcard
    #
    def on_epoch_begin(self, epoch, logs=None):
        # files = sorted(glob.glob(self.wildcard))
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                # os.remove(file)
                tf.io.gfile.remove(file)
            print('Kept ckpt: %s' % files[-1])
        else:
            print('No ckpt to keep')
    #
    def on_train_end(self, logs=None):
        # files = sorted(glob.glob(self.wildcard))
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                # os.remove(file)
                tf.io.gfile.remove(file)
            print('\nKept ckpt (final): %s' % files[-1])
        else:
            print('\nNo ckpt to keep (final)')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_cv_split(data_dir, n_folds):
    '''
    train_df, test_df = create_cv_split(args.data_dir, args.n_folds)
    '''
    print('Load csv')    
    train_df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'submission_format.csv'))
    #--------------------------------------------------------------------------

    print('Create features')
    # In-place fix
    # For some reason "auxd_060.jpg" does not exist - we use "auxd_059.jpg" instead
    # To avoid changes in the downstream processing we just modify original time
    train_df.loc[train_df['video_id'].str.contains('auxd') & (train_df['time'] == 60), 'time'] = 59
    
    def create_image_id(row):
         return '%s_%03d.jpg' % (row['video_id'].split('.')[0], row['time']) # 'aany_000.jpg'
    
    train_df['image_id'] = train_df.apply(create_image_id, axis=1)
    test_df['image_id'] = test_df.apply(create_image_id, axis=1)
    
    train_df['image_path'] = data_dir + '/train_images_downsampled/' + train_df['image_id']
    test_df['image_path']  = data_dir + '/test_images_downsampled/'  + test_df['image_id']
    
    le = LabelEncoder()
    le = le.fit(list(train_df['distance']))
    train_df['distance_le'] = le.transform(train_df['distance'])
    # Check if consequtive and starts from 0
    lst = list(train_df['distance_le'])
    assert list(np.sort(np.unique(lst))) == list(range(0, max(lst)+1)), 'Non-consequtive, or starts not from 0'
    
    # Template column for fold_id
    train_df['fold_id'] = 0
    test_df['fold_id'] = 0
    # Fake columns (just for compatibility)
    test_df['distance'] = 0.0
    test_df['distance_le'] = 0
    
    print('train:', train_df.shape) # (15229, 7)
    print('test: ', test_df.shape)  # (11932, 7)
    #--------------------------------------------------------------------------
    
    print('Create split')
    gkf = GroupKFold(n_splits=n_folds)
    for fold_id, (train_index, val_index) in enumerate(gkf.split(train_df, groups=train_df['video_id'].values)):
        train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
    # Check
    assert len(train_df['fold_id'].unique()) == n_folds, 'Inconsistent number of splits'
    for i in range(n_folds):
        assert train_df[train_df['fold_id'] == i]['video_id'].isin(train_df[train_df['fold_id'] != i]['video_id']).sum() == 0, 'Groups are intersected'
    # Shuffle
    train_df = train_df.sample(frac=1.0, random_state=33)

    return train_df, test_df

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def compute_cv_scores(data_dir, preds_dir, n_folds, tta_number, print_scores=True):
    # Load csv
    train_df, _ = create_cv_split(data_dir, n_folds)
    
    # Collect all preds
    all_tta = []
    for tta_id in range(tta_number + 1):
        all_folds = []
        for fold_id in range(n_folds):
            all_folds.append(np.load(os.path.join(preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel())
        all_tta.append(np.hstack(all_folds))
    
    # Collect coresponding true label
    y_true_list = []
    for fold_id in range(n_folds):
        y_true_list.append(train_df.loc[train_df['fold_id'] == fold_id, 'distance'].values.ravel())
    y_true = np.hstack(y_true_list)
    
    # Compute score for original image and each TTA
    scores = []
    for tta_id, y_pred in enumerate(all_tta):
        score = mean_absolute_error(y_true, y_pred)
        scores.append(score)
        if print_scores:
            print('TTA %d score: %.4f' % (tta_id, score))

    # Compute score for mean of all TTA
    score = mean_absolute_error(y_true, np.mean(all_tta, axis=0))
    scores.append(score)
    if print_scores:
        print('-------------------')
        print('MEAN of all: %.4f' % score)

    return scores

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_submission(data_dir, preds_dir, n_folds, tta_number, file_name=None):
    if file_name is None:
        file_name = os.getcwd().split('/')[-1][:17] + '.csv'
    # Load csv
    subm_df = pd.read_csv(os.path.join(data_dir, 'submission_format.csv'))

    # Collect test preds
    y_preds_test = []
    for tta_id in range(tta_number + 1):
        for fold_id in range(n_folds):
            y_preds_test.append(np.load(os.path.join(preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel())
    
    # Write submission
    subm_df['distance'] = np.mean(y_preds_test, axis=0)
    subm_df.to_csv(file_name, index=False)
    
    return file_name

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class TFRecordProcessor(object):
    #
    def __init__(self):
        self.n_examples = 0
    #
    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    #
    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    #
    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    #
    def _process_example(self, ind, X_id, X_name, Y, Y_cls):
        self.n_examples += 1
        feature = collections.OrderedDict()
        #
        feature['image_id'] = self._bytes_feature([X_id[ind].encode('utf-8')])
        feature['image']    = self._bytes_feature( list(np.load(X_name[ind])) )
        feature['label']    = self._float_feature(Y[ind])
        feature['label_le'] = self._int_feature(Y_cls[ind])
        #
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        #
        self._writer.write(example_proto.SerializeToString())
    #
    def write_tfrecords(self, X_id, X_name, Y, Y_cls, n_shards=1, file_out='train.tfrecord'):
        n_examples_per_shard = X_name.shape[0] // n_shards
        n_examples_remainder = X_name.shape[0] %  n_shards   
        self.n_examples = 0
        #
        for shard in range(n_shards):
            self._writer = tf.io.TFRecordWriter('%s-%05d-of-%05d' % (file_out, shard, n_shards))
            #
            start = shard * n_examples_per_shard
            if shard == (n_shards - 1):
                end = (shard + 1) * n_examples_per_shard + n_examples_remainder
            else:
                end = (shard + 1) * n_examples_per_shard
            #
            print('Shard %d of %d: (%d examples)' % (shard, n_shards, (end - start)))
            for i in range(start, end):
                self._process_example(i, X_id, X_name, Y, Y_cls)
                print(i, end='\r')
            #
            self._writer.close()
        #
        return self.n_examples

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

