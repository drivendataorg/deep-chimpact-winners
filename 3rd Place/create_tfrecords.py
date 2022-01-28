#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Script to create training examples as .npy files and TFRecord files for image-time-series tasks.
Process includes two stages:
    1) creation of Numpy arrays containing specified number of images stored as byte strings
    2) creation of TFRecord files form numpy arrays where list of images is stored as BytesList

Important: current frame is stored as MIDDLE item in the list, 
i.e. when we have 11 frames current frame is `frames[5]` i.e. `frame[(len(frames) - 1) / 2]`

by logic
-5 -4 -3 -2 -1  0  1  2  3  4  5
by indexing
 0  1  2  3  4  5  6  7  8  9  10
"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Data directory containig .csv files')
parser.add_argument('--image_dir', type=str, default='data/train_images_downsampled', help='Image dir')
parser.add_argument('--numpy_dir', type=str, default='data/train_numpy_from_images_downsampled', help='Numpy dir')
parser.add_argument('--tfrec_dir', type=str, default='data/tfrec_from_images_downsampled', help='TFRecord dir')
parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
parser.add_argument('--n_frames', type=int, default=5, help='Number of frames from each side of current frame.')
parser.add_argument('--step', type=int, default=1, help='Step between frames. Total frame depth from each sides is (n_frames * step)')
parser.add_argument('--subset', type=str, default='train', help='Subset of data ("train" or "test")')
parser.add_argument('--delete_numpy', type=str, default='False', choices=['True', 'False'], help='Whether to delete numpy arrays when TFRecords are ready')
args = parser.parse_args() # []
for a in vars(args): print('%-20s %s' % (a, vars(args)[a]))

import os
import sys
sys.path.append('lib')
import glob
import shutil
import collections
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
print('tf:', tf.__version__)
from vecxoz_utils import create_cv_split
from vecxoz_utils import TFRecordProcessor

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_npy(df):

    os.makedirs(args.numpy_dir, exist_ok=True)

    # ---- prpare path to all available frames from all videos ----
    
    files = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    # print(len(files))
    
    files_df = pd.DataFrame()
    files_df['image_path'] = files
    
    def vi_id(x):
        base_name = os.path.basename(x)
        split = base_name.split('.')[0].split('_')
        video_id = split[0]
        # frame_id = int(split[1])
        return video_id
    
    def fr_id(x):
        base_name = os.path.basename(x)
        split = base_name.split('.')[0].split('_')
        # video_id = split[0]
        frame_id = int(split[1])
        return frame_id
    
    files_df['video_id'] = files_df['image_path'].map(vi_id)
    files_df['frame_id'] = files_df['image_path'].map(fr_id)
    
    sorted_files_df = files_df.groupby('video_id')['image_path'].apply(sorted).reset_index(name='sorted_image_paths')

    # ---- select labeled frame and find corresponding previous and next frames
    
    n_examples = len(df)
    for counter, image_path in enumerate(df['image_path']):
        #
        video_id, frame_id = os.path.basename(image_path).split('.')[0].split('_')
        frame_id = int(frame_id)
        #
        # collect all available frames
        all_frame_sorted = sorted_files_df.loc[sorted_files_df['video_id'] == video_id, 'sorted_image_paths'].values[0]
        # current_frame = all_frame_sorted[frame_id]
        #
        # create ids corresponding to previous, current, and next frames
        ids = []
        # for delta in range(args.n_frames + 1):
        for delta in range(0, (args.n_frames + 1) * args.step, args.step):
            ids.append(frame_id + delta) # next
            ids.append(frame_id - delta) # previous
        ids = np.sort(np.unique(ids)) # unique to remove duplicated current frame
        assert len(ids) == (2 * args.n_frames) + 1, 'Check number of frames'
        #
        # if any id is greater or less than available id - just use copy of the available id
        max_id = len(all_frame_sorted) - 1
        min_id = 0
        ids[ids > max_id] = max_id
        ids[ids < min_id] = min_id
        #
        # select final set of frames
        selected_frames_sorted = np.array(all_frame_sorted)[ids]
        #
        # read images which will form single multi-frame example
        files_to_stack_bytes = []
        for file in selected_frames_sorted:
            with open(file, 'rb') as f:
                files_to_stack_bytes.append(f.read())
        #
        # save
        np.save(os.path.join(args.numpy_dir, os.path.basename(image_path).replace('.jpg', '.npy')), np.array(files_to_stack_bytes))
        #
        print('%d of %d' % (counter, n_examples), end='\r')
        # break

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Loading .csv...')

train_df, test_df = create_cv_split(args.data_dir, args.n_folds)

if args.subset == 'train':
    df = train_df.copy()
else:
    df = test_df.copy()

print('Creating .npy...')

create_npy(df)

print('Creating TFRecord...')

os.makedirs(args.tfrec_dir, exist_ok=True)

# Create paths to numpy arrays
df['image_path'] = args.numpy_dir + '/' + df['image_id'].str.replace('.jpg', '.npy', regex=False)

tfrp = TFRecordProcessor()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if args.subset == 'train':
    for fold_id in range(args.n_folds):
        print('Fold:', fold_id)
        n_written = tfrp.write_tfrecords(
            df[df['fold_id'] == fold_id]['image_id'].values,
            df[df['fold_id'] == fold_id]['image_path'].values,
            df[df['fold_id'] == fold_id]['distance'].values,
            df[df['fold_id'] == fold_id]['distance_le'].values,
            #
            n_shards=8, 
            file_out=os.path.join(args.tfrec_dir, 'fold.%d.tfrecord' % fold_id),)
else:
    n_written = tfrp.write_tfrecords(
        df['image_id'].values,
        df['image_path'].values,
        df['distance'].values,
        df['distance_le'].values,
        #
        n_shards=16, 
        file_out=os.path.join(args.tfrec_dir, 'test.tfrecord'),)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if args.delete_numpy == 'True':
    print('Removing Numpy...')
    shutil.rmtree(args.numpy_dir, ignore_errors=True)

print('DONE')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


