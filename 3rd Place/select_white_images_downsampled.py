#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import os
import glob
import sys
sys.path.append('lib')
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from vecxoz_utils import create_cv_split

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('--data_dir', default='data', type=str, help='Data directory')
parser.add_argument('--out_dir', default='data/train_images_downsampled_white', type=str, help='Output directory')
parser.add_argument('--subset', default='train', type=str, help='Subset "train" or "test"')
args = parser.parse_args()


os.makedirs(args.out_dir, exist_ok=True)

train_df, test_df = create_cv_split(args.data_dir, 5)

if args.subset == 'train':
    df = train_df.copy()
else:
    df = test_df.copy()


n_whites = 0
for counter, file in enumerate(df['image_path'].values):
    image = np.array(Image.open(file))
    ratio = ((image > 250).sum() / image.size)
    if ratio > 0.2:
        shutil.copy(file, args.out_dir)
        n_whites += 1
    print('%d - %d' % (counter, n_whites), end='\r')


# 15228 - 1340
# 11931 - 1183

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
