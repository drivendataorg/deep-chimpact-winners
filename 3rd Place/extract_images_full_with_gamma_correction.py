#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import sys
import glob
import time
import subprocess
import numpy as np
from PIL import Image

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--in_dir', default='data/train_videos', type=str, help='Input data directory')
parser.add_argument('--white_dir', default='data/train_images_downsampled_white', type=str, help='Directory containing "white" images')
parser.add_argument('--out_dir', default='data/train_images_full_gamma', type=str, help='Output data directory')
args = parser.parse_args()


print('IN:   ', args.in_dir)
print('WHITE:', args.white_dir)
print('OUT:  ', args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.in_dir, '*.*')))


# collect white
files_white = sorted(glob.glob(os.path.join(args.white_dir, '*.*')))
white_ids = list(set([file.split('/')[-1].split('_')[0] for file in files_white]))


n_white_processed = 0
start = time.time()
for counter_file, file in enumerate(files):
    video_id = file.split('/')[-1].split('.')[0] # 'aany'
    out_template = os.path.join(args.out_dir, os.path.basename(file).split('.')[0] + '_%03d.jpg')
    if video_id in white_ids:
        n_white_processed += 1
        cmd_list = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', file, '-start_number', '0', '-r', '1', '-vf', 'eq=gamma=0.1', '-q:v', '1', out_template]
    else:
        cmd_list = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', file, '-start_number', '0', '-r', '1', '-q:v', '1', out_template]
    res_list = subprocess.run(cmd_list, stdout=subprocess.PIPE).stdout.decode('utf-8')
    #
    print('%d of %d, time: %d' % (counter_file, len(files), (time.time() - start)), end='\r')

print('n_white_processed:', n_white_processed)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------