#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import sys
import glob
import time
import numpy as np
from PIL import Image
import ffmpeg

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('--in_dir', default='data/train_videos_downsampled', type=str, help='Input data directory')
parser.add_argument('--out_dir', default='data/train_images_downsampled', type=str, help='Output data directory')
args = parser.parse_args()


def load_video(filepath, fps):
    #
    def _get_video_stream(path):
        probe = ffmpeg.probe(path)
        return next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    #
    video_stream = _get_video_stream(filepath)
    w = int(video_stream['width'])
    h = int(video_stream['height'])
    #
    pipeline = ffmpeg.input(filepath)
    pipeline = pipeline.filter('fps', fps=fps, round='up')
    pipeline = pipeline.output('pipe:', format='rawvideo', pix_fmt='rgb24')
    out, err = pipeline.run(capture_stdout=True, capture_stderr=True)
    arr = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    #
    return arr


print('IN: ', args.in_dir)
print('OUT:', args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.in_dir, '*.*')))


start = time.time()
for counter_file, file in enumerate(files):
    frames = load_video(file, fps=1)
    for counter_frame, frame in enumerate(frames):
        p = os.path.join(args.out_dir, '%s_%03d%s' % (os.path.basename(file).split('.')[0], counter_frame, '.jpg'))
        Image.fromarray(frame).save(p)
    print('%d of %d, time: %d' % (counter_file, len(files), (time.time() - start)), end='\r')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

