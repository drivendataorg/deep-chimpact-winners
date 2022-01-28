import pandas as pd
import numpy as np
import ffmpeg
from joblib import Parallel, delayed
import warnings
import cv2
from tqdm import tqdm

# UPSAMPLE DATA
def upsample(df, frac=2):
    up_df = df.query("distance>=20").copy()
    up_df = up_df.sample(frac=frac, replace=True)
    up_df = pd.concat([df.query("distance<20"), up_df], axis=0)
    return up_df

# CLEAN DATA
def clean_data(df):
    df = df.query("distance>0").copy()
    return df

def load_video(filepath, fps=1):
    """Use ffmpeg to load a video as an array with a specified frame rate.

    filepath (pathlike): Path to the video
    fps (float): Desired number of frames per second
    """
    def _get_video_stream(path):
        probe = ffmpeg.probe(path)
        return next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    video_stream = _get_video_stream(filepath)
    w = int(video_stream["width"])
    h = int(video_stream["height"])

    pipeline = ffmpeg.input(filepath)
    pipeline = pipeline.filter("fps", fps=fps, round="up")
    pipeline = pipeline.output("pipe:", format="rawvideo", pix_fmt="rgb24")
    out, err = pipeline.run(capture_stdout=True, capture_stderr=True)
    arr      = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    return arr

def video2image(df, image_dir, debug=False):
    if debug: 
        df = df.iloc[:50]
    def convert(df, video_id):
        video_df = df.query("video_id==@video_id")
        video    = load_video(video_df.video_path.iloc[0], fps=1) # video path is same for all images within a video
        height, width = video.shape[1:-1]
        for time in video_df.time.unique():
            image = video[time]
            check = cv2.imwrite(f'{image_dir}/{video_id}-{time:03d}.png', image)
            if not check:
                warnings.warn(f'{video_id} writing failed')
        return [video_id, width, height]
    info = Parallel(n_jobs=-1, backend='threading', verbose=0)(delayed(convert)(df, video_id)\
                                                        for video_id in tqdm(df.video_id.unique(),
                                                                                             desc='video2image '))
    return info
