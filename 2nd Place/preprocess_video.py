import argparse
from pathlib import Path

import albumentations as A
import ffmpeg
import cv2
import numpy as np
import pandas as pd
import tqdm

HEIGHT = 270
WIDTH = 480
FPS = 1
WINDOW_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to csv file",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Path to directory with video files",
    )
    parser.add_argument(
        "--save-image-dir",
        type=str,
        required=True,
        help="Path to directory with images to save",
    )
    parser.add_argument(
        "--save-csv-name",
        type=str,
        required=True,
        help="Filename of csv to save",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        required=False,
        default=WINDOW_SIZE,
        help="Window size",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=HEIGHT,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        default=WIDTH,
        help="Image width",
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=False,
        default=FPS,
        help="Video fps",
    )

    args = parser.parse_args()

    return args


def load_video(filepath, fps):
    """Use ffmpeg to load a video as an array with a specified frame rate.

    filepath (pathlike): Path to the video
    fps (float): Desired number of frames per second
    """

    def _get_video_stream(path):
        probe = ffmpeg.probe(path)
        
        return next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
        )

    video_stream = _get_video_stream(filepath)
    w = int(video_stream["width"])
    h = int(video_stream["height"])

    pipeline = ffmpeg.input(filepath)
    pipeline = pipeline.filter("fps", fps=fps, round="up")
    pipeline = pipeline.output("pipe:", format="rawvideo", pix_fmt="rgb24")
    out, err = pipeline.run(capture_stdout=True, capture_stderr=True)
    arr = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])

    return arr


def main():
    args = parse_args()
    print(args)

    df = pd.read_csv(args.csv_path)
    df = df.groupby("video_id").agg(list).reset_index()

    resize_fn = A.Resize(height=args.height, width=args.width)

    path_to_videos = Path(args.video_dir)
    path_to_save = Path(args.save_image_dir)
    path_to_save.mkdir(exist_ok=True, parents=True)

    dataset = []
    with tqdm.tqdm(df.itertuples(), total=len(df)) as pbar:
        for item in pbar:
            filepath = path_to_videos / item.video_id
            video_array = load_video(filepath, fps=args.fps)

            assert len(item.time) == len(item.distance)

            for t, d in zip(item.time, item.distance):
                for i in range(1, args.window_size + 1):
                    s = t - i
                    if s >= 0:
                        save_img_name = f"{filepath.stem}_{s}.png"
                        save_fpath = path_to_save / save_img_name
                        if not save_fpath.is_file():
                            img = resize_fn(image=video_array[s])['image']
                            cv2.imwrite(str(save_fpath), img)

                    s = t + i
                    if s < len(video_array):
                        save_img_name = f"{filepath.stem}_{s}.png"
                        save_fpath = path_to_save / save_img_name
                        if not save_fpath.is_file():
                            img = resize_fn(image=video_array[s])['image']
                            cv2.imwrite(str(save_fpath), img)
                
                save_img_name = f"{filepath.stem}_{t}.png"
                dataset.append((save_img_name, d))
                save_fpath = path_to_save / save_img_name
                if not save_fpath.is_file():
                    img = resize_fn(image=video_array[t])['image']
                    cv2.imwrite(str(save_fpath), img)
        
    dataset = pd.DataFrame(dataset, columns=['fpath', 'distance'])
    dataset.to_csv(path_to_save.parent / args.save_csv_name, index=False)


if __name__ == "__main__":
    main()
