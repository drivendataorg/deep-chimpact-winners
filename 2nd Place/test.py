import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-df",
        type=str,
        default="./data/test.csv",
        help="path to train df",
    )
    parser.add_argument(
        "--sub-format-df",
        type=str,
        default="./data/submission_format.csv",
        help="path to train df",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        help="path to train df",
        default="./data/test_images",
    )
    parser.add_argument(
        "--load", type=str, required=True, help="path to pretrained model weights"
    )
    parser.add_argument(
        "--save-sub", type=str, required=True, help="path to save submission"
    )

    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=64)
    parser.add_argument("--window-size", type=int, help="batch size", default=5)
    parser.add_argument("--tta", type=int, help="test time augmentation", default=1)

    parser.add_argument(
        "--local_rank",
        type=int,
        help="local rank",
        default=0,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu",
        default=0,
    )
    parser.add_argument("--use-log", action="store_true", help="logarithmate targets")

    args = parser.parse_args()

    return args


def imread(path):
    img = cv2.imread(str(path))

    return img

def normalize(img):
    img = np.transpose(img, (2, 0, 1))
    img = img.astype("float32") / 255
    img = torch.from_numpy(img)

    return img


class Chimpact(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, window_size=0):
        self.df = df
        self.img_dir = img_dir
        self.window_size = window_size
        self.order = [f"image_{i}" for i in range(1, window_size + 1)]
        self.order.append("image")
        self.order.extend([f"image{i}" for i in range(1, window_size + 1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        img_path = self.img_dir / item.fpath
        img = imread(img_path)
        inputs = {"image": img}

        if self.window_size > 0:
            stem, time = img_path.stem.split("_")
            time = int(time)
            suffix = img_path.suffix
            for i in range(1, self.window_size + 1):
                s = time - i
                if s >= 0:
                    new_stem = "_".join([stem, str(s)])
                    new_name = f"{new_stem}{suffix}"
                    new_path = img_path.with_name(new_name)
                    assert new_path.is_file(), new_path
                    inputs[f"image_{i}"] = imread(new_path)
                else:
                    inputs[f"image_{i}"] = np.zeros_like(img)

                s = time + i
                new_stem = "_".join([stem, str(s)])
                new_name = f"{new_stem}{suffix}"
                new_path = img_path.with_name(new_name)
                if new_path.is_file():
                    inputs[f"image{i}"] = imread(new_path)
                else:
                    inputs[f"image{i}"] = np.zeros_like(img)

        img = np.concatenate([inputs[i] for i in self.order], axis=-1)

        img = normalize(img)

        target = item.fpath
        video_id, time = target.split("_")
        time, _ = time.split(".")

        assert video_id == item.video_id.split(".")[0]
        assert time == str(item.time)

        video_id = item.video_id
        time = item.time

        return img, video_id, time


def test(args):
    assert 1 <= args.tta <= 2, args.tta

    torch.backends.cudnn.benchmark = True

    device = "cuda"
    model_paths = sorted(Path(args.load).rglob("model_best.pt"))
    print(model_paths)
    models = [
        torch.jit.load(p, map_location=device).eval()
        for p in model_paths
    ]
    print(len(models))

    test_df = pd.read_csv(args.test_df)
    sub_df = pd.read_csv(args.sub_format_df)
    test_df = pd.concat([test_df, sub_df], axis=1)
    print(test_df)

    test_images_dir = Path(args.test_images_dir)
    test_dataset = Chimpact(
        df=test_df,
        img_dir=test_images_dir,
        window_size=args.window_size,
    )
    test_sampler = None

    args.num_workers = min(args.batch_size, 8)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    submission = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as pbar:
            distance: torch.Tensor = torch.zeros(args.batch_size, device=device)
            for image, video_id, time in pbar:
                bs = image.size(0)
                image = image.to(device, non_blocking=True)

                distance.zero_()
                for model in models:
                    logits = model(image)
                    logits = logits.squeeze(1)
                    distance[:bs] += logits

                    if args.tta > 1:
                        logits = model(torch.flip(image, dims=[-1]))
                        logits = logits.squeeze(1)
                        distance[:bs] += logits

                distance /= len(models) * args.tta

                if args.use_log:
                    distance.expm1_()

                time = time.numpy()

                for d, vid, t in zip(distance.cpu().numpy(), video_id, time):
                    submission.append((vid, t, d))

    submission = pd.DataFrame(submission, columns=sub_df.columns)
    submission.to_csv(args.save_sub, index=False)
    assert submission.shape == sub_df.shape


def main():
    args = parse_args()
    if args.local_rank == 0:
        print(args)

    test(args)


if __name__ == "__main__":
    main()
