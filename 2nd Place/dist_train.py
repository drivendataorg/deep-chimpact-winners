import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import random
from pathlib import Path

import albumentations as A
import apex
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import numpy as np
import pandas as pd
import timm
import torch
import torch.distributed
import torch.nn as nn
import torch.utils
import torch.utils.data
import tqdm
from sklearn.model_selection import KFold
from torch.utils.tensorboard.writer import SummaryWriter


HEIGHT = 270
WIDTH = 480
IMG_SIZE = (HEIGHT, WIDTH)


def str2tuple(a):
    h, w = a.split(",")
    h = int(h)
    w = int(w)

    return h, w


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-df",
        type=str,
        default="./data/train.csv",
        help="path to train df",
    )
    parser.add_argument(
        "--train-images-dir",
        type=str,
        help="path to train df",
        default="./data/train_images",
    )
    parser.add_argument(
        "--test-df",
        type=str,
        default=None,
        help="path to test df",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        help="path to test df",
        default=None,
    )
    parser.add_argument("--checkpoint-dir", type=str, default="logs")
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b5")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--n-classes", type=int, default=1)
    parser.add_argument("--optim", type=str, default="fusedadam", help="optimizer name")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--scheduler", type=str, default="cosa", help="scheduler name")
    parser.add_argument("--scheduler-mode", type=str, default="epoch", choices=["step", "epoch"], help="scheduler mode")
    parser.add_argument("--T-max", type=int, default=25)
    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=1005
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=10,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="fold",
        default=0,
    )

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
    parser.add_argument(
        "--distributed", action="store_true", help="distributed training"
    )
    parser.add_argument("--syncbn", action="store_true", help="sync batchnorm")
    parser.add_argument(
        "--deterministic", action="store_true", help="deterministic training"
    )
    parser.add_argument(
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument(
        "--channels-last", action="store_true", help="Use channels_last memory layout"
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--use-log", action="store_true", help="logarithmate targets")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--img-size", type=str2tuple, default=IMG_SIZE)

    args = parser.parse_args()

    return args


def imread(path):
    img = cv2.imread(str(path))

    return img


def get_transforms(is_train, window_size, img_size=None, p=0.5):
    if not is_train and img_size is None:
        return

    if window_size > 0:
        additional_targets = {f"image{i}": "image" for i in range(1, window_size + 1)}
        additional_targets.update({f"image_{i}": "image" for i in range(1, window_size + 1)})
    else:
        additional_targets = None

    if is_train:
        augs = [
            A.HorizontalFlip(p=p),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=p),
            A.OneOf(
               [
                   A.Blur(p=1),
                   A.GlassBlur(p=1),
                   A.GaussianBlur(p=1),
                   A.MedianBlur(p=1),
                   A.MotionBlur(p=1),
               ],
               p=p,
            ),
            A.RandomBrightnessContrast(p=p),
            A.OneOf(
               [
                   A.RandomGamma(p=1),  # works only for uint
                   A.ColorJitter(p=1),
                   A.RandomToneCurve(p=1),  # works only for uint
               ],
               p=p,
            ),
            A.OneOf(
               [
                   A.GaussNoise(p=1),
                   A.MultiplicativeNoise(p=1),
               ],
               p=p,
            ),
            A.OneOf(
                [
                    A.PiecewiseAffine(),
                    A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT),
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            A.FancyPCA(p=0.2),
            A.RandomFog(p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(src_radius=150, p=0.2),
            A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, p=0.2),
        ]
    else:
        augs = []

    if img_size is not None:
        augs.append(A.PadIfNeeded(*img_size))

    albu_train = A.Compose(augs, additional_targets=additional_targets)

    return albu_train


def normalize(img):
    img = np.transpose(img, (2, 0, 1))

    img = img.astype("float32") / 255
    img = torch.from_numpy(img)

    return img


class Chimpact(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None, window_size=0):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.window_size = window_size
        self.order = [f"image_{i}" for i in range(1, window_size + 1)]
        self.order.append("image")
        self.order.extend([f"image{i}" for i in range(1, window_size + 1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
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

            if self.transform is not None:
                inputs = self.transform(**inputs)

            imgs = [inputs[i] for i in self.order]
            img = np.concatenate(imgs, axis=-1)

            img = normalize(img)

            target = item.distance

            return img, target
        except:
            return self.__getitem__(index)


class MAEMetric:
    def __init__(self, from_log=True):
        self.from_log = from_log

        self.clear()

    def update(self, logits, target):
        logits = logits.squeeze(1)
        if self.from_log:
            logits = torch.expm1(logits)

        self.ae += torch.abs(logits - target).sum().item()
        self.n += logits.size(0)

    def eval(self):
        if self.n > 0:
            mae = self.ae / self.n
        else:
            mae = 0.0

        return mae, self.ae, self.n

    def clear(self):
        self.ae = 0.0
        self.n = 0.0


def epoch_step(
    loader, desc, model, criterion, metrics, optimizer=None, scheduler=None, scaler=None, local_rank=0,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    loc_loss = n = 0
    for i, (image, target) in enumerate(loader):
        if is_train:
            optimizer.zero_grad()  # set_to_none=True)

        image = image.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(image)
                loss = criterion(logits, target)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )  # , error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(image)
            loss = criterion(logits, target)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )  # , error_if_nonfinite=False)
                optimizer.step()

        bs = image.size(0)
        loc_loss += loss.item() * bs
        n += bs

        for metric in metrics:
            metric.update(logits, target)

        torch.cuda.synchronize()

        if scheduler is not None:
            scheduler.step()

        if local_rank == 0:
            postfix = {
                "loss": f"{loc_loss / n:.3f}",
            }
            pbar.set_postfix(**postfix)
            pbar.update()

    if local_rank == 0:
        pbar.close()

    return loc_loss, n


class MAEwithLog(nn.Module):
    def __init__(self, use_log=True):
        super().__init__()

        self.mae = torch.nn.L1Loss()
        self.use_log = use_log

    def forward(self, logits, target):
        logits = logits.squeeze(1)
        if self.use_log:
            target = torch.log1p(target)

        loss = self.mae(logits, target)

        return loss


def train_dev_split(df, args):
    df = df.copy()
    df['video_id'] = df.fpath.str.split('_').str[0]
    df = df.groupby('video_id').agg(list).reset_index()
    df["fold"] = None

    n_col = len(df.columns) - 1
    skf = KFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.random_state
    )
    for fold, (_, dev_index) in enumerate(skf.split(df)):
        df.iloc[dev_index, n_col] = fold

    df = pd.concat([df.fpath.explode(), df[["distance", "fold"]].explode("distance")], axis=1)

    train, dev = (
        df[df.fold != args.fold].copy(),
        df[df.fold == args.fold].copy(),
    )

    return train, dev


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def save_jit(model, args, model_path):
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    if args.encoder_name.startswith("efficientnet-b"):
        model.set_swish(memory_efficient=False)

    inp = torch.rand(1, args.in_channels * (2 * args.window_size + 1), args.img_size[0], args.img_size[1]).cuda()

    with torch.no_grad():
        traced_model = torch.jit.trace(model, inp)

    traced_model.save(model_path)

    if args.encoder_name.startswith("efficientnet-b"):
        model.set_swish(memory_efficient=True)


def all_gather(value, n, is_dist):
    if is_dist:
        vals = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(vals, value)
        ns = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(ns, n)
        val = sum(vals) / sum(ns)
    else:
        val = value / n

    return val


def train(args):
    if args.distributed:
        init_dist(args)

    torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    summary_writer = None
    if args.local_rank == 0:
        summary_writer = SummaryWriter(checkpoint_dir / "logs")

    model = build_model(args)
    model = model.cuda(args.gpu)

    checkpoint = None
    if args.load:
        path_to_resume = Path(args.load).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            checkpoint = torch.load(
                path_to_resume,
                map_location=lambda storage, loc: storage.cuda(args.gpu),
            )
            
            nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
            model.load_state_dict(checkpoint["state_dict"])
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    weight_decay = args.weight_decay
    if weight_decay > 0:  # and filter_bias_and_bn:
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    optimizer = build_optimizer(parameters, args)

    if args.distributed:
        if args.syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True,
        )

    df = pd.read_csv(args.train_df)

    train_df, dev_df = train_dev_split(df, args)
    
    if args.local_rank == 0:
        print(train_df)
        print(dev_df)

    train_images_dir = Path(args.train_images_dir)
    img_size = args.img_size if args.img_size != IMG_SIZE else None
    train_dataset = Chimpact(
        df=train_df,
        img_dir=train_images_dir,
        transform=get_transforms(True, args.window_size, img_size=img_size),
        window_size=args.window_size,
    )

    if args.test_df is not None:
        test_df = pd.read_csv(args.test_df)
        def pr(row):
            video_id = row.video_id.split(".")[0]

            fpath = f"{video_id}_{row.time}.png"

            return fpath

        test_df["fpath"] = test_df.apply(pr, axis=1)
        test_df.drop(["video_id", "time"], axis=1, inplace=True)
        if args.local_rank == 0:
            print(test_df)

        test_dataset = Chimpact(
            df=test_df,
            img_dir=Path(args.test_images_dir),
            transform=get_transforms(True, args.window_size, img_size=img_size),
            window_size=args.window_size,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    val_dataset = Chimpact(
        df=dev_df,
        img_dir=train_images_dir,
        transform=get_transforms(False, args.window_size, img_size=img_size),
        window_size=args.window_size,
    )

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    args.num_workers = min(args.batch_size, 8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
    val_batch_size = 4 * args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    scheduler = build_scheduler(optimizer, args, n=len(train_loader) if args.scheduler_mode == "step" else 1)

    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    use_log = args.use_log
    criterion = MAEwithLog(use_log=use_log)

    metrics = [MAEMetric(from_log=use_log)]

    def saver(path, score):
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "score": score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": args,
            },
            path,
        )

    start_epoch = 0
    best_score = float("+inf")
    if args.resume and checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        optimizer.load_state_dict(checkpoint["opt_state_dict"])

        if checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for metric in metrics:
            metric.clear()

        desc = f"{epoch}/{args.num_epochs}"
        train_loss, n = epoch_step(
            train_loader,
            desc,
            model,
            criterion,
            metrics,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler_mode == "step" else None,
            scaler=scaler,
            local_rank=args.local_rank,
        )

        train_loss = all_gather(train_loss, n, args.distributed)

        train_scores = []
        for metric in metrics:
            _, ae, n = metric.eval() 
            mae = all_gather(ae, n, args.distributed)
            train_scores.append(mae)
            metric.clear()

        with torch.no_grad():
            dev_loss, n = epoch_step(
                val_loader,
                desc,
                model,
                criterion,
                metrics,
                optimizer=None,
                scaler=None,
                local_rank=args.local_rank,
            )

        dev_loss = all_gather(dev_loss, n, args.distributed)

        dev_scores = []
        for metric in metrics:
            _, ae, n = metric.eval() 
            mae = all_gather(ae, n, args.distributed)
            dev_scores.append(mae)
            metric.clear()

        if scheduler is not None and args.scheduler_mode == "epoch":
            scheduler.step()

        if args.local_rank == 0:
            for idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                summary_writer.add_scalar(
                    "group{}/lr".format(idx), float(lr), global_step=epoch
                )

            summary_writer.add_scalar("loss/train_mae", train_loss, global_step=epoch)
            summary_writer.add_scalar("loss/dev_mae", dev_loss, global_step=epoch)

            for train_score, dev_score in zip(train_scores, dev_scores):
                summary_writer.add_scalar("score/train_mae", train_score, global_step=epoch)
                summary_writer.add_scalar("score/dev_mae", dev_score, global_step=epoch)

            score = min(dev_scores)

            if score < best_score:
                best_score = score

                saver(checkpoint_dir / "model_best.pth", best_score)
                save_jit(model, args, checkpoint_dir / f"model_best.pt")

            saver(checkpoint_dir / "model_last.pth", score)
            save_jit(model, args, checkpoint_dir / "model_last.pt")

            if epoch % (2 * args.T_max) == (args.T_max - 1):
                saver(checkpoint_dir / f"model_last_{epoch + 1}.pth", score)
                save_jit(model, args, checkpoint_dir / f"model_last_{epoch + 1}.pt")

    if args.local_rank == 0:
        summary_writer.close()


def build_model(args):
    if args.encoder_name.startswith("vit_"):
        model = timm.create_model(
            args.encoder_name,
            pretrained=True,
            in_chans=args.in_channels * (2 * args.window_size + 1),
            num_classes=args.n_classes,
            img_size=args.img_size,  # (272, 480),
        )
    else:
        model = timm.create_model(
            args.encoder_name,
            pretrained=True,
            in_chans=args.in_channels * (2 * args.window_size + 1),
            num_classes=args.n_classes,
        )

    return model


def build_optimizer(parameters, args):
    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "fusedadam":
        optimizer = apex.optimizers.FusedAdam(
            parameters,
            adam_w_mode=True,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "fusedsgd":
        optimizer = apex.optimizers.FusedSGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"not yet implemented {args.optim}")

    return optimizer


def build_scheduler(optimizer, args, n=1):
    scheduler = None

    if args.scheduler.lower() == "cosa":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max * n,
            eta_min=max(args.learning_rate * 1e-2, 1e-7),
        )
    elif args.scheduler.lower() == "cosawr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.args.T_max,
            T_mult=2,
            eta_min=max(args.learning_rate * 1e-2, 1e-7),
        )
    else:
        print("No scheduler")

    return scheduler


def main():
    args = parse_args()
    if args.local_rank == 0:
        print(args)

    if args.local_rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
