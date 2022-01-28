#!/usr/bin/env sh


set -eu


GPU=${GPU:-0}


CUDA_VISIBLE_DEVICES="${GPU}" python3 \
    ./test.py \
        --test-df ./data/test.csv \
        --sub-format-df ./data/submission_format.csv \
        --test-images-dir ./data/test_images \
        --load ./weights/ \
        --save-sub ./sub.csv \
        --batch-size 256 \
        --window-size 2 \
        --tta 2 \


python -c '\
import pandas as pd;\
sub = pd.read_csv("sub.csv");\
sub.distance = (sub.distance * 2).round() / 2;\
sub.to_csv("subr.csv", index=False);\
'
