#!/usr/bin/env sh


height=270
width=480
fps=1
ws=2


python3 \
    preprocess_video.py \
        --csv-path ./data/train_labels.csv \
        --video-dir ./data/train_videos \
        --save-image-dir ./data/train_images \
        --save-csv-name train.csv \
        --height "${height}" \
        --width "${width}" \
        --fps "${fps}" \
        --window-size "${ws}" \


python3 \
    preprocess_video.py \
        --csv-path ./data/submission_format.csv \
        --video-dir ./data/test_videos \
        --save-image-dir ./data/test_images \
        --save-csv-name test.csv \
        --height "${height}" \
        --width "${width}" \
        --fps "${fps}" \
        --window-size "${ws}" \
