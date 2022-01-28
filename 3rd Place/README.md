3rd place solution
==================

"Deep Chimpact: Depth Estimation for Wildlife Conservation"
https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/

Author: Igor Ivanov

License: MIT


Summary
=======

Solution is an ensemble of 12 models each of which is a self-ensemble of 5 folds.
All models are based on CNN-LSTM architecture with EfficientNet-B0 backbone. 
Input data is a sequence of 7 or 9 video frames taken with 
an interval (time-step) of 1 or 2 seconds.
Each sequence has an equal number of frames taken before and after the target frame.
Each frame is a 3-channel image with a resolution 512 x 512. 
Optimization performed using Adam optimizer and MAE loss.
Significant improvements were obtained using different types of augmentation, 
especially affine transformations (flips, rotations) and gaussian blur.
Another source of improvement is gamma correction. 
Around 10% of all videos were overexposed, and 
gamma correction allows to extract much more information from such examples.


Demo
====

Notebook `notebook/demo.ipynb` in a minimalistic form demonstrates
how to infer distance for any frame in a single video.


Hardware and time
=================

Data preparation:
12 CPU, 16 GB RAM, 1 TB HDD
Prepare inference (test set): 4 hours
Prepare training (train set): 6 hours

Inference:
1x V100-16GB, 12 CPU, 16 GB RAM, 1 TB HDD
Inference time: 5 hours total (all models)

Training (in fact):
TPUv3-8, 1 CPU, 4 GB RAM, 1 TB HDD
Training time: 120 hours total (all models)

Training (possible alternative):
It's possible to use the following configuration with 3x smaller batch size
and explicit mixed precision "mixed_float16":
8x V100-16GB, 12 CPU, 48 GB RAM, 1 TB HDD
Training time: 300 hours total (all models)


Software
========

Ubuntu 18.04
Python: 3.6.9
CUDA: 11.1
cuDNN: 8.0.4

AWS CLI v2
```
curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip
unzip awscliv2.zip
sudo ./aws/install
```

FFmpeg v3
```
sudo apt update
sudo apt install ffmpeg
```

Python packages
```
pip3 install -r requirements.txt
```


Run inference
=============

All scripts use absolute paths with the `HOME` variable.
By default "3rd Place" directory is expected to be extracted as: `$HOME/3rd Place`
Final submission should appear as `$HOME/3rd Place/submission.csv`

```
cd $HOME/3rd\ Place
bash prepare_inference.sh
bash run_inference.sh
```

*Note:* Depending on your system, you may need to add `tf.keras.backend.set_image_data_format('channels_last')` to the top of each `run.py` script. The same is true when running training.


Run training
============

```
cd $HOME/solution
bash prepare_training.sh
bash run_training.sh
```


Acknowledgement
===============

Thanks to [TRC program]( https://sites.research.google/trc/about/) I had an opportunity to run experiments on TPUv3-8.


