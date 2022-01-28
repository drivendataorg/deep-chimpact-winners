[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/depth_side_by_side_magma_r.jpg)

# Deep Chimpact: Depth Estimation for Wildlife Conservation

## Goal of the Competition

Camera traps are widely used in conservation research to capture images and videos of wildlife without human interference. Using statistical models for [distance sampling](https://en.wikipedia.org/wiki/Distance_sampling), the frequency of animal sightings can be combined with the distance of each animal from the camera to estimate a species' full population size.

However, getting distances from camera trap footage currently entails an extremely manual, time-intensive process. This creates a bottleneck for critical information that conservationists can use to monitor and better protect wildlife populations and ecosystems. In this challenge, participants used machine learning to automatically estimate the distance between a camera trap and an animal in a series of camera trap videos.

## What's in this Repository

This repository contains code from winning competitors in the [Deep Chimpact: Depth Estimation for Wildlife Conservation](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/page/390/) DrivenData challenge. Additional solution details can be found in the `reports` folder inside the directory for each submission. Model scores represent [mean absolute error](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/page/391/#metric).

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

| Place              | Team or User                                                                                                                                                                                                                           | Public Score | Private Score | Summary of Model                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1                  | RTX 4090: [najib_haq](https://www.drivendata.org/users/najib_haq/), [awsaf49](https://www.drivendata.org/users/awsaf49/), [Bishmoy](https://www.drivendata.org/users/Bishmoy/), [zaber666](https://www.drivendata.org/users/zaber666/) | 1.648        | 1.620         | 11 deep neural network backbones were trained on high-dimension images with a variety of data augmentations, and then ensembled. Cross validation was used to avoid overfitting. Each model minimized either MAE or Huber Loss.                                                                                                                                                                        |
| 2                  | [kbrodt](https://www.drivendata.org/users/kbrodt/)                                                                                                                                                                                     | 1.633        | 1.626         | An EfficientNetV2 CNN model was trained using MAE loss and heavy augmentations. Sequences of images were stacked to capture motion. Predictions were ensembled across folds, and then used as pseudo labels for the train set for finetuning.                                                                                                                                                          |
| 3                  | [vecxoz](https://www.drivendata.org/users/vecxoz/)                                                                                                                                                                                     | 1.6575       | 1.6768        | Images were augmented with flips, rotations, and blurs. Gamma correction was used to extract more information from overexposed images. Models were trained on sequences of stacked frames 1-2 seconds apart, with the target frame in the middle. The final prediction is an ensemble of 12 models based on CNN-LSTM architecture with EfficientNet-B0 backbones, which were optimized using MAE loss. |
| MATLAB Bonus Prize | Team K_A: [kaveh9877](https://www.drivendata.org/users/kaveh9877/), [AZK90](https://www.drivendata.org/users/AZK90/)                                                                                                                   | 1.988        | 1.948         | A ResNet backbone was trained with augmentation from MATLAB's Deep Learning Toolbox. Two random partitions were considered and then combined using boosted trees from MATLAB's Regression Learner toolbox.                                                                                                                                                                                             |

The **benchmark blog post**, which is written in MATLAB, can be found [here](https://www.drivendata.co/blog/deep-chimpact-benchmark/).