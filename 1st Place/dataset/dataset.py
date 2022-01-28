# SEEDING
import numpy as np
import random
import os
import tensorflow as tf
from dataset.augment import transform, RandomBorder, dropout, RandomFlip, RandomJitter
from dataset.augment import OpticalDistortion, GridDistortion
from dataset.augment import random_float

# SEEDING
def seeding(CFG):
    SEED = CFG.seed
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.random.set_seed(SEED)
    print('> SEEDING DONE')

# DATA PIPELINE
def build_decoder(with_labels=True, target_size=[360, 640], CFG=None, ext='png'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
        img = tf.cast(img, tf.float32)
        # some images from test have different dimensions
        if tf.math.reduce_any(target_size!=tf.shape(img)[:2]):
            if CFG.resize_with_pad:
                img = tf.image.resize_with_pad(img, target_size[0], target_size[1])
            else:
                img = tf.image.resize(img, target_size)
        if CFG.remove_border:
            img = RandomBorder(img, target_size=target_size,mode='constant',
                               fill_border=CFG.fill_border,remove_symbol=CFG.remove_symbol)
        img = tf.cast(img, tf.float32)
        # normalization

        img = img / 255.0
        img = tf.reshape(img, [*target_size, 3])

        return img
    
    def decode_with_labels(path, label):
        label = tf.cast(label, tf.float32)
        return decode(path), label
    
    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True, dim=[360, 640], CFG=None):
    def augment(img, dim=dim):
        if random_float()<CFG.transform_prob:
            img = transform(img,DIM=dim,
                            ROT=CFG.rot,SHR=CFG.shr,
                            H_ZOOM=CFG.hzoom, V_ZOOM=CFG.vzoom,
                            H_SHIFT=CFG.hshift, V_SHIFT=CFG.vshift, 
                            FILL_MODE=CFG.fill_mode)
        # flip
        img = RandomFlip(img, prob_hflip=CFG.hflip, prob_vflip=CFG.vflip)
        # color-jitter
        img = RandomJitter(img, hue=CFG.hue, sat=CFG.sat, cont=CFG.cont, bri=CFG.bri)
        # rgb->gray
        if random_float()<CFG.gray_prob: # rgb to grayscale
            img = tf.image.rgb_to_grayscale(img)
            img = tf.concat([img for _ in range(3)], axis=-1)
        # random botder remove
        if random_float()<CFG.border_prob: # border augmentation -> randomly create black border
            img = RandomBorder(img, target_size=dim, mode='random',
                               fill_border=CFG.fill_border, remove_symbol=CFG.remove_symbol)
        # distortion augmentation -> randomly distort image
        if random_float()<CFG.distortion_prob:
            if random_float()<0.5:
                img = OpticalDistortion(img, distort_limit=0.2, shift_limit=0.0)
            else:
                img = GridDistortion(img, num_steps=5, distort_limit=0.2)
        # cutout
        if with_labels:
            img = dropout(img, DIM=dim, PROBABILITY = CFG.drop_prob, CT = CFG.drop_cnt, SZ = CFG.drop_size)
        # clip
        if CFG.clip:
            img = tf.clip_by_value(img, 0, 1)  
        img = tf.reshape(img, [*dim, 3])
        return img
    
    def augment_with_labels(img, label):    
        return augment(img), label
    
    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False, CFG=None):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None,target_size=CFG.img_size,
                                  CFG=CFG)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None,dim=CFG.img_size,
                                     CFG=CFG)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    opt = tf.data.Options()
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt.experimental_deterministic = False
    if CFG.device=='GPU':
        opt = tf.data.Options()
        opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(opt)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)
    return ds
