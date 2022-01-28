# IMPORT LIBRARIES
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import seaborn as sns
sns.set(style='dark')

import tensorflow as tf
import tensorflow.keras.backend as K

import argparse
from utils.config import dict2cfg
from utils.device import get_device
from dataset.dataset import seeding, build_dataset
from dataset.processing import clean_data
from dataset.split import create_folds
from models.model import build_model
from utils.viz import plot_batch, plot_dist
from utils.schedulers import get_lr_scheduler
from utils.callbacks import get_callbacks
from utils.metrics import MAE
from utils.submission import path2info, get_shortest_distance
import time
from functools import partial
print('> IMPORT COMPLETE')




def train(CFG):
    oof_pred = []; oof_tar = []; oof_val = []; oof_ids = []; oof_folds = []
    preds = [] #np.zeros((test_df.shape[0],1))
    for fold in range(CFG.folds):
        start = time.time()
        CFG.fold = fold if not CFG.all_data else -1
        if fold not in CFG.selected_folds:
            continue
                
        # TRAIN AND VALID DATAFRAME
        train_df = df.query("fold!=@fold")
        valid_df = df.query("fold==@fold")
        
        # CREATE TRAIN AND VALIDATION SUBSETS
        train_paths = train_df.image_path.values; train_labels = train_df[CFG.target_col].values.astype(np.float32)
        valid_paths = valid_df.image_path.values; valid_labels = valid_df[CFG.target_col].values.astype(np.float32)
        
        # SHUFFLE IMAGE AND LABELS
        index = np.arange(len(train_paths))
        np.random.shuffle(index)
        train_paths  = train_paths[index]
        train_labels = train_labels[index]
        
        if CFG.debug:
            train_paths = train_paths[:CFG.min_samples]; train_labels = train_labels[:CFG.min_samples]
            valid_paths = valid_paths[:CFG.min_samples]; valid_labels = valid_labels[:CFG.min_samples]
            
        # BATCH SIZE FOR INEFERENCE
        infer_bs = int(CFG.batch_size*CFG.infer_bs_scale)
           
        print('#'*25); print('#### FOLD',CFG.fold)
        print('#### IMAGE_SIZE: (%i, %i) | MODEL_NAME: %s | BATCH_SIZE: %i'%
            (CFG.img_size[0],CFG.img_size[1],CFG.model_name,CFG.batch_size*CFG.replicas))
        print('#### OPTIMIZER: %s | LOSS: %s | SCHEDULER: %s | PRETRAIN: %s'%(CFG.optimizer,CFG.loss,
                                                                              CFG.scheduler,CFG.pretrain))
        train_images = len(train_paths)
        val_images   = len(valid_paths)
        print('#### NUM_TRAIN: %i | NUM_VALID: %i'%(train_images, val_images))
        print('#### ALL_DATA: True') if CFG.all_data else None
        
        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = build_model(CFG, compile_model=True)

        # DATASET
        cache    = True if np.sqrt(np.prod(CFG.img_size))<=768 else False
        train_ds = build_dataset(train_paths, train_labels, cache=cache, batch_size=CFG.batch_size*CFG.replicas,
                    repeat=True, shuffle=True, augment=CFG.augment, drop_remainder=False, CFG=CFG)
        val_ds   = build_dataset(valid_paths, valid_labels, cache=cache, batch_size=CFG.batch_size*CFG.replicas,
                    repeat=False, shuffle=False, augment=False, drop_remainder=False, CFG=CFG)
        
        print('#'*25)   
        
        # CALLBACKS
        callbacks = get_callbacks(CFG)

        # TRAIN
        history = model.fit(
            train_ds, 
            epochs=CFG.epochs if not CFG.debug else 2, 
            callbacks = callbacks, 
#             initial_epoch=CFG.initial_epoch,
            steps_per_epoch=len(train_paths)/CFG.batch_size/CFG.replicas,
            validation_data=val_ds,
            #validation_steps=len(valid_paths)/CFG.batch_size/CFG.replicas,
            #class_weight = {0:1,1:2},
            verbose=CFG.verbose
        )
        # Loading best model for inference
        if not CFG.all_data:
            print('\nLoading best model...') 
            model.load_weights(f'{CFG.output_dir}/fold-%i.h5'%fold)  

        if not CFG.all_data:
            # PREDICT OOF USING TTA
            print('Predicting OOF with TTA...')
            ds_valid = build_dataset(valid_paths, labels=None, cache=False, batch_size=infer_bs*CFG.replicas,
                                    repeat=True, shuffle=False, augment=True if CFG.tta>1 else False, drop_remainder=False,
                                    CFG=CFG)
            ct_valid = len(valid_paths); STEPS = CFG.tta * ct_valid/infer_bs/CFG.replicas
            pred = model.predict(ds_valid,steps=STEPS,verbose=CFG.verbose)[:CFG.tta*ct_valid,] 
            oof_pred.append(getattr(np, CFG.agg)(pred.reshape((ct_valid,-1,CFG.tta),order='F'),axis=-1))                 
            
            # GET OOF TARGETS AND idS
            oof_tar.append(valid_df[CFG.target_col].values[:ct_valid])
            oof_folds.append(np.ones_like(oof_tar[-1],dtype='int8')*fold )
            oof_ids.append(valid_paths)
            
        if not CFG.all_data:
            # REPORT RESULTS
            y_true = oof_tar[-1]; y_pred = oof_pred[-1]
            mae    = MAE()(y_true.astype(np.float32),y_pred).numpy()
            oof_val.append(np.min( history.history['val_mae'] ))
            print('\n>>> FOLD %i OOF MAE without TTA = %.3f, with TTA = %.3f'%(fold,oof_val[-1],mae));
            # SAVING OOF & PREDICTIONS
            if CFG.debug:
                valid_df = valid_df.iloc[:CFG.min_samples]
            valid_df['pred'] = y_pred.reshape(-1)
            valid_df['diff'] = abs(valid_df.distance - valid_df.pred)
            valid_df         = valid_df.sort_values(by='diff', ascending=False)
            valid_df.to_csv(f'{CFG.output_dir}/oof_{CFG.fold:02d}.csv', index=False)
            valid_df         = valid_df.groupby('video_id').head(2) # do not take many samples from one video
            valid_df         = valid_df.reset_index(drop=True)
        end = time.time()
        eta = (end-start)/60
        print(f'>>> TIME: {eta:0.2f} min\n\n')
        
    
    if not CFG.all_data:
        # COMPUTE OVERALL OOF MAE
        oof = np.concatenate(oof_pred); true = np.concatenate(oof_tar);
        ids = np.concatenate(oof_ids); folds = np.concatenate(oof_folds)
        mae = MAE()(true.astype(np.float32),oof)
        print('> Overall OOF MAE with TTA = %.3f'%mae)

        # SAVE OOF TO DISK
        print('> PROCESSING OOF:')
        columns = ['image_path', 'fold', 'true', 'pred']
        df_oof = pd.DataFrame(np.concatenate([ids[:,None], folds[:, 0:1], true, oof], axis=1), columns=columns)
        tqdm.pandas(desc='path2info ')
        df_oof = df_oof.progress_apply(path2info,axis=1)
        df_oof = df_oof.drop(['image_path'], axis=1)
        if CFG.rounding:
            tqdm.pandas(desc='rounding ')
            df_oof = df_oof.progress_apply(partial(get_shortest_distance, zero_distance=CFG.zero_distance), axis=1)
        df_oof.to_csv(f'{CFG.output_dir}/oof-{CFG.img_size[0]}x{CFG.img_size[1]}.csv',index=False)

        # PLOT HISTOGRAM
        plot_dist(df, df_oof, output_dir=CFG.output_dir)
    return 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/deep-chimpact.yaml', help='config file')
    parser.add_argument('--debug', type=int, default=None, help='process only small portion in debug mode')
    parser.add_argument('--model-name', type=str, default=None, help='name of the model')
    parser.add_argument('--img-size', type=int, nargs='+', default=None, help='image size: H x W')
    parser.add_argument('--batch-size', type=int, default=None, help='batch_size for the model')
    parser.add_argument('--loss', type=str, default=None, help='name of the loss function')
    parser.add_argument('--scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument('--selected-folds', type=int, nargs='+', default=None, help='folds to train')
    parser.add_argument('--all-data', type=int, default=None, help='use all data for training no-val')
    parser.add_argument('--ds-path', type=str, default=None, help='path to dataset')
    parser.add_argument('--output-dir', type=str, default=None, help='output path to save the model')
    opt = parser.parse_args()
    
    # LOADING CONFIG
    CFG_PATH = opt.cfg
    cfg_dict = yaml.load(open(CFG_PATH, 'r'), Loader=yaml.FullLoader)
    CFG      = dict2cfg(cfg_dict) # dict to class
    # print('config:', cfg)

    # OVERWRITE
    if opt.debug is not None:
        CFG.debug = opt.debug
    print('> DEBUG MODE:', bool(CFG.debug))
    if opt.model_name:
        CFG.model_name = opt.model_name
    if opt.img_size:
        assert len(opt.img_size)==2, 'image size must be H x W'
        CFG.img_size = opt.img_size
    if opt.batch_size:
        CFG.batch_size = opt.batch_size
    if opt.loss:
        CFG.loss = opt.loss
    if opt.scheduler:
        CFG.scheduler = opt.scheduler
    if opt.output_dir:
        output_dir = os.path.join(opt.output_dir, '{}-{}x{}'.format(CFG.model_name, CFG.img_size[0], CFG.img_size[1]))
        os.system(f'mkdir -p {output_dir}')
    else:
        output_dir = os.path.join('output', '{}-{}x{}'.format(CFG.model_name, CFG.img_size[0], CFG.img_size[1]))
        os.system(f'mkdir -p {output_dir}')
    CFG.output_dir = output_dir
    if opt.selected_folds:
        CFG.selected_folds = opt.selected_folds
    if opt.all_data:
        CFG.all_data = opt.all_data
    if CFG.all_data:
        CFG.selected_folds = [0]
        
    # CONFIGURE DEVICE
    strategy, device = get_device()
    CFG.device   = device
    AUTO         = tf.data.experimental.AUTOTUNE
    CFG.replicas = strategy.num_replicas_in_sync
    print(f'> REPLICAS: {CFG.replicas}')
    
    # MINIMUM SAMPLES FOR DEBUG
    CFG.min_samples = CFG.batch_size * CFG.replicas * 2
    
    # SEEDING
    seeding(CFG)
    
    # DS_PATH
    if opt.ds_path is not None:
        CFG.ds_path = opt.ds_path
    if not os.path.isdir(CFG.ds_path):
        raise ValueError(f'directory, <{CFG.ds_path}> not found')
    
    print('> DS_PATH:',CFG.ds_path)
    
    # META DATA
    ## Train Data
    df = pd.read_csv(F'{CFG.ds_path}/train.csv')
    df['image_path'] = CFG.ds_path + '/train_images/' + df.video_id + '-' + df.time.map(lambda x: f'{x:03d}') + '.png'
    # print(df.head(2))

    # CHECK FILE FROM DS_PATH
    assert os.path.isfile(df.image_path.iloc[0])
    print('> DS_PATH: OKAY')
    
    # CLEAN DATA
    if CFG.clean_data:
        df = clean_data(df)
    
    # DATA SPLIT
    df = create_folds(df, CFG=CFG)

    # CHECK OVERLAP IN FOLDS
    overlap = set(df.query("fold==0").site_id.unique()).intersection(set(df.query("fold!=0").site_id.unique()))
    assert len(overlap)==0
    
    # PLOT SOME DATA
    fold = 0
    fold_df = df.query('fold==@fold')[100:200]
    paths  = fold_df.image_path.tolist()
    labels = fold_df[CFG.target_col].values
    ds     = build_dataset(paths, labels, cache=False, batch_size=CFG.batch_size*CFG.replicas,
                           repeat=True, shuffle=True, augment=True, CFG=CFG)
    ds = ds.unbatch().batch(20)
    batch = next(iter(ds))
    plot_batch(batch, 5, output_dir=CFG.output_dir)
    
    # PLOT LR SCHEDULE
    get_lr_scheduler(CFG.batch_size*CFG.replicas, CFG=CFG, plot=True)
    # Training
    print('> TRAINING:')
    train(CFG)
    