import tensorflow as tf
import os
from utils.schedulers import get_lr_scheduler

        

def get_callbacks(CFG,monitor='val_mae'):
    # model checkpoint
    if not CFG.all_data:
        ckpt_path = 'fold-%i.h5'%(CFG.fold)
        ckpt_path = os.path.join(CFG.output_dir, ckpt_path)
        sv = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor=monitor, verbose=CFG.verbose, 
                                            save_best_only=True,save_weights_only=False, mode='min',
                                            save_freq='epoch')
    else:
        ckpt_path = 'model.h5'
        ckpt_path = os.path.join(CFG.output_dir, ckpt_path)
        sv = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor=monitor, verbose=CFG.verbose, 
                                                    save_best_only=True,save_weights_only=False, mode='min',
                                                    save_freq='epoch')
    # learning rate scheduler
    lr_scheduler = get_lr_scheduler(CFG.batch_size*CFG.replicas, CFG=CFG)
    lr_callback  = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=False)
    callbacks    = [sv, lr_callback]
        
    return callbacks