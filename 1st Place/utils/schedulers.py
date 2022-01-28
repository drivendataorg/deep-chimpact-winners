# LR_SCHEDULE
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
        
def get_lr_scheduler(batch_size=8, CFG=None, plot=False):
    lr_start   = 5e-6
    lr_max     = CFG.lr_max * batch_size
    lr_min     = CFG.lr_min
    lr_ramp_ep = CFG.lr_ramp_ep
    lr_sus_ep  = CFG.lr_sus_ep
    lr_decay   = CFG.lr_decay
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        elif CFG.scheduler=='exp':
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        elif CFG.scheduler=='step':
            lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep)//2)
            
        elif CFG.scheduler=='cosine':
            decay_total_epochs = CFG.epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.5 * (1 + math.cos(phase))
            lr = (lr_max - lr_min) * cosine_decay + lr_min
        return lr
    
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(CFG.epochs), [lrfn(epoch) for epoch in np.arange(CFG.epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('learnig rate')
        plt.title('Learning Rate Scheduler')
        plt.savefig(f'{CFG.output_dir}/lr_schedule.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
        # plt.show()

    return lrfn