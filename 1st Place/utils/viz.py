import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('error', UserWarning)
import seaborn as sns
sns.set(style='dark')
import tensorflow as tf
import numpy as np

# VISUALIZATION - SAMPLES
def plot_batch(batch, size=2, output_dir='',overlay_mask=False):
    if isinstance(batch, tuple):
        imgs, tars = batch
    else:
        imgs = batch; tars = None
    
    ## IF MASK IS IN INPUT OR OUTPUT
    mask_exists=False
    sl_exists= False
    if isinstance(imgs,dict):
        msks=imgs[list(imgs.keys())[-1]]
        imgs=imgs[list(imgs.keys())[0]]
        mask_exists=True

    elif isinstance(tars,dict):
        if 'msk' in list(tars.keys()):
            msks=tars['msk']
            mask_exists=True
        if 'soft_label' in list(tars.keys()):
            soft_tars=tars['soft_label']
            sl_exists=True
        tars=tars['tars']

    ## mask shape mismatch
    if mask_exists and msks[0].shape[:2]!=imgs[0].shape[:2]:
        msks=tf.image.resize(msks,imgs[0].shape[:2])

    plt.figure(figsize=(size*5, 5))
    for img_idx in range(size):
        plt.subplot(1, size, img_idx+1)
        if tars is not None:
            title=f'distance: {tars[img_idx].numpy()[0]}'
            title = title+ f'\nsoft label: {soft_tars[img_idx].numpy()[0]}' if sl_exists else title
            plt.title(title, fontsize=15)
        plt.imshow(imgs[img_idx,:, :, :])

        if overlay_mask & mask_exists:
            msk=msks[img_idx][:,:,0]
            plt.imshow(np.ma.masked_where(msk==0, msk), alpha=0.4, cmap='viridis_r')

        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_images.jpg',dpi=300,bbox_inches='tight',pad_inches=0)

def plot_dist(df, df_oof,output_dir=''):
    plt.figure(figsize=(10*2,6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(x=df['distance'], color='b',shade=True);
    sns.kdeplot(x=df_oof.pred.values, color='r',shade=True);
    plt.grid('ON')
    plt.xlabel('distance');plt.ylabel('freq');plt.title('KDE')
    plt.legend(['train', 'oof'])

    plt.subplot(1, 2, 2)
    sns.histplot(x=df['distance'], color='b');
    sns.histplot(x=df_oof.pred.values, color='r');
    plt.grid('ON')
    plt.xlabel('distance');plt.ylabel('freq');plt.title('Histogram')
    plt.legend(['train', 'oof'])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/histogram.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
    # plt.show()