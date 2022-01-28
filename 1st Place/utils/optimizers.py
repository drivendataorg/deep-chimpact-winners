import tensorflow as tf
import tensorflow_addons as tfa

def get_optimizer(CFG):
    opt_name = CFG.optimizer
    lr       = CFG.lr
    if opt_name=='Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name=='AdamW':
        opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=lr)
    elif opt_name=='RectifiedAdam':
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    else:
        raise ValueError("Wrong Optimzer Name")
    return opt