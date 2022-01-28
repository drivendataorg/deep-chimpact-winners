import tensorflow as tf

# LOSS
def get_loss(CFG):
    loss_name = CFG.loss
    if loss_name=='MAE':
        loss = tf.keras.losses.MeanAbsoluteError()
    elif loss_name=='BCE':
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)
    elif loss_name=='Huber':
        loss = tf.keras.losses.Huber(delta=CFG.huber_delta)
    else:
        raise NotImplemented
    return loss