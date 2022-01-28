import tensorflow as tf

# METRIC
def MAE():
    metric = tf.keras.metrics.MeanAbsoluteError(name='mae', dtype=None)
    return metric