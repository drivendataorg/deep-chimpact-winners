# MODELING
import tensorflow as tf
import efficientnet.tfkeras as efn
from keras_cv_attention_models import nfnets, resnest
import keras_efficientnet_v2 as effnetv2
from utils.metrics import MAE
from utils.losses import get_loss
from utils.optimizers import get_optimizer

def get_base(CFG):
    model_name = CFG.model_name
    DIM        = CFG.img_size
    pretrain   = CFG.pretrain
    if 'EfficientNet' in model_name and 'V2' not in model_name:
        base = getattr(efn, model_name)(input_shape=(*DIM, 3),
                                      include_top=False,
                                       weights=pretrain,
                                      )
    elif 'EfficientNetV2' in model_name: # `pretrained` = [None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]
        base = getattr(effnetv2,model_name)(input_shape=(*DIM, 3),
                                         pretrained=pretrain,
#                                          drop_connect_rate=0.2,
                                         num_classes=0, )
    elif 'NFNet' in model_name:
        base = getattr(nfnets, model_name)(input_shape=(*DIM,3),
                                   pretrained=pretrain,
                                   num_classes=0)
        
    elif 'ResNest' in model_name:
        base = getattr(resnest, model_name)(input_shape=(*DIM, 3),
                   pretrained=pretrain,
                   num_classes=0)
    else:
        raise NotImplemented
    return base
    
def build_model(CFG, compile_model=True):
    base  = get_base(CFG)
    inp   = base.input
    out   = base.output
    out   = tf.keras.layers.GlobalAveragePooling2D()(out)
    out   = tf.keras.layers.Dense(CFG.num_features, activation='selu')(out)
    out   = tf.keras.layers.Dense(1, activation=None)(out)
    model = tf.keras.Model(inputs=inp, outputs=out)
    if compile_model:
        #optimizer
        opt  = get_optimizer(CFG)
        #loss
        loss = get_loss(CFG)
        #metric
        mae  = MAE()
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=[mae])
    return model
