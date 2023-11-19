import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import (
    Input, 
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate
)


def conv_block(inputs=None, n_filters=32, dropout_prob=0., max_pooling=True):
    conv = Conv2D(n_filters,
                  3,   
                  activation="relu",
                  padding="SAME",
                  kernel_initializer='he_normal')(inputs)
    
    conv = Conv2D(n_filters,
                  3,
                  activation="relu",
                  padding="SAME",
                  kernel_initializer="he_normal")(conv)
    
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
         
        
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(
                 n_filters,
                 3,
                 strides=2,
                 padding="SAME")(expansive_input)
    
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                 3,
                 activation="relu",
                 padding="SAME",
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,
                 3,
                 activation="relu",
                 padding="SAME",
                 kernel_initializer="he_normal")(conv)
    
    return conv


def unet_model(input_size, n_filters=32, n_classes=2):
    inputs = Input(shape=input_size)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2**1)
    cblock3 = conv_block(cblock2[0], n_filters * 2**2)
    cblock4 = conv_block(cblock3[0], n_filters * 2**3, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 2**4, dropout_prob=0.3, max_pooling=False) 
    
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 2**3)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 2**2)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2**1)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, (1, 1), padding="same")(conv9)
    outputs = keras.layers.Activation('sigmoid')(conv10)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model