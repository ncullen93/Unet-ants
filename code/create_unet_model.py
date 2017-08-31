
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                            MaxPooling2D, Concatenate)
from keras import optimizers as opt


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def create_Unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=32,
                        convolution_kernel_size=(3,3),
                        deconvolution_kernel_size=(3,3),
                        pool_size=(2,2),
                        strides=(2,2)):
    """
    Create a 2D Unet model

    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, np.arange(4))
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                activation='relu',
                                padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                activation='relu',
                                padding='same')(pool)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        activation='relu',
                                                        padding='same')(conv))

        if i < len(layers)-1:
            pool = MaxPooling2D(pool_size=pool_size,
                                strides=strides)(encoding_convolution_layers[i])


    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                    strides=strides, padding='same')(outputs)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)

    if number_of_classification_labels == 1:
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                        activation='sigmoid')(outputs)
    else:
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                        activation='softmax')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    if number_of_classification_labels == 1:
        unet_model.compile(loss=loss_dice_coefficient_error, 
                            optimizer=opt.Adam(lr=0.0001), metrics=[dice_coefficient])
    else:
        unet_model.compile(loss='categorical_crossentropy', 
                            optimizer=opt.Adam(lr=5e-5), metrics=['accuracy', 'categorical_crossentropy'])

    return unet_model


