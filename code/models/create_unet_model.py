
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                            MaxPooling2D, Concatenate, UpSampling2D,
                            Conv3D, Conv3DTranspose, MaxPooling3D,
                            UpSampling3D)
from keras import optimizers as opt


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def create_unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5,5),
                        deconvolution_kernel_size=(5,5),
                        pool_size=(2,2),
                        strides=(2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001):
    """
    Create a 2D Unet model

    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
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
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model


def create_unet_model3D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5,5,5),
                        deconvolution_kernel_size=(5,5,5),
                        pool_size=(2,2,2),
                        strides=(2,2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001):
    """
    Create a 3D Unet model

    Example
    -------
    unet_model = create_unet_model3D( (128,128,128,1), 1, 4)
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
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            activation='relu',
                            padding='same')(inputs)
        else:
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            activation='relu',
                            padding='same')(pool)

        encoding_convolution_layers.append(Conv3D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        activation='relu',
                                                        padding='same')(conv))

        if i < len(layers)-1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        tmp_deconv = Conv3DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling3D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=4)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])

        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)
        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model
