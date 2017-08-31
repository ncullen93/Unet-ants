
import glob
import os
gpu_idx = 0
os.environ['THEANO_FLAGS'] = 'device=gpu%i, floatX=float32' % gpu_idx

import numpy as np
import keras.backend as K
from keras import callbacks as cbks
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# local import
os.chdir('/scratch/ncullen/unet-ants/code')
from create_unet_model import create_Unet_model2D

base_dir = '/scratch/ncullen/unet-ants/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'TrainingData/'
results_dir = base_dir+'results/'

x_train = np.load(data_dir+'x_train_multi.npy')
y_train = np.load(data_dir+'y_train_multi.npy')
# y_train is already in onehot format
segmentation_labels = np.arange(y_train.shape[-1])
n_labels = len(segmentation_labels)


unet_model = create_Unet_model2D(x_train[0].shape, n_labels=n_labels, layers=5)
n_epoch = 150
track = unet_model.fit(x_train, y_train,
                        epochs=n_epoch, batch_size=20,
                        verbose=1, shuffle=True,
                        callbacks=[cbks.ModelCheckpoint(results_dir+'weights-multi-py.h5', monitor='val_loss', save_best_only=False),
                                   cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)],
                        validation_split=0.2)

y_pred = unet_model.predict(x_train)
np.save(results_dir+'y_pred.npy', y_pred)
np.save(results_dir+'y_true.npy', y_train)

np.save(results_dir+'val_loss_multi.npy', track.history['val_loss'])
np.save(results_dir+'loss_multi.npy', track.history['loss'])
np.save(results_dir+'acc_multi.npy', track.history['acc'])
np.save(results_dir+'val_acc_multi.npy', track.history['val_acc'])



