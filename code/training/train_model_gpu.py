
import glob
import os
gpu_idx = 0
os.environ['THEANO_FLAGS'] = 'device=gpu%i, floatX=float32' % gpu_idx

import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks as cbks

# local import
os.chdir('/scratch/ncullen/unet-ants/code/')
from create_unet_model import create_Unet_model2D

base_dir = '/scratch/ncullen/unet-ants/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'TrainingData/'
number_of_labels = 1

x_train = np.load(data_dir + 'x_train.npy')
y_train = np.load(data_dir + 'y_train.npy')

unet_model = create_Unet_model2D(x_train[0].shape, n_labels=number_of_labels, layers=4)

n_epoch = 100
track = unet_model.fit(x_train, y_train,
                        epochs=n_epoch, batch_size=20,
                        verbose=1, shuffle=True,
                        callbacks=[cbks.ModelCheckpoint(base_dir+'weights-py.h5', monitor='val_loss', save_best_only=True),
                                   cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)],
                        validation_split=0.2)


plt.plot(np.arange(n_epoch), track.history['loss'])
plt.plot(np.arange(n_epoch), track.history['val_loss'])
plt.savefig(base_dir+'loss_fig.png')
plt.clf()

plt.plot(np.arange(n_epoch), track.history['dice_coefficient'])
plt.plot(np.arange(n_epoch), track.history['val_dice_coefficient'])
plt.savefig(base_dir+'dice_fig.png')



