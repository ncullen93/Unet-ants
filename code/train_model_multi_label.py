
import glob
import os
import ants
import numpy as np
import keras.backend as K
from keras import callbacks as cbks
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# local import
os.chdir('/users/ncullen/desktop/projects/unet-ants/code')
from create_unet_model import create_Unet_model2D

base_dir = '/users/ncullen/desktop/projects/unet-ants/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'TrainingData/'
training_dir = data_dir + 'TestingData/'

number_of_labels = 1
train_img_files = glob.glob(training_dir+'*H1_2D*')
train_mask_files = glob.glob(training_dir+'*Mask_2D*')

train_imgs = []
train_masks = []
train_img_arrays = []
train_mask_arrays = []
for i in range(len(train_img_files)):
    train_imgs.append(ants.image_read(train_img_files[i], dimension=2))
    train_masks.append(ants.image_read(train_mask_files[i], dimension=2))
    train_img_arrays.append(train_imgs[i].numpy())
    train_mask_arrays.append(train_masks[i].numpy())

train_data = np.asarray(train_img_arrays)
train_data = (train_data - train_data.mean()) / train_data.std()
train_label_data = np.asarray(train_mask_arrays)

segmentation_labels = np.sort(np.unique(train_label_data)).astype('int')
n_labels = len(segmentation_labels)

if K.image_data_format() == 'channels_last':
    x_train = np.expand_dims(train_data, -1)
else:
    x_train = np.expand_dims(train_data, 1)

y_train = train_label_data
ytshape = y_train.shape
y_train = to_categorical(y_train).reshape(list(ytshape)+[n_labels])


unet_model = create_Unet_model2D(x_train[0].shape, n_labels=n_labels, layers=4)
n_epoch = 100
track = unet_model.fit(x_train, y_train,
                        epochs=n_epoch, batch_size=20,
                        verbose=1, shuffle=True,
                        callbacks=[cbks.ModelCheckpoint(base_dir+'weights-multi-py.h5', monitor='val_loss', save_best_only=True),
                                   cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)],
                        validation_split=0.2)


#plt.plot(np.arange(n_epoch), track.history['loss'])
#plt.plot(np.arange(n_epoch), track.history['val_loss'])
#plt.savefig(base_dir+'loss_fig.png')
#plt.clf()

#plt.plot(np.arange(n_epoch), track.history['dice_coefficient'])
#plt.plot(np.arange(n_epoch), track.history['val_dice_coefficient'])
#plt.savefig(base_dir+'dice_fig.png')



