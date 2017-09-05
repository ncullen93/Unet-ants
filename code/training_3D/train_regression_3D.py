"""
Train a UNET model to predict a continuous 3D image from a given
3D continuous brain image.

The example here uses the input image as a target image (aka an 'Autoencoder') but the
target image can be any other brain image.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from keras import callbacks as cbks

base_dir = '/users/ncullen/desktop/projects/unet-ants/'
os.chdir(base_dir+'code/')

# local imports
from sampling import DataLoader, CSVDataset
from sampling import transforms as tx
from models import create_unet_model3D


data_dir = base_dir + 'data_3D/'
results_dir = base_dir+'results_3D/'
try:
    os.mkdir(results_dir)
except:
    pass

# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.TypeCast('float32'),
                    tx.MinMaxScaler((-1,1)), # scale between -1 and 1
                    tx.ExpandDims(axis=-1), # expand from (128,128,128) to (128,128,128,1) -> RandomAffine and Keras expect that
                    tx.RandomAffine(rotation_range=(-15,15), # rotate btwn -15 & 15 degrees
                                    translation_range=(0.1,0.1), # translate btwn -10% and 10% horiz, -10% and 10% vert
                                    shear_range=(-10,10), # shear btwn -10 and 10 degrees
                                    zoom_range=(0.85,1.15), # between 15% zoom-in and 15% zoom-out
                                    turn_off_frequency=5,
                                    fill_value='min',
                                    target_fill_mode='constant',
                                    target_fill_value='min') # how often to just turn off random affine transform (units=#samples)
                    ])

# use a co-transform, meaning the same transform will be applied to input+target images at the same time 
# this is necessary since Affine transforms have random parameter draws which need to be shared
dataset = CSVDataset(filepath=data_dir+'image_filemap.csv', 
                    base_path=os.path.join(data_dir,'images'), # this path will be appended to all of the filenames in the csv file
                    input_cols=['Images'], # column in dataframe corresponding to inputs (can be an integer also)
                    target_cols=['Images'],# column in dataframe corresponding to targets (can be an integer also)
                    co_transform=co_tx)


# split into train and test set based on the `train-test` column in the csv file
# this splits alphabetically by values, and since 'test' comes before 'train' thus val_data is returned before train_data
val_data, train_data = dataset.split_by_column('TrainTest')

# overwrite co-transform on validation data so it doesnt have any random augmentation
val_data.set_co_transform(tx.Compose([tx.MinMaxScaler((-1,1)),
                                      tx.ExpandDims(axis=-1)]))

# create a dataloader .. this is basically a keras DataGenerator -> can be fed to `fit_generator`
batch_size = 10
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# write an example batch to a folder as JPEG
#train_loader.write_a_batch(data_dir+'example_batch/')

# create model
model = create_unet_model3D(input_image_size=train_data[0][0].shape, n_labels=1, layers=3,
                            mode='regression', output_activation='tanh')

# get baseline score predicting mean
# use this to make sure model isnt just predicting mean during training
#train_x, train_y = train_data.load()
#scores = []
#for i in range(len(train_x)):
#    scores.append(np.mean((train_x[i] - train_x[i].mean())**2))
#print('Baseline:' , np.mean(scores))

callbacks = [cbks.ModelCheckpoint(results_dir+'regression-weights.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size), 
                    epochs=100, verbose=1, callbacks=callbacks, 
                    shuffle=True, 
                    validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size), 
                    class_weight=None, max_queue_size=10, 
                    workers=1, use_multiprocessing=False,  initial_epoch=0)


### RUNNING INFERENCE ON THE NON-AUGMENTED DATA

# load all the validation data into memory.. not at all necessary but easier for this example
#val_x, val_y = val_data.load()
#real_val_x, real_val_y = val_data.load()
#real_val_y_pred = model.predict(real_val_x)



