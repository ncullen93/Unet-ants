"""
Train Unet model from a sampler
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
from models import create_unet_model2D


data_dir = base_dir + 'data/'


input_tx = tx.MinMaxScaler((0,1))
target_tx = tx.BinaryMask(cutoff=0.5)

co_tx = tx.Compose([tx.ExpandDims(axis=-1),
                    tx.RandomAffine(fill_value='min',rotation_range=(-15,15), # rotate btwn -15 & 15 degrees
                                    translation_range=(0.1,0.1), # translate btwn 0-10% horiz, 0-10% vert
                                    shear_range=(-10,10), # shear btwn -10 and 10 degrees
                                    zoom_range=(0.85,1.15)) # between 15% zoom-in and 15% zoom-out
                    ])

# use a co-transform, meaning the same tx will be applied to both input+target images 
# this is necessary for image-to-image (e.g. segmentation) problems
dataset = CSVDataset(filepath=data_dir+'image_filemap.csv', base_path=data_dir,
                     input_cols=['images'], target_cols=['masks'],
                     input_transform=input_tx, target_transform=target_tx, co_transform=co_tx)

# split into train and test set based on the `train-test` column in the filemap
val_data, train_data = dataset.split_by_column('train-test')

# create a dataloader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

#train_loader.write_a_batch(data_dir+'example_batch/')

model = create_unet_model2D(input_image_size=train_data[0][0].shape, n_labels=1, layers=4)

callbacks = [cbks.ModelCheckpoint(base_dir+'weights-py.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

model.fit_generator(generator=train_loader, steps_per_epoch=np.ceil(len(train_data)/batch_size), 
                    epochs=100, verbose=1, callbacks=callbacks, 
                    shuffle=True, validation_data=val_loader, validation_steps=np.ceil(len(val_data)/batch_size), 
                    class_weight=None, max_queue_size=10, 
                    workers=1, use_multiprocessing=False,  initial_epoch=0)