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


input_tx = tx.MinMaxScaler((0,1)) # scale input images between 0 and 1
target_tx = tx.BinaryMask(cutoff=0.5) # convert target segmentation to a binary mask

# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.ExpandDims(axis=-1), # expand from (128,128) to (128,128,1) -> RandomAffine and Keras expect that
                    tx.RandomAffine(fill_value='min', # can be integer also
                                    rotation_range=(-15,15), # rotate btwn -15 & 15 degrees
                                    translation_range=(0.1,0.1), # translate btwn -10% and 10% horiz, -10% and 10% vert
                                    shear_range=(-10,10), # shear btwn -10 and 10 degrees
                                    zoom_range=(0.85,1.15)) # between 15% zoom-in and 15% zoom-out
                    ])

# use a co-transform, meaning the same transform will be applied to input+target images at the same time 
# this is necessary since Affine transforms have random parameter draws which need to be shared
dataset = CSVDataset(filepath=data_dir+'image_filemap.csv', 
                    base_path=data_dir, # this path will be appended to all of the filenames in the csv file
                    input_cols=['images'], # column in dataframe corresponding to inputs (can be an integer also)
                    target_cols=['masks'],# column in dataframe corresponding to targets (can be an integer also)
                    input_transform=input_tx, target_transform=target_tx, co_transform=co_tx)


# split into train and test set based on the `train-test` column in the csv file
# this splits alphabetically by values, and since 'test' comes before 'train' thus val_data is returned before train_data
val_data, train_data = dataset.split_by_column('train-test')

# create a dataloader .. this is basically a keras DataGenerator -> can be fed to `fit_generator`
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# write an example batch to a folder as JPEG
#train_loader.write_a_batch(data_dir+'example_batch/')

# create model
model = create_unet_model2D(input_image_size=train_data[0][0].shape, n_labels=1, layers=4)

callbacks = [cbks.ModelCheckpoint(base_dir+'weights-py.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size), 
                    epochs=100, verbose=1, callbacks=callbacks, 
                    shuffle=True, 
                    validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size), 
                    class_weight=None, max_queue_size=10, 
                    workers=1, use_multiprocessing=False,  initial_epoch=0)


### RUNNING INFERENCE ON THE NON-AUGMENTED DATA

real_dataset = CSVDataset(filepath=data_dir+'image_filemap.csv', 
                        base_path=data_dir,
                        input_cols=['images'],
                        target_cols=['masks'],
                        input_transform=input_tx, target_transform=target_tx,
                        co_transform=tx.ExpandDims(axis=-1))

real_val_data, real_train_data = real_dataset.split_by_column('train-test')
real_val_x, real_val_y = real_val_data.load()

real_val_y_pred = model.predict(real_val_x)
real_val_y_pred[real_val_y_pred>=0.5] = 1
real_val_y_pred[real_val_y_pred<0.5] = 0

real_val_y_pred_plot = real_val_y_pred.copy()
real_val_y_pred_plot[real_val_y_pred_plot!=1]=None
real_val_y_plot = real_val_y.copy()
real_val_y_plot[real_val_y_plot!=1]=None

for i in range(len(real_val_data)):
    print(*['-']*20)
    plt.imshow(real_val_x[i,:,:,0])
    plt.imshow(real_val_y_plot[i,:,:,0],cmap='Reds_r')
    plt.title('ground truth')
    plt.show()
    plt.imshow(real_val_x[i,:,:,0])
    plt.imshow(real_val_y_pred_plot[i,:,:,0],cmap='Reds_r')
    plt.title('predicted')
    plt.show()
    print(*['-']*20)


