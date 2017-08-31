"""
Train Unet model from a sampler
"""
from sklearn.preprocessing import StandardScaler

# local imports
from sampling import DataLoader, CSVDataset
from sampling import transforms as tx
#from models import get_Unet_model2D

base_dir = '/users/ncullen/desktop/projects/unet-ants/'
data_dir = base_dir + 'data/'


in_tx = StandardScaler()

co_tx = tx.Compose([tx.RandomAffine(rotation_range=(-15,15), # rotate btwn -15 & 15 degrees
                                    translation_range=(0.1,0.1), # translate btwn 0-10% horiz, 0-10% vert
                                    shear_range=(-10,10), # shear btwn -10 and 10 degrees
                                    zoom_range=(0.85,1.15)), # between 15% zoom-in and 15% zoom-out
                    tx.ExpandDims(axis=-1)])

# use a co-transform, meaning the same tx will be applied to both input+target images 
# this is necessary for image-to-image (e.g. segmentation) problems
dataset = CSVDataset(csv=data_dir+'image_filemap.csv', base_path=data_dir,
                     input_cols=['images'], target_cols=['masks'],
                     input_transform=in_tx, co_transform=co_tx)

# split into train and test set based on the `train-test` column in the filemap
train_data, val_data = dataset.split_by_column('train-test')

# create a dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

#model = get_Unet_model2D()
