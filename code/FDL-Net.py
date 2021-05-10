# Forgery Localization on RGB images 
# Mltimedia-lab
import os
import segmentation_models as sm
import albumentations as A
import numpy as np
from skimage.io import imsave 
from keras.preprocessing import image
import pandas as pd
import cv2
import keras
import tensorflow as tf
import datetime
from timeit import default_timer as timer
from keras_flops import get_flops


# backbone can be: efficientnetb0, efficientnetb2, efficientnetb3, efficientnetb4, mobilenetv2  
BACKBONE = 'efficientnetb4'

# To initialize the hyperparameters
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 80

preprocess_input = sm.get_preprocessing(BACKBONE)

#for binary segmentation like our use case the number of classes is 1
CLASSES = ['one']
DATA_DIR = './data/'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(320, 320)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    CLASSES = CLASSES
   
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image,(320,320))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask,(320,320))

        # extract certain classes from mask (e.g. forged)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def train():
	dataset = Dataset(x_train_dir, y_train_dir,classes=CLASSES)
	# define network parameters
	n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  
	activation = 'sigmoid' if n_classes == 1 else 'softmax'

	# create model
	model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(320,320,3))
	# define optomizer
	optim = keras.optimizers.Adam(LR)

	# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
	dice_loss = sm.losses.DiceLoss()
	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	total_loss = dice_loss + (0.5 * focal_loss)

	metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

	model.summary()
	# compile keras model with defined optimozer, loss and metrics
	model.compile(optim, total_loss, metrics)

	# Dataset for train images
	train_dataset = Dataset(
	    x_train_dir, 
	    y_train_dir, 
	    classes=CLASSES, 
	    augmentation=get_training_augmentation(),
	    preprocessing=get_preprocessing(preprocess_input),
	)

	# Dataset for validation images
	valid_dataset = Dataset(
	    x_valid_dir, 
	    y_valid_dir, 
	    classes=CLASSES, 
	    augmentation=get_validation_augmentation(),
	    preprocessing=get_preprocessing(preprocess_input),
	)

	train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

	# check shapes for errors
	assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
	assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

	# define callbacks for learning rate scheduling and best checkpoints saving
	callbacks = [
	    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
	    keras.callbacks.ReduceLROnPlateau(),
	]
	# plot the architecture of U-net style network with backbone of 'efficientnetb4'
	tf.keras.utils.plot_model(
	model,
	to_file='model_arch.png',
	show_shapes=True,
	show_layer_names=True,
	rankdir='TB'
	)

	# train model
	history = model.fit_generator(
	    train_dataloader, 
	    steps_per_epoch=len(train_dataloader), 
	    epochs=EPOCHS, 
	    callbacks=callbacks, 
	    validation_data=valid_dataloader, 
	    validation_steps=len(valid_dataloader),
	)
	
	# to save the history of training
	hist_df = pd.DataFrame(history.history)
	hist_csv_file = 'history.csv'
	with open(hist_csv_file, mode='w') as f:
		hist_df.to_csv(f)
	


if __name__ == "__main__":
	print(BACKBONE)
	# To load the data and train the model
	train()	
	


