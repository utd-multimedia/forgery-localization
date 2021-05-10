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


# backbone can be: efficientnetb0, efficientnetb2, efficientnetb3, efficientnetb4, mobilenetv2  
BACKBONE = 'efficientnetb4'

preprocess_input = sm.get_preprocessing(BACKBONE)


# predict the segmentation mask to localize forged area on RGB images
def test_predict():
	model = sm.Unet(BACKBONE, classes=1, activation='sigmoid', input_shape=(320,320,3))
	
	# load the trained model
	model.load_weights('./best_model.h5')

	# load the unseen test image
	imgg = cv2.imread('./test.png')
	
	orig_shape = imgg.shape
	print(orig_shape)
	# resize the image to be fit to the model
	imgg = cv2.resize(imgg,(320,320))
	imgg = imgg.astype('float') / 255.0
	x = np.expand_dims(imgg, axis=0)
	
	# compute the inference time
	start = timer()
	mask = model.predict(x)
	end = timer()
	print(str(end - start))

	# threshod of .5 for the predicted probablity to decide the pixel is forged or not
	mask[mask > 0.5] = 255
	mask[mask <= 0.5] = 0

	mask = cv2.resize(mask[0,:,:,:],(orig_shape[1],orig_shape[0]))
	# save the predicted mask for unseen test data. The mask represents the loalization of attack in the scene
	cv2.imwrite('test_manipulated12.jpg',mask)

if __name__ == "__main__":
	print(BACKBONE)	
	# predict on test RGB image
	test_predict()	
	

