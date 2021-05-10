## FDL-Net: Forgery Detection And Localization Network
Convolutional Neural Network based forgery detection and localization.

This repository contains an implementation of FDL-Net, a convolutional neural network model for forgery detection and localization on 2D RGB images and implementation of forgery localization in 3D LiDAR data in MATLAB. The predicted segmented mask by FDL-Net model is used to localize the area under attack on corresponding 3D LiDAR data. To implement the CNN model, we used a python library called Segmnetation Models based on Keras and Tensorflow.

# Quick start
First, import the Segmentation Models library:

.. code:: python

    import segmentation_models as sm
	
Installation
~~~~~~~~~~~~

**Requirements**

1) python 3
2) keras >= 2.2.0 or tensorflow >= 1.13
3) keras-applications >= 1.0.7, <=1.0.8


**PyPI stable package**

.. code:: bash

    $ pip install -U segmentation-models	
	
Training/Validation
~~~~~~~~~~~~~~~~~~~
After downloading the dataset (such as CASIA V2), you need to split it to three divisions and store in subfolders: train, val, and test. Their corresponding segmented masks will be stored in subfolders trainannot, valannot, and testannot. All 6 subfolders will be located in root folder of "data".

Below you can see the sample data from CASIA V2, RGB image and its segmnetad mask that shows the forged area.


|RGB Image   | Binary Mask |
| ---------- | ------------|
|RGB image   | ![RGB]      |
|Seg. mask   | ![seg]      |

[RGB]: images/01421.tif
[seg]: images/01421_gt.jpg


To train the model with FDL-Net: run FDL-Net.py
To predict the forgery segmented mask of the unseen test data: run predict.py
To evaluate the model on test set: run evaluate.py
To localize the forgery on 3D LiDAR (or point cloud) data: run TwoD_to_PC_Convert_Git.m
