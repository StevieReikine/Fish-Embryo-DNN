# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

The original dataset is from the lab of Dr. Lalita Ramakrishnan. The images of fish and cropped fish were kindly shared with me, and I've done the pre-processing.

You can find it in folder data/fish.

The original image is of a zebrafish embryo infected with fluorescent bacteria and imaged in the fluorescence channel. This looks like:

![img/0test.png](img/0test.png)

The high level of background fluorescence means that to count the bright pixels inside the fish, ie. to quantify the level of infection, a researcher first manually crops out the area around the fish, resulting in the following image:

![img/0label.png](img/0label.png)

After the image has been cropped, the bright pixels are quantified by manually setting a threshold of brightness. 
The lab has hundreds-thousands of image pairs like this, so data for training is abundant. 

I propose to automate this process, and I would like to compare three different methods for doing this. 

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.


---





