# The purpose of this script is to count bright pixels that are inside of a fish. 
# The method is through a machine learning image segmentation algorithm.
# The training data is generated from images of fish and corresponding cropped images of fish.
# Author: Stevie Reikine

import os
from os import listdir
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_minimum
from skimage.transform import rescale, resize
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

#get set number (batch size) of images from two folders (take path as input) and create two tensors- one of raw images and the second of the corresponding cropped images
data_path = './fish-test/'
label_path = './fish-test_BLACKED-OUT/'
#check that both paths have the same files
filenames_fish = os.listdir(data_path)
filenames_blackedout = os.listdir(label_path)
if len(filenames_fish) == len(filenames_blackedout):
    print('Same number of files.')
else:
    print('Different number of files.')
filenames_confirmed = [] 
for filename in filenames_fish:
    if filename in filenames_blackedout:
        filenames_confirmed.append(filename)
    else:
        print(filename + ' is not in blacked out folder')
#training data generator        
def DataGenerator(batch_size, data_path, label_path, filenames_confirmed):
    for i in range (0, len(filenames_confirmed)//batch_size):
        # read files into arrays    
        fish_images = []
        fish_labels = []
        for b in range (0, batch_size):
            image = io.imread(data_path + filenames_confirmed[i + b])
            fish_images.append(image)
            #could implement check for same file name here (if have more than ~hundred files)
            label = io.imread(label_path + filenames_confirmed[i + b])
            fish_labels.append(label)
        
        fish_images = np.array(fish_images)
        fish_labels = np.array(fish_images)

        # generate labeled data to use for training
        # from cropped images make array of labels- 0 or 1 for counting or not counting, based on input threshold
        threshold = 60
        fish_masks = fish_labels >= threshold

        # add black pixels to side to reshape from 2044x2048 to 2048x2048
        resized_fish= np.pad(fish_images,((0,0),(2,2),(0,0)),'minimum')
        resized_masks= np.pad(fish_masks,((0,0),(2,2),(0,0)),'minimum')

        # scale data and labels to 256 x 256 x 1 to be optimal for current u-net architecture
        scale = 8   #value by which to scale down the images, 2048/256 = 8
        resized_fish = resize(resized_fish, (resized_fish.shape[0], resized_fish.shape[1] / scale, resized_fish.shape[2] / scale,1), anti_aliasing=True)
        resized_masks = resize(resized_masks, (resized_masks.shape[0], resized_masks.shape[1] / scale, resized_masks.shape[2] / scale,1), anti_aliasing=True)
        print(resized_fish.shape)
        yield (resized_fish,resized_masks)


# print out plots of images to see data and labels (check)
def plot_dataset(data, data_name, label, label_name, display_num):
    num_images = len(data)
    r_choices = np.random.choice(num_images, display_num)
    plt.figure(figsize=(10,15))
    for i in range(0, display_num*2, 2):
        img_num = r_choices[i // 2]
        image = data[img_num]
        mask = label[img_num]
        plt.subplot(display_num, 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(data_name)
        plt.subplot(display_num, 2, i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(label_name)
    plt.show()

#plot_dataset(resized_fish,"fish", resized_masks, "masks", 4)


# U-NET architecture, largely followed from https://github.com/zhixuhao/unet

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# training 
batch_size = 3

myData = DataGenerator(batch_size, resized_fish, resized_masks, filenames_confirmed)
model = unet()
model_checkpoint = ModelCheckpoint('unet_fish.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myData,steps_per_epoch=3,epochs=1,callbacks=[model_checkpoint])

# test output



# count how many bright pixels are present
