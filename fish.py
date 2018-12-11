import skimage
from skimage import io
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_minimum
from skimage.transform import rescale, resize


fish_ic = io.imread_collection('./fish-test/*.tif')
fishes = fish_ic.concatenate()
print("fishes: ", fishes.shape)
scale = 1   #value by which to scale down the images
#fishes = resize(fishes, (fishes.shape[0], fishes.shape[1] / scale, fishes.shape[2] / scale), anti_aliasing=True)
print("fishes: ", fishes.shape)
print("len ", len(fish_ic))


def plot_image(ic):
    f, axes = plt.subplots(nrows=1, ncols=len(ic), figsize=(10, 10))
    for i, image in enumerate(ic):
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
    plt.show()

def plot_image_hist(ic):
    f, axes = plt.subplots(nrows=1, ncols=len(ic), figsize=(10, 10))
    for i, image in enumerate(ic):
        axes[i].hist(image.ravel(), bins=256)
        axes[i].axis('on')
    plt.show()

#plot_image(fish_ic)


cropped_ic = io.imread_collection('./fish-test_BLACKED-OUT/*.tif')
croppedfishes = cropped_ic.concatenate()
print("cropped fishes: ", croppedfishes.shape)
#plot_image(croppedfishes)
#plot_image_hist(croppedfishes)
croppedfishes = resize(croppedfishes, (croppedfishes.shape[0], croppedfishes.shape[1] / scale, croppedfishes.shape[2] / scale), anti_aliasing=False, preserve_range=False)
print("cropped fishes: ", croppedfishes.shape)

#plot_image(croppedfishes)
#plot_image_hist(croppedfishes)

brightpixels = np.copy(croppedfishes)
threshold = 60
threshold = 60/256
label = brightpixels >= threshold
#print(label[:1000,1000])
"""
#brightpixels[label] = 0
cond = [i >= threshold for i in brightpixels]
#print(count)
count = np.extract(cond,brightpixels)
cond2 = [i < threshold for i in brightpixels]
ommit = np.extract(cond2,brightpixels)
print("count: " , len(count),  " ommit: " , len(ommit))
"""
print("count: ", np.count_nonzero(label))


#works for one cropped image: 

filename = "./fish-test_BLACKED-OUT/atg5_sa22749_KT37_5dpi_plate1_005.tif"
cropfish = io.imread(filename,as_gray=True)
print("cropped fish: ", cropfish.shape)
image = np.array(cropfish)

image2 = np.copy(image)
threshold = 60
mask = image2 < threshold
#mask2 = image2 >= threshold
image2[mask] = 0
#image[mask2] = 100
cond = [i >= threshold for i in image2]
#print(count)
count = np.extract(cond,image2)
cond2 = [i < threshold for i in image2]
ommit = np.extract(cond2,image2)
print("count: " , len(count),  " ommit: " , len(ommit))
print(len(count) + len(ommit))  # checks total number of pixels counted, ommitted
print(2044*2048)  # total number of pixels

"""
plt.imshow(image2, cmap="gray", interpolation="nearest")
plt.axis("off")
plt.show()
"""
"""
#plots cropped image and modified image: 

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(image, cmap=plt.cm.gray)
ax[0, 0].set_title('Cropped Fish')

ax[0, 1].hist(image.ravel(), bins=256)
ax[0, 1].set_title('Histogram')

ax[1, 0].imshow(image2, cmap=plt.cm.gray)
ax[1, 0].set_title('Modified Crop')

ax[1, 1].hist(image2.ravel(), bins=256)
ax[1, 1].axvline(threshold, color='r')

for a in ax[:, 0]:
    a.axis('off')
plt.show()
"""