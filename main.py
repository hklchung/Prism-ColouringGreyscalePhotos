from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
from PIL import Image, ImageOps
import random
import tensorflow as tf

#==============View images===============
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('Image/Train/IMG_6955.JPG')
imgplot = plt.imshow(img)
plt.show()

#==============Resize image==============
for filename in os.listdir('Image/Train'):
    temp = Image.open('Image/Train/' + filename)
    size = 400, 400
    temp.thumbnail(size, Image.ANTIALIAS)
    temp.save('Image/Train/Resized/' + filename, "JPEG")
    
#==============Get images================
X = []
for filename in os.listdir('Image/Train/Resized'):
    temp = np.array(img_to_array(load_img('Image/Train/Resized/'+filename)), dtype=float)
    hor = 400 - temp.shape[0]
    ver = 400 - temp.shape[1]
    if hor%2 != 0:
        temp = np.pad(temp, ((hor//2 + 1, hor//2), (ver//2, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    elif ver%2 != 0:
        temp = np.pad(temp, ((hor//2, hor//2), (ver//2 + 1, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    else:
    # pad resized images such that they are all 400x400x3
        temp = np.pad(temp, ((hor//2, hor//2), (ver//2, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    X.append(np.array(temp, dtype=float))
#X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95*len(X))
train = X[:split]
train = [1.0/225*x for x in train]
Xtrain = np.array([rgb2lab(x)[:,:,0].reshape(400, 400, 1) for x in train])
ytrain = np.array([rgb2lab(x)[:,:,1:].reshape(400, 400, 2) for x in train])
del(train)
del(X)

#==============Set up model==============
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer='rmsprop',loss='mse')

#=============Train model===============
model.fit(x=Xtrain, 
    y=ytrain,
    batch_size=10,
    epochs=1)