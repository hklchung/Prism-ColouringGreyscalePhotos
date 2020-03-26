from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
from PIL import Image, ImageOps
import random
import tensorflow as tf

#=============Have GPU?==================
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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
images = []
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
    images.append(np.array(temp, dtype=float))
#X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95*len(X))
train = X[:split]
train = [1.0/225*x for x in train]
Xtrain = np.array([rgb2lab(x)[:,:,0].reshape(400, 400, 1) for x in train])
ytrain = np.array([rgb2lab(x)[:,:,1:].reshape(400, 400, 2) for x in train])
Xtrain /= 128
ytrain /= 128
del(train)
del(X)

#==============Transform images==========
X = [rgb2lab(1.0/255*x)[:,:,0] for x in images]
Y = [rgb2lab(1.0/255*x)[:,:,1:] for x in images]
Y = [x/128 for x in Y]
X = [x.reshape(400, 400, 1) for x in X]
Y = [x.reshape(400, 400, 2) for x in Y]
X = np.array(X)
Y = np.array(Y)

#==============Set up model==============
model = Sequential()
model.add(InputLayer(input_shape=(400, 400, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

#model = load_model('model.h5')

#=============Train model===============
model.fit(x=X, 
    y=Y,
    batch_size=8,
    epochs=1000)

#============Test the model==============
output = model.predict(X[0].reshape(1,400,400,1))
output *= 128
# Output colorizations
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result.png", lab2rgb(cur))
imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))

#============Save model=================
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#===========Resize test image============
for filename in os.listdir('Image/Test'):
    temp = Image.open('Image/Test/' + filename)
    size = 400, 400
    temp.thumbnail(size, Image.ANTIALIAS)
    temp.save('Image/Test/Resized/' + filename, "JPEG")
    
#===========Get test images=============
test_images = []
for filename in os.listdir('Image/Test/Resized'):
    temp = np.array(img_to_array(load_img('Image/Test/Resized/'+filename)), dtype=float)
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
    test_images.append(np.array(temp, dtype=float))
    
#===========Transform test images=======
test_X = [rgb2lab(1.0/255*x)[:,:,0] for x in test_images]
test_X = [x.reshape(400, 400, 1) for x in test_X]
test_X = np.array(test_X)

#===========Colourise===================
for i in range(0,len(test_X)):
    output = model.predict(test_X[i].reshape(1,400,400,1))
    output *= 128
    # Output colorizations
    cur = np.zeros((400, 400, 3))
    cur[:,:,0] = test_X[i][:,:,0]
    cur[:,:,1:] = output[0]
    imsave("test{}_img_result.png".format(i+1), lab2rgb(cur))
    imsave("test{}_img_gray_version.png".format(i+1), rgb2gray(lab2rgb(cur)))