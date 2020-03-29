"""
Copyright (c) 2020, Heung Kit Leslie Chung
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from tensorflow.python.client import device_lib
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from PIL import Image, ImageOps
import numpy as np
import os
import pathlib
import random
from datetime import datetime
from tqdm import tqdm

class Prism:
    def __init__(self):
        self.dimls = [128, 256, 400]
        self.start = datetime.utcnow()
        print("Welcome to Prism!")
        print("Will you be training a model today? [y/n]")
        self.train = input()
        if self.train == 'y':
            # Training epochs
            print("How many epochs will your model train? [enter int between 1-1000]")
            self.epochs = int(round(float(input())))
            
            # Training + Output dimensions
            print("Please select your desired output dimensions.")
            print("[0: 128 x 128]")
            print("[1: 256 x 256]")
            print("[2: 400 x 400]")
            self.dim = int(round(float(input())))
            while self.dim not in range(0, 3):
                print("You have not entered a valid option, please try again.")
                self.dim = int(round(float(input())))
                if self.dim in range(0, 3):
                    break
            self.dims = self.dimls[self.dim]
            
            # Training CPU/GPU
            self.trainer = 'CPU'
            for i in range(0, len(device_lib.list_local_devices())):
                if 'GPU' in device_lib.list_local_devices()[i].name:
                    print("Prism found a connected GPU")
                    self.trainer = 'GPU'
            print("The model will be trained on {}".format(self.trainer))
            
            # Detect current path and see if train/test folders available
            self.pwd = str(pathlib.Path().absolute())
            self.train_pwd = self.pwd + '\\Image\\Train'
            self.test_pwd = self.pwd + '\\Image\\Test'
            if os.path.isdir(self.train_pwd) & os.path.isdir(self.test_pwd):
                self.train_size = len([x for x in os.listdir(self.train_pwd) if os.path.isfile(os.path.join(self.train_pwd, x))])
                self.test_size = len([x for x in os.listdir(self.test_pwd) if os.path.isfile(os.path.join(self.test_pwd, x))])
                print("Found 'Train' folder with {} files".format(self.train_size))
                print("Found 'Test' folder with {} files".format(self.test_size))
            else:
                print("[Error] Missing one or both Train/Test folders in path.")
                print("[Error] The required structure is:")
                print("[Error]     Your project folder")
                print("[Error]          L Images")
                print("[Error]              L Train")
                print("[Error]              L Test")
                print("[Error] Exiting now...")
                exit()
            
            # Resize images
            print("Resizing images in the Train folder...")
            self.resize_image(self.train_pwd)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60)) 
            print("Resizing images in the Test folder...")
            self.resize_image(self.test_pwd)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Load train images
            print("Loading the resized images for training...")
            images = self.load_image(self.train_pwd)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Transform train images
            print("Transforming the images for training...")
            X, Y = self.train_transform_image(images)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Initialise model
            print("Initialising model.")
            model = self.initialise_model()
            
            # Train model
            print("Training model...")
            print("Settings: batch size = 8, no. of epochs = {}".format(self.epochs))
            model.fit(x=X, 
                      y=Y,
                      batch_size=8,
                      epochs=self.epochs)
            
            # Save model
            print("Saving model...")
            self.save_model(model)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
        if self.train == 'n':
            # Training + Output dimensions
            print("Please select your desired output dimensions.")
            print("[0: 128 x 128]")
            print("[1: 256 x 256]")
            print("[2: 400 x 400]")
            self.dim = int(round(float(input())))
            while self.dim not in range(0, 3):
                print("You have not entered a valid option, please try again.")
                self.dim = int(round(float(input())))
                if self.dim in range(0, 3):
                    break
            self.dims = self.dimls[self.dim]
            
            # Detect current path and see if train/test folders available
            self.pwd = str(pathlib.Path().absolute())
            self.test_pwd = self.pwd + '\\Image\\Test'
            if os.path.isdir(self.test_pwd):
                self.test_size = len([x for x in os.listdir(self.test_pwd) if os.path.isfile(os.path.join(self.test_pwd, x))])
                print("Found 'Test' folder with {} files".format(self.test_size))
            else:
                print("[Error] Missing Test folder in path.")
                print("[Error] The required structure is:")
                print("[Error]     Your project folder")
                print("[Error]          L Images")
                print("[Error]              L Train")
                print("[Error]              L Test")
                print("[Error] Exiting now...")
                exit()
            
            # Resize images
            print("Resizing images in the Test folder...")
            self.resize_image(self.test_pwd)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Load images
            print("Loading the resized images...")
            images = self.load_image(self.test_pwd)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Transform train images
            print("Transforming the images for training...")
            X = self.test_transform_image(images)
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Load model
            print("Loading model...")
            print("Do you wish to use the Prism pre-trained model? [y/n]")
            self.model_choice = input()
            if self.model_choice == 'y':
                model = self.initialise_model()
                model.load_weights('model_v2.h5')
            if self.model_choice == 'n':
                modells = [x for x in os.listdir(self.pwd + '\\Model') if os.path.isfile(os.path.join(self.pwd + '\\Model', x)) & x.endswith(".h5")]
                print("Please select the model you would like to use.")
                for i in range(0, len(modells)):
                    print("[{}: {}]".format(i, modells[i]))
                self.user_model = int(round(float(input())))
                model = self.initialise_model()
                model.load_weights(self.pwd + '\\Model\\' + modells[self.user_model])
            print("---Time taken since program started: {} minutes".format((datetime.utcnow() - self.start).seconds/60))
            
            # Colourise images
            self.result_pwd = self.pwd + '\\Result'
            self.colourise(model, X, self.result_pwd)
                
    def resize_image(self, path):
        resize_folder = path + '\\Resized'
        if not os.path.exists(resize_folder):
            os.mkdir(resize_folder)
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        for filename in tqdm(files):
            temp = Image.open(path + "\\" + filename)
            size = self.dims, self.dims
            temp.thumbnail(size, Image.ANTIALIAS)
            temp.save(resize_folder + "\\" + filename, "JPEG")
            
    def load_image(self, path):
        path = path + "\\Resized"
        images = []
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        for filename in tqdm(files):
            temp = np.array(img_to_array(load_img(path + "\\" + filename)), dtype=float)
            hor = self.dims - temp.shape[0]
            ver = self.dims - temp.shape[1]
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
        return(images)
        
    def train_transform_image(self, images):
        X = [rgb2lab(1.0/255*x)[:,:,0] for x in images]
        Y = [rgb2lab(1.0/255*x)[:,:,1:] for x in images]
        Y = [x/128 for x in Y]
        X = [x.reshape(self.dims, self.dims, 1) for x in X]
        Y = [x.reshape(self.dims, self.dims, 2) for x in Y]
        X = np.array(X)
        Y = np.array(Y)
        return(X, Y)
        
    def test_transform_image(self, images):
        X = [rgb2lab(1.0/255*x)[:,:,0] for x in images]
        X = [x.reshape(self.dims, self.dims, 1) for x in X]
        X = np.array(X)
        return(X)
        
    def initialise_model(self):
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
        return(model)
    
    def save_model(self, model):
        model_pwd = self.pwd + '\\Model'
        model_json = model.to_json()
        with open(model_pwd + "\\model_{}.json".format(str(datetime.date(datetime.now()))), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_pwd + "\\model_{}.h5".format(str(datetime.date(datetime.now()))))
        
    def colourise(self, model, X, pwd):
        for i in tqdm(range(0,len(X))):
            output = model.predict(X[i].reshape(1,self.dims,self.dims,1))
            output *= 128
            # Output colorizations
            cur = np.zeros((self.dims, self.dims, 3))
            cur[:,:,0] = X[i][:,:,0]
            cur[:,:,1:] = output[0]
            imsave(pwd + "\\" + "test{}_img_result.png".format(i+1), lab2rgb(cur))