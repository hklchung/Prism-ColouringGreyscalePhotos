from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import pathlib
from PIL import Image, ImageOps
import random
import tensorflow as tf
from tensorflow.python.client import device_lib

class Prism:
    def __init__(self):
        self.dimls = [128, 256, 400]
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
                print("[Error] Missing one or both Train/Test folders in path")
                print("[Error] The required structure is:")
                print("[Error]     Your project folder")
                print("[Error]          L Images")
                print("[Error]              L Train")
                print("[Error]              L Test")
                print("[Error] Exiting now...")
                exit()
            
            # Resize images
            print("Resizing images in the Train folder")
            self.resize_image(self.train_pwd)
            print("Resizing images in the Test folder")
            self.resize_image(self.test_pwd)
            
            # Load train images
            print("Loading the resized images for training")
            images = self.load_image(self.train_pwd)
        
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
            
            # Detect current path and see if train/test folders available
            self.pwd = str(pathlib.Path().absolute())
            self.test_pwd = self.pwd + '\\Image\\Test'
            if os.path.isdir(self.test_pwd):
                self.test_size = len([x for x in os.listdir(self.test_pwd) if os.path.isfile(os.path.join(self.test_pwd, x))])
                print("Found 'Test' folder with {} files".format(self.test_size))
            else:
                print("[Error] Missing Test folder in path")
                print("[Error] The required structure is:")
                print("[Error]     Your project folder")
                print("[Error]          L Images")
                print("[Error]              L Train")
                print("[Error]              L Test")
                print("[Error] Exiting now...")
                exit()
            
            # Resize images
            print("Resizing images in the Test folder")
            self.resize_image(self.test_pwd)
            
            # Load images
            print("Loading the resized images")
            images = self.load_image(self.test_pwd)
                
    def resize_image(self, path):
        resize_folder = path + '\\Resized'
        if not os.path.exists(resize_folder):
            os.mkdir(resize_folder)
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        for filename in files:
            temp = Image.open(path + "\\" + filename)
            size = self.dims, self.dims
            temp.thumbnail(size, Image.ANTIALIAS)
            temp.save(resize_folder + "\\" + filename, "JPEG")
            
    def load_image(self, path):
        path = path + "\\Resized"
        images = []
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        for filename in files:
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