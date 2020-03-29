[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![Keras 2.3.1](https://img.shields.io/badge/keras-2.3.1-green.svg?style=plastic)
![TensorFlow-GPU 2.1.0](https://img.shields.io/badge/tensorflow_gpu-2.1.0-green.svg?style=plastic)
![Scikit Image 0.15.0](https://img.shields.io/badge/scikit_image-0.15.0-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

<br />
<p align="center">
  <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Light_dispersion_conceptual_waves.gif/330px-Light_dispersion_conceptual_waves.gif" height="100">
  </a>

  <h3 align="center">Colouring Greyscale Photos</h3>

  </p>
</p>

<p align="center">
  Prism - Using convolutional neural network to colourise greyscale photos.
    <br />
    <a href="https://github.com/hklchung/TravelPlanner"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hklchung/TravelPlanner">View Demo</a>
    ·
    <a href="https://github.com/hklchung/TravelPlanner/issues">Report Bug</a>
    ·
    <a href="https://github.com/hklchung/TravelPlanner/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [Contact](#contact)
* [Known Issues](#known-issues)

<!-- ABOUT THE PROJECT -->

## About the Project
Convolutional Neural Networks (CNN) are commonly used for computer vision. In this project, I built a CNN that can turn black and white (greyscale) images into coloured images. Results from after 500 epochs of training are displayed below.

### Nature
Original Image (B&W)       |  Prism Effect (Colourised)
:-------------------------:|:-------------------------:
![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_100epoch/test1_img_gray_version.png?raw=true)  |  ![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_100epoch/test1_img_result.png?raw=true)
![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_100epoch/test7_img_gray_version.png?raw=true)  |  ![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_100epoch/test7_img_result.png?raw=true)

### People
Original Image (B&W)       |  Prism Effect (Colourised)
:-------------------------:|:-------------------------:
![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_500epoch/test19_img_gray_version.png?raw=true)  |  ![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_500epoch/test19_img_result.png?raw=true)
![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_500epoch/test20_img_gray_version.png?raw=true)  |  ![](https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/Result_500epoch/test20_img_result.png?raw=true)

Prism will take images of any size, rescale and padded (with zeros) to 400x400 and convert from RGB to CIELAB colour space. The CIELAB colour space consists of 3 layers:
- the lightness from black (0) to white (100);
- from green (−) to red (+);
- and from blue (−) to yellow (+).

A training set of 1,187 images, taken from my personal photo album consisting of outdoor, indoor, people at close up and various inanimate objects were used to train the CNN model. The images displayed above belong to a test set of 8 greyscale images scraped from various online sources and the colours were given by the final model after training 500 epochs.

Please note that this is the model that is made available in the current release and Prism uses this model by default to colourise user photos. The architecture of the model is given below.
<p align="center">
  <img src="https://github.com/hklchung/Prism-ColouringGreyscalePhotos/blob/master/prism_cnn.png?raw=true" height="900">
</p>

<!-- GETTING STARTED -->

## Getting Started
Hope you are now excited with testing out Prism on your machine. To get started, please follow the below guidelines on prerequisites and installation.

<!-- PREREQUISITES -->

### Prerequisites
* Keras==2.3.1
* Scikit-Image==0.15.0
* PIL==6.2.0
* Tensorflow-gpu==2.1.0
* Numpy==1.18.2

<!-- INSTALLATION -->

### Installation
1. Fork and star this repo ;)
2. Create a folder on your machine for your project
2. Inside the folder right-click and select Git Bash Here
3. Git clone this repo into the folder by running the below command
```sh
git clone https://github.com/hklchung/Prism-ColouringGreyscalePhotos.git
```
4. Go inside the folder Prism and create the below folders as displayed    
<pre>
   L Prism
      L Image
         L Train
         L Test
      L Model
      L Result
</pre>
- Image folder contains the Train and Test folders
  - Train folder contains all images used for training a model (not required if you are not training a model)
  - Test folder contains all images that you would like to colourise
- Model folder stores all user trained models
- Result folder stores all colourised images

<!-- USAGE -->

## Usage
### Spyder
In this short tutorial, I will walk you through how you can get Prism to work on Spyder
1. Import Prism package and run Prism
```python3
from Prism import *
Prism()
```
2. There is a 'Train' mode and a 'Test' mode
3. Continue reading if you are only interested in colourising greyscale photos/images, otherwise skip to step 4
  - First go to Image/Test folder and place ~10 greyscale images that you would like to colourise
  - Alternatively you can stick with the ones that are provided with this package
  - Will you be training a model today? [y/n]: enter n
  - Please select your desired output dimensions: I recommend 400 x 400, enter 2
  - Do you wish to use the Prism pre-trained model? [y/n]: enter y
  - When the program stops, you will see the colourised photos/images in the Result folder
4. For training, follow the below steps
  - Will you be training a model today? [y/n]: enter y
  - How many epochs will your model train? [enter int between 1-1000]: I recommend picking 1 for the first run
  - Please select your desired output dimensions: I recommend 400 x 400, enter 2
  - When the program stops, you will see your model weights saved in the Model folder

### Bash
Alternatively you can run Prism through Bash
1. Give permission to execute the scripts
```sh
chmod 755 Prism.py
chmod 755 main.py
```
2. Execute main.py
```sh
python3 main.py
```
3. All the subsequent steps are as per above outlined in the Spyder instructions
  
<!-- CONTRIBUTING -->

## Contributing
I welcome anyone to contribute to this project so if you are interested, feel free to add your code.

<!-- CONTACT -->

## Contact
* [Leslie Chung](https://github.com/hklchung)

<!-- KNOWN ISSUES -->

## Known Issues
* Training may take a very long time if you do not have a GPU available
* If you have previously installed tensorflow-gpu with pip, tensorflow may be unable to detect your GPU. To overcome this issue, first uninstall tensorflow-gpu, then reinstall with conda.
* If you are running Prism via Terminal (or Bash) you may encounter this error "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above." If this happens, you can uninstall the cuDNN and the CUDA that was installed by Conda.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/hklchung/Prism-ColouringGreyscalePhotos.svg?style=flat-square
[contributors-url]: https://github.com/hklchung/Prism-ColouringGreyscalePhotos/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hklchung/Prism-ColouringGreyscalePhotos.svg?style=flat-square
[forks-url]: https://github.com/hklchung/Prism-ColouringGreyscalePhotos/network/members
[stars-shield]: https://img.shields.io/github/stars/hklchung/Prism-ColouringGreyscalePhotos.svg?style=flat-square
[stars-url]: https://github.com/hklchung/Prism-ColouringGreyscalePhotos/stargazers
[issues-shield]: https://img.shields.io/github/issues/hklchung/Prism-ColouringGreyscalePhotos.svg?style=flat-square
[issues-url]: https://github.com/hklchung/Prism-ColouringGreyscalePhotos/issues
