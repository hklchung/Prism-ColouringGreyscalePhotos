<br />
<p align="center">
  <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Light_dispersion_conceptual_waves.gif/330px-Light_dispersion_conceptual_waves.gif" height="100">
  </a>

  <h3 align="center">Colouring Greyscale Photos</h3>

  </p>
</p>

## About the Project
Convolutional Neural Networks (CNN) are commonly used for computer vision. In this project, I built a CNN that can turn black and white (greyscale) images into coloured images. Results from after 100 epochs of training are displayed below.

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

A training set of 1,187 images, taken from my personal photo album consisting of outdoor, indoor, people at close up and various inanimate objects were used to train the CNN model. The images displayed above belong to a test set of 8 greyscale images scraped from various online sources and the colours were given by the final model after training 100 epochs.

## Dependencies
* Keras
* Scikit-Image
* PIL
* Tensorflow-gpu
* Matplotlib

## Known issues
* Training may take a very long time if you do not have a GPU available
* If you have previously installed tensorflow-gpu with pip, tensorflow may be unable to detect your GPU. To overcome this issue, first uninstall tensorflow-gpu, then reinstall with conda.

