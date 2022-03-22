# Bilderbazaar
Bilderbazaar - A Pix2Pix practical example for crawling scanned Articles to extract Metadata and Visual Information with Image-to-Image Translation


This relies heaviliy on the Pater of "Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros" to create Image-to-Image Translation with Conditional Adersarial Networks (https://github.com/phillipi/pix2pix)

# Prerequisites
Linux / Win / OSX
NVIDIA GPU + CUDA CuDNN

Python 3.6 
(it is recommendet to clone the pix2pix github first and install the Environment from the included yaml configuration file)
Related Python Packages:


# Start of this Project(History)

https://codingdavinci.de/projekte/bilderbazar-eine-zeitmaschine (German)

# First Iteration of using a Neural Net to extract Foreground Images 

Examplepic Bazaar

The first Idea to extract separated images from the different Pages of the magazines was to use an existing Neural Net available for Torchvision. Seeing The foreground Layer of all the different paintings / drawings on a page and the text as the background layer.

First tests with DeepLabV3-ResNet101/Resnet50 ( see colab playbook: https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Remove_Image_Background_DeepLabV3.ipynb)

Source: Rembg
https://github.com/danielgatis/rembg
Picture

Input

Output  


Conclusion: Failed to deliver 
Solution: Picture to Picture tranlation

# Second Iteration of using a Neural Net to extract Foreground Images based on reference Pictures of before/after states

Theory: Pix2Pix is used to do complicated Image to Image translations based on training different Neural Nets at the same time (See paper).

Gan....

loss function....

Example from pix2pix : Google Streetview to Satellite Pictures


