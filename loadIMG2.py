# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:18:40 2016

@author: Leonard
"""
from array import array as pyarray


import os, struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(dataset="training", path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    Source : http://g.sweyla.com/blog/2012/mnist-numpy/
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
           
    images = np.zeros((size, rows, cols), dtype=np.uint8)
    labels = np.zeros((size, 1), dtype=np.int8)
    for i in range(size):
        images[i] = np.array(img[i*rows*cols : (i+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[i]

    return images, labels
    

images, labels = load_mnist('training')

#Ploting
n_img = 6300
pic = images[n_img]
lab = labels[n_img]
plt.title('Label is {label}'.format(label=lab))
plt.imshow(pic, cmap='gray')
plt.show()
