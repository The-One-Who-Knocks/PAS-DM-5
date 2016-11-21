# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:26:44 2016

@author: Leonard
"""

import loadIMG2 as ld
import numpy as np
from numpy.linalg import inv

def matrixToVectors(matrix): #.ravel or .flatten ?
    a, b, c = matrix.shape
    imgSize = b*c
    vectors = np.zeros((a, imgSize), dtype=np.uint8)
    for i in np.arange(a):
        vectors[i] = matrix[i].ravel()
    return imgSize, vectors

def sigmoid(x):
    return 1/(1+np.exp(-x))

def makeOutput(labels, digit):
    length = len(labels)
    output = np.zeros(length)
    for i in np.arange(length):
        if labels[i] == digit:
            output[i] = 1
    return output

def cleaning(x):
    for i in np.arange(len(x)):
        if x[i] > 0.9:
            x[i] = 1
        else : x[i] = 0
    return x

def Error(labels, outputTest, digit):
    Error = 0
    for i in np.arange(len(labels)):
        if labels[i] == digit and outputTest[i] == 0:
            Error = Error + 1
        if labels[i] != digit and outputTest[i] == 1:
            Error = Error + 1
    return Error
#Training
imgTrain, lblTrain = ld.load_mnist('training')
nb_ex = 2000
imgTrain = imgTrain[0:nb_ex]
lblTrain = lblTrain[0:nb_ex]

imgSize, imgVecTrain = matrixToVectors(imgTrain)
outputTrain = makeOutput(lblTrain, 2)

D = 2000
R = np.random.randn(imgSize, D)
lam = 0.1
gam = np.identity(D)

F = imgVecTrain.dot(R)
theta = inv(F.T.dot(F) + lam*gam).dot(F.T).dot(outputTrain)

#Testing
imgTest, lblTest = ld.load_mnist('testing')
imgTest = imgTest[0:nb_ex]
lblTest = lblTest[0:nb_ex]
imgSize, imgVecTest = matrixToVectors(imgTest)
outputTest = sigmoid(imgVecTest.dot(R).dot(theta))
outputTest = cleaning(outputTest)
Error = Error(lblTest, outputTest, 2)


#count = np.bincount(labels.ravel()) #pour compter le nombre d'éléments de l'ensemble de test de chaque numéro

