# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:11:02 2016

@author: Leonard
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:26:44 2016

@author: Leonard
"""

import loadIMG2 as ld
import numpy as np
from numpy.linalg import inv

#np.seterr(all = 'ignore')
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
        else: output[i] = -1
    return output

def Error(result):
    Error = 0
    for i in np.arange(len(result)):
        if result[i] !=0:
            Error = Error + 1
    return Error/len(result)

#Creation des sets train et test
nb_ex = 30000
imgTrain, lblTrain = ld.load_mnist('training')
imgTrain = imgTrain[0:nb_ex]
lblTrain = lblTrain[0:nb_ex]
imgSize, imgVecTrain = matrixToVectors(imgTrain)

imgTest, lblTest = ld.load_mnist('testing')
imgSize, imgVecTest = matrixToVectors(imgTest)

#Paramètres et matrices
D = 3000
lam = 0.1
R = np.random.randn(imgSize, D)/1000
gam = np.identity(D)


###Training
#Creation de la matrice F
S = imgVecTrain.dot(R)
F = sigmoid(S)

#Creation des sorties normalisées et des estimateurs
outputTrain = []
theta = []

for i in np.arange(10):
    outputTrain.append(makeOutput(lblTrain, i))
    theta.append(inv(F.T.dot(F) + lam*gam).dot(F.T).dot(outputTrain[i]))


###Testing
outputTest = []
for i in np.arange(10):
    outputTest.append(sigmoid(imgVecTest.dot(R)).dot(theta[i]))

result = np.argmax(outputTest, 0) - lblTest[:,0]
error = Error(result)
#count = np.bincount(labels.ravel()) #pour compter le nombre d'éléments de l'ensemble de test de chaque numéro

