#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:37:15 2021

@author: maelle
"""



import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical
from numpy import expand_dims

import matplotlib.pyplot as plt


def rgb2gray(rgb, dim, chan=None):
    """    
    Convert an colored image (3 channels) to a grayscale image
    Parameters
    ----------
    rgb : matrix
        matrix of pixels which represents the image
    dim : int
        image size (dim x dim)
    chan : int, optional
        id of the only channel which is used.
        The default is None (if the 3 channels are merged)

    Returns
    -------
    gray_reshaped : matrix
        matrix with 1 channel instead of 3

    """
    if not chan:
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    else:
        #to keep only one channel
        chan = to_categorical(chan, 3)       
        gray = np.dot(rgb[...,:3], chan)
    # gray_reshaped = np.reshape(gray, (dim, dim, 1))
    return gray


def nagao(path, dim, chan=None):
    """    

    Parameters
    ----------
    path : str
        dataset directory
    dim : int
        dimension of one image
    wgan : bool
        False if data must be in [-1,1]
        True if data must be in [0,1]

    Returns
    -------
    None.

    """
    nb_classes = 2
    
    x_train = np.load(path+"/data/x_train.npy")
    y_train = np.load(path+"/data/y_train.npy")
    x_test = np.load(path+"/data/X_test.npy")
    y_test = np.load(path+"/data/Y_test.npy")  
    
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))  
    
    x = resize(x, dim, chan)
    y = to_categorical(y, nb_classes) 
        
    return(x, y) 


def resize(dataset, dim, chan=None):
    """    

    Parameters
    ----------
    dataset : matrix
        images to be resized
    dim : int
        dimension of one image
    wgan : bool
        False if data must be in [-1,1]
        True if data must be in [0,1]

    Returns
    -------
    Resized images

    """
    dataset_resized = []
    
    for img in dataset:
        img_resized = cv2.resize((img),(dim,dim))
        gray = img_resized     
        dataset_resized.append(gray)
        
    return(dataset_resized)


dataset_list =  ["RPE1_Hoechst"] 
dim = 72

for dataset in dataset_list:          
    path = "/home/maelle/Documents/Stage_m2/data/"+dataset
    x, y = nagao(path, dim)
    img_names = None
    if dataset == "NIH3T3_Cilia":
        class_names = ["Cilia", "notCilia"]
    else:            
        class_names = ["G2", "notG2"]


        
        
matrice = x[1]


hist = np.histogram(gray)
hist
plt.hist(hist)


list_histo = []

for matrice in x[:100]:
    hist = np.histogram(matrice)
    
    red = hist[0]
    green = hist[1]
    
    if red in list_histo:
        print("ok")
    
    list_histo.append(red)
    
    
    
matrice = x[40]
plt.imshow(matrice)
np.histogram(matrice)
matrice = x[41]
hist = np.histogram(matrice)
plt.imshow(matrice)
matrice = x[42]
np.histogram(matrice)
plt.imshow(matrice)
matrice = x[43]
np.histogram(matrice)
plt.imshow(matrice)

matrice = x[44]
np.histogram(matrice)
plt.imshow(matrice)

matrice = x[45]
np.histogram(matrice)
plt.imshow(matrice)

matrice = x[46]
np.histogram(matrice)
plt.imshow(matrice)

matrice = x[47]
np.histogram(matrice)
plt.imshow(matrice)

matrice = x[48]
np.histogram(matrice)
plt.imshow(matrice)





for mat in x:
    hist = np.histogram(matrice)
    rouge = hist[0]
    vert = hist[1]
    
    
nb_images = len(x)

import pandas as pd

red_mean = []
green_mean = []

for img in x:
    hist = np.histogram2d(img)
    red = hist[0]
    green = hist[1]
    
    red_mean.append(red)
    green_mean.append(green)
    
    
df = pd.DataFrame() 
df["red"] = red_mean
df["green"] = green_mean   

r0 = red_mean[0]
r1 = red_mean[1]

import cv2

img = x[1]
np.histogram2d(img[0], img[1])

cv2.calcHist([img], [0], None, [8], [0,255])

cv2.compareHist(r0, r1, cv2.HISTCMP_CORREL) 

# for i in range(1, nb_images):
#     matrice = x[i]
#     hist = np.histogram(matrice)
#     red = hist[0]
#     green = hist[1]
#     if 


cv2.calcHist([img], [0,1], None, [180,256], [0,180,0,256])
