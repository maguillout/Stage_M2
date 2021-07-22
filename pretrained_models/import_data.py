#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:06:31 2021

@author: maelle
"""

import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical
from numpy import expand_dims


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
    gray_reshaped = np.reshape(gray, (dim, dim, 1))
    return gray_reshaped


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
        img_resized = rgb2gray(img_resized, dim, chan)
    return(np.array(dataset_resized))

# def cell_cognition(dim):
#     """   

#     Parameters
#     ----------
#     dim : int
#         dimension of one image
#     wgan : bool
#         False if data must be in [-1,1]
#         True if data must be in [0,1]

#     -------
#     Dataset splitted in train / test

#     """
#     path = "/home/maelle/Documents/Stage_m2/data/dataset"
#     class_names = ['AA', 'BA','I','J']
#     nb_classes = len(class_names)
    
    def create_dataset_cell_cognition(class_dir, label, dim):
        """
        
    
        Parameters
        ----------
        class_dir : str
            directory of the class folder
        label : int
            numeric label for this class
        dim : int
            dimension of one image
        wgan : bool
            False if data must be in [-1,1]
            True if data must be in [0,1]
    
        Returns
        -------
        images and labels
    
        """
        images = []
        labels = []
        img_names = os.listdir(class_dir)
        for img in img_names:
            img_read=cv2.resize(cv2.imread(class_dir+img,cv2.IMREAD_UNCHANGED),(dim,dim))
            img_array=img_to_array(img_read)
            images.append(img_array)
            labels.append(label)
            
        return(images, labels, img_names)

    

    for i in range(nb_classes):
        g = class_names[i]
        train_dir = path + "/train/"+g+"/"
        test_dir = path + "/test/"+g+"/"
        data_tr, labels_tr, tr_names = create_dataset_cell_cognition(train_dir, i, wgan, dim)
        data_ts, labels_ts, ts_names = create_dataset_cell_cognition(test_dir, i, wgan, dim)        
        
    x_train = np.array(data_tr)    
    y_train = np.array(labels_tr)    
    x_test = np.array(data_ts)    
    y_test = np.array(labels_ts)

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))  
    
    img_names = tr_names.append(ts_names)
        
    return(x, y, img_names)



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


def dic(dim):
    """
    

    Parameters
    ----------
    dim : int
        dimension of one image
    wgan : bool
        False if data must be in [-1,1]
        True if data must be in [0,1]
    Returns
    -------
    None.

    """    
    images = []
    labels = []
    lab = 0
    path = "/home/maelle/Documents/Stage_m2/data/DIC_cropped_augmnted/"    
    filenames = []
    for class_dir in ["AA/","NEBD/","Meta/"]:
        img_names = os.listdir(path+class_dir)
        for img in img_names:
            img_read = cv2.resize(cv2.imread(path+class_dir+img,cv2.IMREAD_UNCHANGED),(dim,dim))
            gray_reshaped = rgb2gray(img_read, dim)  
            images.append(gray_reshaped)
            gray_reshaped = gray_reshaped/255
            labels.append(lab) 
            filenames.append(img)
        lab += 1
        

        
    return(images, labels, filenames)

def import_mito(dim):
    """
    

    Parameters
    ----------
    dim : int
        dimension of one image
    wgan : bool
        False if data must be in [-1,1]
        True if data must be in [0,1]
    Returns
    -------
    None.

    """    
    class_names = ["control", "treated"]
    path = "/home/maelle/Documents/Stage_m2/data/data_mito/"   
    images = []
    labels = []    
    lab = 0
    filenames = []
    for class_dir in class_names:
        img_names = os.listdir(path+class_dir)
        for name in img_names: 
            filenames.append(name)
            img_array = cv2.imread(path+class_dir+"/"+name,cv2.IMREAD_COLOR)
            img_resized = cv2.resize((img_array),(dim,dim))
            gray = np.dot(img_resized[...,:3], [0.299, 0.587, 0.144])
            gray_reshaped = np.reshape(gray, (dim, dim, 1))            
            images.append(gray_reshaped)
            labels.append(lab) 
        lab += 1     
        
    return(images, labels, filenames)

def data_augmentation(x, y):    
    images = []
    labels = []    
    
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    nb_images = len(x)
    for i in range(nb_images):
        lab = y[i]
        img = x[i]
        samples = expand_dims(img, 0)
        it = datagen.flow(samples, batch_size=1)     
        for i in range(8):
            batch = it.next()
            image = batch[0].astype('uint8')
            img_array=img_to_array(image)
            images.append(img_array)
            labels.append(lab)      
    
    return(np.array(images), np.array(labels))


