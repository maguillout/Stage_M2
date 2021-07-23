#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:12:49 2021

@author: maelle
"""

############# Packages importation ###############



import os
import cv2
from PIL import Image
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

import matplotlib.pyplot as plt

############ Datasets creation  #########################################

dim = 512
dataset = "/home/maelle/Documents/Stage_m2/data/DIC"

def crop_images(classe):
    class_dir = "/home/maelle/Documents/Stage_m2/data/DIC"+classe
    img_names = os.listdir(class_dir)
    datagen = ImageDataGenerator(rotation_range=45)
    for name in img_names: 
        img_read=cv2.resize(cv2.imread(class_dir+name,cv2.IMREAD_UNCHANGED),(dim,dim))
        samples = expand_dims(img_read, 0)
        it = datagen.flow(samples, batch_size=1)
        for i in range(8):
            batch = it.next()
            image = batch[0].astype('uint8')
            img_crop = image[105:407,105:407] #get only nucleus (middle of the image)
            img_file = Image.fromarray(np.uint8(img_crop))                                   
            img_file.save(f"/home/maelle/Documents/Stage_m2/data/DIC_cropped_augmnted{classe}_{i}{name}")
        print(f"image {name} has been cropped and augmented by 8")





def augmentation_mito():
    class_names = ["control", "treated"]
    path = "/home/maelle/Documents/Stage_m2/data/data_mito/"  
    path_results = "/home/maelle/Documents/Stage_m2/data/mito_rotations"
    for classe in class_names:
        img_names = os.listdir(path+classe)     
        if not os.path.exists(path_results+"/"+classe):
            os.mkdir(path_results+"/"+classe)   
        
        datagen = ImageDataGenerator(rotation_range=45)
        for name in img_names: 
            img_array = cv2.imread(path+classe+"/"+name,cv2.IMREAD_COLOR)
            samples = expand_dims(img_array, 0)
            it = datagen.flow(samples, batch_size=1)
            for i in range(8):
                batch = it.next()
                image = batch[0].astype('uint8')
                # plt.imshow(image)
                img_file = Image.fromarray(np.uint8(image))                                   
                img_file.save(f"{path_results}/{classe}/{i}_{name}")
            print(f"image {name} has been augmented by 8")
                
                
crop_images("/AA/")  
crop_images("/NEBD/")  
crop_images("/Meta/")  

augmentation_mito()
        
