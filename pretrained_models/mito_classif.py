#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:26:15 2021

@author: maelle
"""


import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import import_data
from numpy import expand_dims
from PIL import Image
# Utilities
import os
import getopt
import sys

# Neural networks
import keras
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import train_test_split

# Import functions from other files
import import_data
import retraining

import time
import classification_sGAN, classification_wGAN


class_names = ["control", "treated"]
dim = 72

path = "/home/maelle/Documents/Stage_m2/data/data_mito/"    

# def data_augmentation(x, y):    
#     images = []
#     labels = []
    
#     datagen = ImageDataGenerator(
# 	rotation_range=30,
# 	zoom_range=0.15,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	shear_range=0.15,
# 	horizontal_flip=True,
# 	fill_mode="nearest")
#     nb_images = len(x)
#     for i in range(nb_images):
#         lab = y[i]
#         img = x[i]
#         samples = expand_dims(img, 0)
#         it = datagen.flow(samples, batch_size=1)     
#         images.append(img)
#         labels.append(lab) 
#         for j in range(16):
#             batch = it.next()
#             image = batch[0].astype('uint8')
#             img_array=img_to_array(image)
#             plt.imshow(img_array, cmap='gray')
#             images.append(img_array)
#             labels.append(lab)    
            
#     return(np.array(images), np.array(labels))


def data_augmentation(x, y):    
    images = []
    labels = []
    
    datagen = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True)
    nb_images = len(x)
    for i in range(nb_images):
        lab = y[i]
        img = x[i]
        samples = expand_dims(img, 0)
        it = datagen.flow(samples, batch_size=1)     
        images.append(img)
        labels.append(lab) 
        for i in range(8):
            batch = it.next()
            image = batch[0].astype('uint8')
            img_array=img_to_array(image)
            images.append(img_array)
            labels.append(lab)   
    
    return(np.array(images), np.array(labels))
            
images = []
labels = []
lab = 0
path = "/home/maelle/Documents/Stage_m2/data/DIC_cropped_augmnted/"    
filenames = []
for class_dir in ["AA/","NEBD/","Meta/"]:
    img_names = os.listdir(path+class_dir)
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    for img in img_names:
        img_read = cv2.resize(cv2.imread(path+class_dir+img,cv2.IMREAD_UNCHANGED),(dim,dim))
        img_read = import_data.rgb2gray(img_read, dim) 
        samples = np.expand_dims(img_read, 0)            
        it = datagen.flow(samples, batch_size=1)
        images.append(img_read)
        labels.append(lab) 
        # for i in range(4): #rotation by 45°
        #     filenames.append(img)
        #     batch = it.next()
        #     image = batch[0].astype('uint8')
        #     img_array=img_to_array(image)
        #     # if wgan:               
        #     #     img_array=img_array/255.0 #for Wasserstein GAN, values must be in [0,1]  
        #     # else:
        #     #     img_array=(img_array-127.5)/127.5 #for classic GAN, must be in [-1,1]
        #     images.append(img_array)
        #     labels.append(lab) 
    lab += 1
    
x, y =data_augmentation(images, labels)

        
plt.imshow(img_array, cmap='gray')
plt.imshow(img_resized, cmap='gray')
plt.imshow(gray_reshaped, cmap='gray')
plt.imshow(img_resized, cmap='gray')
plt.imshow(img2, cmap='gray')
plt.imshow(image, cmap='gray')
plt.imshow(img, cmap='gray')


plt.imshow(x_train[0], cmap='gray')
plt.imshow(x_train[1], cmap='gray')
plt.imshow(x_train[2], cmap='gray')
plt.imshow(x_train[3], cmap='gray')
plt.imshow(x_train[4], cmap='gray')

def import_mito():    
    images = []
    labels = []    
    lab = 0
    for class_dir in class_names:
        img_names = os.listdir(path+class_dir)
        for name in img_names: 
            img_array = cv2.imread(path+class_dir+"/"+name, cv2.IMREAD_UNCHANGED)
            img_resized = cv2.resize(img_array,(dim,dim))
            gray_reshaped = np.reshape(img_resized, (dim, dim, 1))
            if wgan:               
                img_array=gray_reshaped/255.0 #for Wasserstein GAN, values must be in [0,1]  
            else:
                img_array=(gray_reshaped-127.5)/127.5 #for classic GAN, must be in [-1,1]                
            images.append(img_array)
            labels.append(lab) 
        lab += 1     
        
    return(images, labels)





for i in range(2):
    if i == 0:
        wgan = True
    else:
        wgan = False
    for j in range(3):     
        if j == 0:
            mode = "FT"
        elif j == 1:
            mode = "TL"
        else:
            mode = "FR"        
        
        # Model importation
        if wgan:    
            model_path = "/home/maelle/Documents/Stage_m2/Models/d_model_Trained_All_Labled_Images.h5"
            path_results = "/home/maelle/Documents/Stage_m2/Results/classification_mito"+"/wGAN"
        else:
            model_path = "/home/maelle/Documents/Stage_m2/Models/c_model_5400.h5"
            path_results = "/home/maelle/Documents/Stage_m2/Results/classification_mito"+"/sGAN"
        
        dataset = "Mitochondria images"
        nb_classes = len(class_names)
        
        batch_size = 8
        
        base_model=keras.models.load_model(model_path)
        base_model.trainable = False
        base_model.summary()   
        
        t0 = time.time() #starting timer  
        kf = 0      
        
        acc_kf = [] 
        epo_kf = []
        
        
        x, y = import_mito()     
        
        x_array = np.array(x)  # x and y must be lists for kfold splitting but must be arrays for training
        y_array = to_categorical(y, nb_classes)     
        
        # prepare cross validation
        kfold = StratifiedKFold(n_splits=10) 
        kfold.get_n_splits(x,y)         
        
        for train_index, test_index in kfold.split(x, y):           
       
            
            x_train = x_array[train_index]
            y_train = y_array[train_index]
            
            x_test = x_array[test_index]
            y_test = y_array[test_index]            
            
            print("Before data augmentation:")
            print(f"Train: {x_train.shape}")
            print(f"Test: {x_test.shape}")      
            
            x_train, y_train = data_augmentation(x_train, y_train)
            x_test, y_test = data_augmentation(x_test, y_test)
            
            print("After data augmentation:")
            print(f"Train: {x_train.shape}")
            print(f"Test: {x_test.shape}")
            
                
            if not wgan:
                
                if mode == "FR":
                    title_name = f"Full retraining for prediction of classes {class_names} for {dataset} with sGAN kfold - {kf}"
                    model = classification_sGAN.full_retraining(base_model, nb_classes)        
                    save_path = f'{path_results}/{dataset}_full_retrain'
                    
                elif mode == "TL":       
                    title_name = f"Transfer Learning for prediction of classes {class_names} for {dataset} with sGAN - kfold {kf}"
                    model = classification_sGAN.transfer_learning(base_model, nb_classes)        
                    save_path = f'{path_results}/{dataset}_TL'
                    
                else:
                    title_name = f"Fine Tuning for prediction of classes {class_names} for {dataset}  with sGAN - kfold {kf}"
                    model = classification_sGAN.fine_tuning(base_model, nb_classes) 
                    save_path = f'{path_results}/{dataset}_FT'            
         
                
            else:        
                if mode == "FR":
                    title_name = f"Full retraining for prediction of classes {class_names} for {dataset} with wGAN kfold - {kf}"
                    model = classification_wGAN.full_retraining(base_model, nb_classes)        
                    save_path = f'{path_results}/{dataset}_full_retrain'
                    
                elif mode == "TL":       
                    title_name = f"Transfer Learning for prediction of classes {class_names} for {dataset} with wGAN - kfold {kf}"
                    model = classification_wGAN.transfer_learning(base_model, nb_classes)        
                    save_path = f'{path_results}/{dataset}_TL'
                    
                else:
                    title_name = f"Fine Tuning for prediction of classes {class_names} for {dataset}  with wGAN - kfold {kf}"
                    model = classification_wGAN.fine_tuning(base_model, nb_classes) 
                    save_path = f'{path_results}/{dataset}_FT' 
                
                
            print(title_name)    
           
            
            acc, epo, tab = retraining.fitting(model, x_train, y_train, x_test, y_test, batch_size, save_path, title_name, kf, class_names, dataset)
            
            acc_kf.append(acc)
            epo_kf.append(epo)
            
            kf+=1
            
        

        if wgan:
            f = open(path_results+"/test_"+dataset+"_results_wGAN.txt","a")
        else:                
            f = open(path_results+"/test_"+dataset+"_results_sGAN.txt","a")  
        # tab.to_csv(f"{path_results}/test_{dataset}_results_table_kf_{kf}.csv",sep=';')
        f.write("mode\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_5\tepo_6\tepo_7\tepo_8\tepo_9\tacc_mean\tepo_mean\ttime\n")  
                
        f.write(f"{mode}\t")
        for i in range(10): #print accuracy and number of epochs for each fold
            f.write(f"{acc_kf[i]}\t")
            f.write(f"{epo_kf[i]}\t")  
        f.write(f"{round(np.mean(acc_kf),2)}(\u00B1{round(np.std(acc_kf),2)})\t")
        f.write(f"{round(np.mean(epo_kf),2)}(\u00B1{round(np.std(epo_kf),2)})\t")
    
        t1 = time.time() 
        f.write(f"{t1-t0}\n")
        f.close()      


