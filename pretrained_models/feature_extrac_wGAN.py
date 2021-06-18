#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:15:47 2021

@author: maelle
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Model
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import import_data
import wGAN
import retraining

# data = "Nagao"
data = "DIC"
# data = "CellCognition"

path_results = "/home/maelle/Documents/Stage_m2/Results/features_extrac/"

base_model = keras.models.load_model("/home/maelle/Documents/Stage_m2/Models/d_model.h5")
base_model.summary()
dim = 72
batch_size = 8
mode = "FT"
per_train_data = 0.8


outputs = []
conv_layers = []
for layer in (base_model.layers):
    if "conv" in layer.name:
        conv_layers.append(layer.name)
        outputs.append(layer.output)
    
model = Model(inputs=base_model.inputs, outputs=outputs)


if data == "Nagao": #for nagao images, there are 4 different datasets
    dataset_list =  ["HeLa_Hoechst-EB1", "RPE1_Hoechst", "HeLa_Hoechst-GM130","NIH3T3_Cilia"] 

else:        
    dataset_list =  ["only one dataset"] 
    
for dataset in dataset_list:
    if data == "Nagao":                
        path = "/home/maelle/Documents/Stage_m2/data/"+dataset
        x, y = import_data.nagao(path, dim, wgan=True)
        if dataset == "NIH3T3_Cilia":
            class_names = ["Cilia", "notCilia"]
        else:            
            class_names = ["G2", "notG2"]
        
    elif data == "CellCognition":
        dataset = "CellCognition"
        class_names = ['AA', 'BA','I','J']
        x, y, img_names  = import_data.cell_cognition(dim, wgan=True)

    elif data == "DIC":
        dataset = "DIC"
        class_names = ["AA","NEBD","Meta"]
        x, y, img_names  = import_data.dic(dim, wgan=True)
        
    # Get one image of each class
    nb_classes = len(class_names)
    df = pd.DataFrame({'class':y.argmax(axis=1)})
    Samplesize = 1
        
    selected_idx = df.groupby('class', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:]).reset_index(level=[0,1])['level_1']
    selected_labels = y[selected_idx]
    selected_images = x[-selected_idx]
    
    # Removing these images and labels from the original dataset
    y = np.delete(y, selected_idx, axis=0)
    x = np.delete(x, selected_idx, axis=0)
    
    
    # Retraining the model:
    if mode == "FR":
        title_name = f"Features extraction of wGAN model after full retraining"
        model = wGAN.full_retraining(base_model, nb_classes)        
        save_path = f'{path_results}/{dataset}_full_retrain'
        
    elif mode == "TL":       
        title_name = f"Features extraction of wGAN model after transfer learning"
        model = wGAN.transfer_learning(base_model, nb_classes)        
        save_path = f'{path_results}/{dataset}_TL'
        
    else:
        title_name = f"Features extraction of wGAN model after fine tuning"
        model = wGAN.fine_tuning(base_model, nb_classes) 
        save_path = f'{path_results}/{dataset}_FT'   
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1-per_train_data)
    acc, epo, _ = retraining.fitting(model, x_train, y_train, x_test, y_test, batch_size, save_path, title_name, '_', class_names, dataset)
    
    for img, label in selected_images, selected_labels:       
        feature_maps = model.predict(img)    
        
        square = 8
        j = 0
        for fmap in feature_maps:
            for ix in range(3):
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                ix +=1 
        plt.title(f"{title_name} for one image of class {label}")
        plt.savefig()   
        plt.show()
        j+=1
