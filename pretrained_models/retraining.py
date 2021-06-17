#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:05:28 2021

@author: maelle
"""



import time

import os

import keras

import numpy as np
import pandas as pd


import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical #from keras.utils import np_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations

from keras.optimizers import Adam

import figures

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import getopt, sys
from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold, train_test_split

from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from mpl_toolkits.axes_grid1 import ImageGrid


def check_neural_network(model):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print(model.summary())
    for layer in model.layers:
        print(layer.name," ", layer.trainable)  
        
        
def fitting(model, x_train, y_train, x_test, y_test, batch_size, save_path, title_name, kf, class_names, dataset):    
    
    # Compilation
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5),metrics=["accuracy"])
    
    model.summary()
    
    # Running
    early_stopping = EarlyStopping(patience=10, verbose=1)
    H = model.fit(x_train, y_train, batch_size=batch_size, verbose=1, validation_data=(x_test,y_test), epochs=150, callbacks=[early_stopping])           
    
    # summarize history for accuracy  and loss
    figures.plot_accuracy(H, save_path+f"_acc_classes_{batch_size}_{kf}.png", title="Accuracy curve")
    figures.plot_loss(H, save_path+f"_loss_classes_{batch_size}_{kf}.png", title="Loss curve")
    
    preds = model.predict(x_test)    

    tab = pd.DataFrame({'True Label':y_test.argmax(axis=1), 'Predicted Label': preds.argmax(axis=1)})                
    tab['Confusion'] = tab['True Label'].astype(str)+tab['Predicted Label'].astype(str)   
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            
            if i == j:
                title = f"Cells in class {class_names[i]} well classified"
                
            else:
                title = f"Cells in class {class_names[i]} classified as class {class_names[j]}"                
            
            figures.generate_mosaic(tab, f"{i}{j}", x_test, title, save_path+"_"+str(kf))
    

    acc = accuracy_score(y_test.argmax(axis=1), preds.argmax(axis=1))
    epo = len(H.history['accuracy'])       
    
    print('Accuracy score on test dataset',acc)
    print('classification_report',classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))
    
    figures.plot_confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1), classes=np.array(class_names), title='Confusion Matrix')
    plt.title(f'confusion matrix for classification of {dataset} dataset')
    plt.savefig(f'{save_path}_confusion_matrix_{dataset}_{batch_size}_kf_{kf}.png')
    plt.clf()     
    
    return(acc,epo)