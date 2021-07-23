#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:05:28 2021

@author: maelle
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.optimizers import Adam

import figures

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

import import_data


def check_neural_network(model):
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
    
    print('Accuracy score on validation dataset',acc)
    print('classification_report',classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))
    
    figures.plot_confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1), classes=np.array(class_names), title='Confusion Matrix')
    plt.title(f'confusion matrix for classification of {dataset} dataset')
    plt.savefig(f'{save_path}_confusion_matrix_{dataset}_{batch_size}_kf_{kf}.png')
    plt.clf()     
    
    if dataset == "Mito":
        test_img, test_labels = import_data.mito_test(72)
        preds = model.predict(test_img)
        
        print('Accuracy score on test dataset',accuracy_score(test_labels, preds.argmax(axis=1)))        
        print('classification_report',classification_report(test_labels, preds.argmax(axis=1)))        
        figures.plot_confusion_matrix(np.array(test_labels), preds.argmax(axis=1), classes=np.array(class_names), title='Confusion Matrix') 
        plt.title(f'confusion matrix for classification of {dataset} testing dataset (2 images augmented by 8)')
        plt.savefig(f'{save_path}_confusion_matrix_test_{dataset}_{batch_size}_kf_{kf}.png')
        plt.clf()   
    
    return(acc,epo, tab)


def shuffle_weights(model, weights=None):
    """
    Source https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
    Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)