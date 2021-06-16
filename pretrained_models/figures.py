#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:36:43 2021

@author: maelle
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def check_neural_network(model):
    """
    Print names/types of all the layers, and if they are trainable or not
    To check if the Fine Tuning or Transfer Learning is well done
    """
    print(model.summary())
    for layer in model.layers:
        print(layer.name," ", layer.trainable)


############### Plotting ######################
def plot_accuracy(H, save_path, title=""):
    """
    Plots the accuracy according to epoch
    Save the plot

    Parameters
    ----------
    H : TYPE
        History of predictions.
    savepath : str
        Path where save the plot
    title : str, optional
        Plot title. The default is "".

    """
    plt.plot(H.history['accuracy'])
    plt.plot(H.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0,1)
    plt.legend(['train','test'], loc='upper left')
    plt.title("Accuracy of predictions")
    plt.savefig(save_path)
    plt.clf()
    
def plot_loss(H, save_path, title=""):
    """
    Plots the accuracy according to epoch
    Save the plot

    Parameters
    ----------
    H : TYPE
        History of predictions.
    savepath : str
        Path where save the plot
    title : str, optional
        Plot title. The default is "".

    """
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.title("Loss function")
    plt.savefig(save_path)
    plt.clf()
    
    
   


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.title("Confusion matrix")
    return ax


def generate_mosaic(tab, value, x_test, title, save_path):  
    idx = tab.index[tab['Confusion']==value].tolist()  
    print(f"{len(idx)} {title}")
    if len(idx) != 0: 
        if len(idx) > 40:
            title += " (40 first images)"
            idx = idx[:40]
            n_col = 8        
            n_row = len(idx)//n_col
        else:
            n_col = 5      
            n_row = (len(idx)//n_col)+1            
        _, axs = plt.subplots(n_row, n_col)
        plt.suptitle(title)  
        axs = axs.flatten()
        for i, ax in zip(idx, axs): #iterate on index of images and on grid spots
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(x_test[i], cmap='gray')
        fig1 = plt.gcf() #get current figure
        plt.savefig(f"{save_path}_{value}.png")
        fig1.clf()
        
        
    

