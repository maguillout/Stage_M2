#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:03:12 2021

@author: maelle
"""


############# Packages importation ###############

# Utilities
import os
import getopt
import sys

# Neural networks
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Optimizers
from keras.optimizers import Adam


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

import pandas as pd
import time

from sklearn.model_selection import train_test_split

# Import functions from other files
import import_data
import retraining
import figures


######################### Adjusting pretrained model #########################

def fine_tuning(base_model, nb_classes):
    """
    

    Parameters
    ----------
    base_model : TYPE
        DESCRIPTION.
    nb_classes : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    model = Sequential()

    for layer in base_model.layers[:-2]:
        model.add(layer)
    
    layer = layers.Dense(2)
    layer.trainable = True
    model.add(layer)
    
    layer = layers.Activation(activations.softmax, name="activation_5")
    layer.trainable = True
    model.add(layer)
    
    return(model)

def transfer_learning(base_model, nb_classes):
    """
    

    Parameters
    ----------
    base_model : TYPE
        DESCRIPTION.
    nb_classes : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Creation of a new model based on the imported model
    model = Sequential()

    for layer in base_model.layers[:-1]:
        model.add(layer)

    layer = layers.Dense(units=800, activation='relu')
    layer.trainable = True
    model.add(layer)

    layer = layers.Dense(units=400, activation='relu')
    layer.trainable = True
    model.add(layer)

    layer = layers.Dense(nb_classes)
    layer.trainable = True
    model.add(layer)
    
    layer = layers.Activation(activations.softmax, name="activation_5")
    layer.trainable = True
    model.add(layer)
    
    for layer in model.layers[9:]:
        layer.trainable=True

    return(model)


def full_retraining(base_model, nb_classes):    
    model = Sequential()

    for layer in base_model.layers[:-2]:
        model.add(layer)

    layer = layers.Dense(nb_classes, name="dense_2")
    model.add(layer)

    layer = layers.Activation(activations.softmax, name="activation_5")
    model.add(layer)    
    
    model.trainable = True
    
    return(model)

def full_retraining_weight(base_model, nb_classes):    
    #correct
    model = Sequential()

    for layer in base_model.layers[:-2]:
        model.add(layer)

    layer = layers.Dense(nb_classes, name="dense_2")
    model.add(layer)

    layer = layers.Activation(activations.softmax, name="activation_5")
    model.add(layer)    
    
    model.trainable = True
    
    return(model)


######################### Retraining the model ###############################

def cross_validation(x, y, mode, dataset, class_names, batch_size, path_results, base_model, nb_classes):
    t0 = time.time() #starting timer  
    kf = 0   
    # prepare cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    
    #################### Cross Validation #######################################   
    # The dataset is split in k groups (k=5), each group will be used as a testing dataset.
    
    acc_kf = [] 
    epo_kf = []
    
    for train_index, test_index in kfold.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        
        x_test = x[test_index]
        y_test = y[test_index]
        
        print("cross validation")
        print(dataset)
        print(f'Shape of training image : {x_train.shape}')
        print(f'Shape of testing image : {x_test.shape}')
        print(f'Shape of training labels : {y_train.shape}')
        print(f'Shape of testing labels : {y_test.shape}')   
        
        if mode == "FR":
            title_name = f"Full retraining for prediction of classes {class_names} for {dataset} with wGAN kfold - {kf}"
            model = full_retraining(base_model, nb_classes)        
            save_path = f'{path_results}/{dataset}_full_retrain'
            
        elif mode == "TL":       
            title_name = f"Transfer Learning for prediction of classes {class_names} for {dataset} with wGAN - kfold {kf}"
            model = transfer_learning(base_model, nb_classes)        
            save_path = f'{path_results}/{dataset}_TL'
            
        else:
            title_name = f"Fine Tuning for prediction of classes {class_names} for {dataset}  with wGAN - kfold {kf}"
            model = fine_tuning(base_model, nb_classes) 
            save_path = f'{path_results}/{dataset}_FT'
        
        acc, epo = retraining.fitting(model, x_train, y_train, x_test, y_test, batch_size, save_path, title_name, kf, class_names, dataset)
        
        acc_kf.append(acc)
        epo_kf.append(epo)
        
        kf+=1
        
    
    if mode == "FR":
        f = open(path_results+"/test_"+dataset+"_FR.txt","a")
        if os.path.getsize(path_results+"/test_"+dataset+"_FR.txt") == 0:
            f.write("Full Retraining on dataset "+dataset+"\n")            
            f.write("batch_size\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_4\tacc_mean\tepo_mean\ttime\n")   
            
    elif mode == "TL":   
        f = open(path_results+"/test_"+dataset+"_TL.txt","a")
        if os.path.getsize(path_results+"/test_"+dataset+"_TL.txt") == 0:
            f.write("Transfer Learning on dataset "+dataset+"\n")            
            f.write("batch_size\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_4\tacc_mean\tepo_mean\ttime\n")   
            
    else:    
        f = open(path_results+"/test_"+dataset+"_FT.txt","a")   
        if os.path.getsize(path_results+"/test_"+dataset+"_FT.txt") == 0:
            f.write("Fine Tuning on dataset "+dataset+"\n")
            f.write("batch_size\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_4\tacc_mean\tepo_mean\ttime\n")   
            
    f.write(f"{batch_size}\t")
    for i in range(5): #print accuracy and number of epochs for each fold
        f.write(f"{acc_kf[i]}\t")
        f.write(f"{epo_kf[i]}\t")  
    f.write(f"{round(np.mean(acc_kf),2)}(\u00B1{round(np.std(acc_kf),2)})")
    f.write(f"{np.mean(epo_kf)}\t")

    t1 = time.time() 
    f.write(f"{t1-t0}\n")
    f.close()      


############################ Usage ###########################################

def usage():
    """Print usage of the program."""
    print(f"\nUSAGE\npython {sys.argv[0]}")
    print("Use a pretrained wGAN  model on a new dataset")
    print("\nOPTIONS")
    print("\t --data : dataset: Nagao or DIC or CellCognition")
    print("\t -r --path_results : path directory where plots are registred")    
    print("\t -d --dim : image dimension (in pixels) is d*d")
    print("\t -b --batch_size : size of each batch")
    print("\t -c --cross_valid : optionnal parameter to enable cross validation")
    print("\t -p --per_train_data : optionnal parameter to enable cross validation")
    print("\t -m --mode : retraining mode: FT, TL, FR or FR_w")
    
if __name__ == "__main__":
    # default parameters
    batch_size = 8
    dim = 72
    path_results = None
    mode = "FT"
    cross_valid = False
    per_train_data = 0.8
    data = "Nagao"  
    model_path = "/home/maelle/Documents/Stage_m2/Models/d_model_Trained_All_Labled_Images.h5"

    
    try:
        opts, _ = getopt.getopt(sys.argv[1:],"m:r:d:b:p:c:h",
                                ["mode=", "path_results=", "dim=", "batch_size=",
                                 "data=", "per=", "cross_valid","help"])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)      
        
    for option, arg in opts:        
        
        if option in ("-h", "--help"):
            usage()
            sys.exit(0)
        if option in ("-m", "--mode"):
            mode = arg
        elif option in ("--dataset", "--data"):
            data = arg
        elif option in ("-r", "--path_results"):
            path_results = arg
        elif option in ("-d", "--dim"):
            dim = int(arg)
        elif option in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif option in ("-p", "--per"):
            per_train_data = int(arg)/100
        elif option in ("-c","--cross_valid"):                
            cross_valid = True
        
                
    if data == "Nagao":
        dataset_list =  ["HeLa_Hoechst-EB1", "RPE1_Hoechst", "HeLa_Hoechst-GM130","NIH3T3_Cilia"] 
        for dataset in dataset_list:
            path = "/home/maelle/Documents/Stage_m2/data/"+dataset
            x, y = import_data.nagao(path, dim, wgan=True)
        
    else:
        dataset_list = ["testing dataset"] #for cellcognition and dic there is only one dataset
        
    # Model importation
    base_model=keras.models.load_model(model_path)
    base_model.trainable = False
    base_model.summary()   
        
    
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
            x, y = import_data.cell_cognition(dim, wgan=True)
    
        elif data == "DIC":
            dataset = "DIC"
            class_names = ["AA","NEBD","Meta"]
            x, y = import_data.dic(dim, wgan=True)
            
        else:
            print("Enter a right dataset name (Nagao, DIC or CellCognition)")
            usage()
            sys.exit(0)
            
        ############## Fitting and results ##############################################  
        nb_classes = len(class_names)
        
        if cross_valid:
            cross_validation(x, y, mode, dataset)
            
        else:
            t0 = time.time() #starting timer    
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1-per_train_data)
            train_len = len(y_train)    

            if mode == "FR":
                title_name = f"Full retraining for prediction of classes {class_names} for {dataset} with train_len size {train_len}"
                model = full_retraining(base_model, nb_classes)        
                save_path = f'{path_results}/{dataset}_full_retrain'
            
            elif mode == "TL":       
                title_name = f"Transfer Learning for prediction of classes {class_names} for {dataset} with train_len size {train_len}"
                model = transfer_learning(base_model, nb_classes)        
                save_path = f'{path_results}/{dataset}_TL'
                
            else:
                title_name = f"Fine Tuning for prediction of classes {class_names} for {dataset} with train_len size {train_len}"
                model = fine_tuning(base_model, nb_classes) 
                save_path = f'{path_results}/{dataset}_FT'

            if mode == "FR":
                f = open(path_results+"/test_"+dataset+"_FR.txt","a")
                if os.path.getsize(path_results+"/test_"+dataset+"_FR.txt") == 0:
                    f.write("Full Retraining on dataset "+dataset+"\n")            
                    f.write("train_len\tacc\tepo\ttime\n")   
                    
            elif mode == "TL":   
                f = open(path_results+"/test_"+dataset+"_TL.txt","a")
                if os.path.getsize(path_results+"/test_"+dataset+"_TL.txt") == 0:
                    f.write("Transfer Learning on dataset "+dataset+"\n")            
                    f.write("train_len\tacc\tepo\ttime\n")     
                    
            else:    
                f = open(path_results+"/test_"+dataset+"_FT.txt","a")   
                if os.path.getsize(path_results+"/test_"+dataset+"_FT.txt") == 0:
                    f.write("Fine Tuning on dataset "+dataset+"\n")
                    f.write("train_len\tacc\tepo\ttime\n")   

            acc, epo = retraining.fitting(model, x_train, y_train, x_test, y_test, batch_size, save_path, title_name, '_', class_names, dataset)

            f.write(f"{train_len}\t")
            f.write(f"{acc}\t")
            f.write(f"{epo}\t")
            t1 = time.time() 
            f.write(f"{t1-t0}\n")
        

            
            