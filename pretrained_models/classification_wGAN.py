
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

import time

# Neural networks
import keras
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

# Import functions from other files
import import_data
import retraining



from tensorflow.keras.utils import to_categorical

######################### Adjusting pretrained model #########################

def fine_tuning(base_model, nb_classes):
    """
    Adjust the model to adapt it to a new dataset with different classes

    """
    model = Sequential()

    for layer in base_model.layers[:-2]:
        model.add(layer)
    
    layer = layers.Dense(nb_classes)
    layer.trainable = True
    model.add(layer)
    
    layer = layers.Activation(activations.softmax, name="activation_5")
    layer.trainable = True
    model.add(layer)
    
    return(model)

def transfer_learning(base_model, nb_classes):
    """    
    Retrain fully connected layers of a pretrained model
    to adapt it to a new dataset
    
    Parameters
    ----------
    base_model : keras model
        The pretrained model
    nb_classes : int
        the new number of classes.

    Returns
    -------
    The model with trainable dense layers.

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
    """    

    Retrain all the layers of a pretrained model
    to adapt it to a new dataset
    
    Parameters
    ----------
    base_model : keras model
        The pretrained model
    nb_classes : int
        the new number of classes.

    Returns
    -------
    The model with trainable dense layers.

    """
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

def cross_validation(x, y, mode, dataset, class_names, batch_size, path_results, nb_classes, img_names):
    """
    Split the dataset in k groups to vary the testing dataset at each fold
    Draw plots
    Store results in txt file (accuracy, number of epochs, time, images names)

    Parameters
    ----------
    x : images
    y : labels
    mode : FT, TL, FR or FR_w
    dataset : name of the dataset
    class_names : list of different labels
    batch_size : size of one batch
    path_results : directory to store plots
    nb_classes : number of clases

    Returns
    -------
    None.

    """
    x_array = np.array(x)  # x and y must be lists for kfold splitting but must be arrays for training
    y_array = to_categorical(y, nb_classes)    
    
    
    t0 = time.time() #starting timer  
    kf = 0   
    
    # prepare cross validation
    kfold = StratifiedKFold(n_splits=nkf, shuffle=True) 
    kfold.get_n_splits(x,y)   
    
    #################### Cross Validation #######################################   
    # The dataset is split in k groups (k=5), each group will be used as a testing dataset.
    
    acc_kf = [] 
    epo_kf = []
    
    for train_index, test_index in kfold.split(x, y):
        x_train = x_array[train_index]
        y_train = y_array[train_index]
        
        x_valid = x_array[test_index]
        y_valid = y_array[test_index]            
        
        print("Before data augmentation:")
        print(f"Train: {x_train.shape}")
        print(f"Test: {x_valid.shape}")         
        
        x_train, y_train = import_data.data_augmentation(x_train, y_train)
        x_valid, y_valid = import_data.data_augmentation(x_valid, y_valid)
        
        print("After data augmentation:")
        print(f"Train: {x_train.shape}")
        print(f"Test: {x_valid.shape}")
        
        # Normalization:  for classic GAN, values must be in [-1,1]
        x_train = x_train/255
        x_valid = x_valid/255        
        
        # if img_names:
        #     filenames = np.array(img_names)[test_index]            
        
        # else: 
        #     filenames = None
        
        # Model importation
        base_model=keras.models.load_model(model_path)
        base_model.trainable = False
        base_model.summary()   
            
        if mode == "FR":
            title_name = f"Full retraining for prediction of classes {class_names} for {dataset} with sGAN kfold - {kf}"
            model = full_retraining(base_model, nb_classes)        
            save_path = f'{path_results}/{dataset}_full_retrain'
            
        elif mode == "TL":       
            title_name = f"Transfer Learning for prediction of classes {class_names} for {dataset} with sGAN - kfold {kf}"
            model = transfer_learning(base_model, nb_classes)        
            save_path = f'{path_results}/{dataset}_TL'
            
        else:
            title_name = f"Fine Tuning for prediction of classes {class_names} for {dataset}  with sGAN - kfold {kf}"
            model = fine_tuning(base_model, nb_classes) 
            save_path = f'{path_results}/{dataset}_FT'
            
        print(title_name)    
        
        if random_weights:
            retraining.shuffle_weights(model)       
        
        acc, epo, tab = retraining.fitting(model, x_train, y_train, x_valid, y_valid, batch_size, save_path, title_name, kf, class_names, dataset)
        
        acc_kf.append(acc)
        epo_kf.append(epo)
        
        # tab["names"] = filenames
        tab["kfold"] = kf        
        
        kf+=1
        
    
    if mode == "FR":
        f = open(path_results+"/test_"+dataset+"_FR.txt","a")
        tab.to_csv(f"{path_results}/test_{dataset}_FR_table_kf_{kf}.csv",sep=';')
        if os.path.getsize(path_results+"/test_"+dataset+"_FR.txt") == 0:
            f.write("Full Retraining on dataset "+dataset+"\n")            
            f.write("batch_size\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_4\tacc_mean\tepo_mean\ttime\n")   
            
    elif mode == "TL":   
        f = open(path_results+"/test_"+dataset+"_TL.txt","a")
        tab.to_csv(f"{path_results}/test_{dataset}_TL_table_kf_{kf}.csv",sep=';')
        if os.path.getsize(path_results+"/test_"+dataset+"_TL.txt") == 0:
            f.write("Transfer Learning on dataset "+dataset+"\n")            
            f.write("batch_size\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_4\tacc_mean\tepo_mean\ttime\n")   
            
    else:    
        f = open(path_results+"/test_"+dataset+"_FT.txt","a")   
        tab.to_csv(f"{path_results}/test_{dataset}_FT_table_kf_{kf}.csv",sep=';')
        if os.path.getsize(path_results+"/test_"+dataset+"_FT.txt") == 0:
            f.write("Fine Tuning on dataset "+dataset+"\n")
            f.write("batch_size\tacc_0\tepo_0\tacc_1\tepo_1\tacc_2\tepo_2\tacc_3\tepo_3\tacc_4\tepo_4\tacc_mean\tepo_mean\ttime\n")   
            
    f.write(f"{batch_size}\t")
    for i in range(5): #print accuracy and number of epochs for each fold
        f.write(f"{acc_kf[i]}\t")
        f.write(f"{epo_kf[i]}\t")  
    f.write(f"{round(np.mean(acc_kf),2)}(\u00B1{round(np.std(acc_kf),2)})\t")
    f.write(f"{round(np.mean(epo_kf),2)}(\u00B1{round(np.std(epo_kf),2)})\t")

    t1 = time.time() 
    f.write(f"{t1-t0}\n")
    f.close() 


############################ Usage ###########################################

def usage():
    """Print usage of the program."""
    print(f"\nUSAGE\npython {sys.argv[0]}")
    print("Use a pretrained wGAN  model on a new dataset")
    print("\nOPTIONS")
    print("\t --data : dataset: HeLa_Hoechst-EB1, RPE1_Hoechst, HeLa_Hoechst-GM130, NIH3T3_Cilia, Nagao (for classification on the 4 subsets), DIC, CellCognition, Mito")
    print("\t -r --path_results : path directory where plots are registred")    
    print("\t -d --dim : image dimension (in pixels) is d*d")
    print("\t -b --batch_size : size of each batch")
    print("\t -c --cross_valid : optionnal parameter to enable cross validation")
    print("\t -p --per_train_data : optionnal parameter to enable cross validation")
    print("\t -m --mode : retraining mode: FT, TL, FR")
    print("\t -w --random_weights : initialize the model with random weights")
    print("\t --chan: selected color red -> 0 green -> 1 blue -> 2, by default, the thre channels are merged ")
    print("\t --nkf: number of folds for cross validation (default: 5)")
          
    
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
    random_weights = False
    chan = None
    nkf = 5
    
    try:
        opts, _ = getopt.getopt(sys.argv[1:],"m:r:d:b:p:chw",
                                ["mode=", "path_results=", "dim=", "batch_size=",
                                 "data=", "per=", "cross_valid","help","random_weights","chan","nkf="])

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
        elif option in ("-w", "--random_weights"):
            random_weights = True
        elif option in ("--chan"):
            chan = arg
        elif option in ("--,nkf"):
            nkf = arg
        
                
    if data == "Nagao": #for nagao images, there are 4 different datasets
        dataset_list =  ["HeLa_Hoechst-EB1", "RPE1_Hoechst", "HeLa_Hoechst-GM130","NIH3T3_Cilia"] 
    
    else:        
        dataset_list = [data] 
        
        
    # Model importation
    base_model=keras.models.load_model(model_path)
    base_model.trainable = False
    base_model.summary()   
        
    
    for dataset in dataset_list:        
        img_names = None
        
        if dataset == "NIH3T3_Cilia":
            path = "/home/maelle/Documents/Stage_m2/data/"+dataset
            x, y = import_data.nagao(path, dim)
            class_names = ["Cilia", "notCilia"]
            
        elif dataset in ["HeLa_Hoechst-EB1", "RPE1_Hoechst", "HeLa_Hoechst-GM130"]:      
            path = "/home/maelle/Documents/Stage_m2/data/"+dataset
            x, y = import_data.nagao(path, dim)
            class_names = ["G2", "notG2"]
            
        elif data == "CellCognition":
            class_names = ['AA', 'BA','I','J']
            x, y, img_names  = import_data.cell_cognition(dim)
    
        elif data == "DIC":
            class_names = ["AA","NEBD","Meta"]
            x, y, img_names  = import_data.dic(dim)
            
        elif data == "Mito":
            class_names = ["control", "treated"]
            x, y, img_names  = import_data.mito(dim)

            
        else:
            print("Enter a right dataset name (Nagao, DIC or CellCognition)")
            usage()
            sys.exit(0)
            
        ############## Fitting and results ##############################################  
        nb_classes = len(class_names)
        
        if cross_valid:
            cross_validation(x, y, mode, dataset, class_names, batch_size, path_results, nb_classes, img_names)
            
        else:
            t0 = time.time() #starting timer    
            x_train, x_valid, y_train, y_valid = train_valid_split(x, y, test_size = 1-per_train_data)
            train_len = len(y_train)    
            
            # Model importation
            base_model=keras.models.load_model(model_path)
            base_model.trainable = False
            base_model.summary()   

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
                    
            if random_weights:
                retraining.shuffle_weights(model)                

            acc, epo, _ = retraining.fitting(model, x_train, y_train, x_valid, y_valid, batch_size, save_path, title_name, '_', class_names, dataset)

            f.write(f"{train_len}\t")
            f.write(f"{acc}\t")
            f.write(f"{epo}\t")
            t1 = time.time() 
            f.write(f"{t1-t0}\n")
        

            
            