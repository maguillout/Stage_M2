#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:15:47 2021

@author: maelle
"""

import numpy as np
from tensorflow import keras
from keras.models import Model
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

base_model = keras.models.load_model("/home/maelle/Documents/Stage_m2/Models/d_model.h5")
base_model.summary()

outputs = []
conv_layers = []
for layer in (base_model.layers):
    if "conv" in layer.name:
        conv_layers.append(layer.name)
        outputs.append(layer.output)
    
model = Model(inputs=base_model.inputs, outputs=outputs)

