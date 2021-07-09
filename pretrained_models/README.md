# Pretrained models

This folder contains the scripts which permit to adapt pretrained models on other images.

Adaptation is available on:
- Cell Cognition dataset
- Nagao datasets
- DIC datasets (Differential interference contrast microscopy)

There is one script for sGAN models and one other for wGAN models.
Executing a script make the model classify a dataset and return for each dataset:
- a txt file with accuracy, number of epochs, and computing time
- plots of accuracy and loss functions and confusion matrix
- mosaic of right/wrong classified images

## Default usage
```
python sGAN.py -r directory/to/store/results
```

## Datasets
The `-d` option (or `--data`) allows to change the dataset (default is Nagao)
- Cell Cognition dataset: `-d CellCognition`
- Nagao datasets `-d Nagao`
- DIC dataset (Differential interference contrast microscopy) `-d DIC`

## Retraining mode

The `-m` option (or `--mode`) permits to change the layers which have to be retrained
By default, the model is fine tuned before retraining, but it is also possible to use:
- Fine Tuning `-m FT`
- Transfer Learning `-m TL` 
- Full retraining `-m FR` 

## Other options:
- batch size (default is 8): `-b 16` or `--batch_size 16`
- cross validation (defaut is false): `-c` or `--cross_valid`
- proportion data which be used for training (defaut is 0.8): `-p 0.6` or `--per 0.6`
- initialize the model with random weights (defaut is false): `-w` or `--random_weights`
