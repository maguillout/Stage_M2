# Testing of the robustness of deep learning models for classification of various cell image datasets

Thanks to technological advances in microscopy and image analysis, there is a large amount of various biological samples images. Image analysis has many applications in biology and medicine, especially for cell biology. High Content Screening and classification models can be used to get a lot of information from these images, and thus identify cells of interest. 
Semi-supervised classification models have been developed to be integrated into a microscopy system. The aim of this study is to evaluate the robustness of these models by having them classify various cell datasets.

The tests showed that the transfer learning method makes these models generalizable. They are able to adapt easily to classify cell images from various datasets (different colorations, cell origins, biological issue). 


## Project Overview
```
├── features_extraction
├── pretrained_models
│   ├── import_data.py
│   └── retraining_functions.py
└── README.md
```
