# -*- coding: utf-8 -*-
"""
Mask R-CNN - Inspect Weights of a Trained Model

This code includes code and visualizations to test,
debug, and evaluate the Mask R-CNN model.

@author: Jacky Gao
@date: Thu Dec 10 02:15:47 2020
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from frcnn.model import FasterRCNN
from frcnn import visualize

from frcnn.dataset.voc import VocConfig


# Directory to save logs and trained model
MODEL_DIR = 'log_frcnn'

# Local path to trained weights file
weights_path = os.path.join('log_frcnn', 'hand20201229T1628',
                            'weights', 'faster_rcnn_resnet101_hand_0001.h5')
# weights_path = os.path.join('log_frcnn', 'hand20201229T1628',
#                             'weights', 'faster_rcnn_resnet101_hand_0060.h5')

# Configurations
class VocConfig(VocConfig):
    NAME = 'hand'
    BACKBONE_NAME = 'resnet101'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = [64,128,256]
    
    CLASSIF_FC_LAYERS_SIZE = 256
    POOL_SIZE = 7
    
    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.0001
    STEPS_PER_EPOCH = 200

    NUM_CLASSES = 1 + 1  # BG + Hand 1 classes

config = VocConfig()


#%% Notebook Preferences
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0


#%% 
def get_ax(rows=1, cols=1, size=16):
    return plt.subplots(rows, cols, figsize=(size*cols, size*rows))[1]


#%% Load Model
# Create model in inference mode
with tf.device(DEVICE):
    model = FasterRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
model.load_weights(weights_path, by_name=True)


#%% Review Weight Stats
# Show stats of all trainable weights
# visualize.display_weight_stats(model, html=True)
visualize.display_weight_stats(model, html=False)


#%% Histograms of Weights
# Pick layer types to display
LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']

# Get layers
layers = model.get_trainable_layers()
layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, 
                layers))

# Display Histograms
fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3*len(layers)),
                       gridspec_kw={"hspace":1})
for l, layer in enumerate(layers):
    weights = layer.get_weights()
    for w, weight in enumerate(weights):
        tensor = layer.weights[w]
        ax[l, w].set_title(tensor.name)
        _ = ax[l, w].hist(weight[w].flatten(), 50)
