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

from frcnn.dataset.coco import CocoConfig
from frcnn.dataset.coco import CocoDataset
from frcnn.dataset.coco import evaluate_coco
from frcnn.dataset.shapes import ShapesConfig
from frcnn.dataset.shapes import ShapesDataset
from frcnn.dataset.voc import VocConfig
from frcnn.dataset.voc import VocDataset
from frcnn.config import Config


# Directory to save logs and trained model
MODEL_DIR = 'log_frcnn'
PROJECT_DIR = os.path.join(MODEL_DIR, 'voc20210102T1143')


dataset_dir = r'D:\YJ\MyDatasets\VOC\voc2007'
# dataset_dir = r'D:\YJ\MyDatasets\IOPLAB\Jerry_happycells_help\cell_label_data'



weights_path = os.path.join(PROJECT_DIR, #'weights',
                            'faster_rcnn_best_202101021143.h5')



config_dir = os.path.join(PROJECT_DIR,
                          'config_202101021143.json')



config = Config()
config.load(config_dir)
config.GPU_COUNT = 1
config.IMAGES_PER_GPU = 1
config.__init__()
config.display()


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
