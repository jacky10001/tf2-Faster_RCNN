# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Fri Dec 18 03:25:13 2020
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from frcnn import data, utils, visualize
from frcnn.core import common
from frcnn.model import log
from frcnn.model import FasterRCNN

from frcnn.samples.shapes import ShapesConfig
from frcnn.samples.shapes import ShapesDataset

def get_ax(rows=1, cols=1, size=5):
    return plt.subplots(rows, cols, figsize=(size*cols, size*rows))[1]

LOG_ROOT = 'log_frcnn_shapes'


#%%
class ShapesConfig(ShapesConfig):
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = [32,64,128]
    
    CLASSIF_FC_LAYERS_SIZE = 256
    POOL_SIZE = 7
    
    IMAGES_PER_GPU = 5
    LEARNING_RATE = 0.0005
    STEPS_PER_EPOCH = 100

config = ShapesConfig()
# config.display()


#%% Dataset
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(1000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 4)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 4)
dataset_val.prepare()

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 1)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     bbox, class_ids = dataset_train.load_bbox(image_id)
#     # visualize.display_image(image)
#     visualize.display_instances(image, bbox, class_ids, dataset_train.class_names, ax=get_ax())


#%% Create Model
# Create model in training mode
model = FasterRCNN(mode="training", config=config, model_dir=LOG_ROOT)
tf.keras.utils.plot_model(model.keras_model, to_file='archi_tra.png', show_shapes=True)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=10)


#%% Detection
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = FasterRCNN(mode="inference", config=inference_config, model_dir=LOG_ROOT)
tf.keras.utils.plot_model(model.keras_model, to_file='archi_inference.png', show_shapes=True)

model_path = model.find_last()
# model_path = os.path.join("log_shapes",
#                           "weights",
#                           "shapes20201219T1642",
#                           "faster_rcnn_shapes_0001.h5")

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


#%% Test on a random image
image_id = random.choice(dataset_train.image_ids)
original_image, image_meta, gt_class_id, gt_bbox =\
    data.load_image_gt(dataset_train, inference_config, image_id)

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


#%% Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox =\
    data.load_image_gt(dataset_val, inference_config, image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)

# visualize.display_instances(original_image, gt_bbox, gt_class_id, 
#                             dataset_train.class_names, ax=get_ax())

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


#%% Evaluation
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.

image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox =\
        data.load_image_gt(dataset_val, inference_config, image_id)
    molded_images = np.expand_dims(common.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))

