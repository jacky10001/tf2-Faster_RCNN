# -*- coding: utf-8 -*-
""" Using BCCD dataset for implementation of Faster RCNN
Classes: 3 (RBC WBC Platelets)
Total of image: 365
Image size: 640 x 480 (Width x Height)
Image type: jpeg (JPEG)
Annotation: VOC format

@author: Jacky Gao
@date: Fri Dec 11 21:47:40 2020
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from frcnnfpn import data, utils, visualize
from frcnnfpn.core import common
from frcnnfpn.model import log
from frcnnfpn.model import FasterRCNN

from frcnnfpn.samples.bccd import BccdConfig
from frcnnfpn.samples.bccd import BccdDataset

def get_ax(rows=1, cols=1, size=5):
    return plt.subplots(rows, cols, figsize=(size*cols, size*rows))[1]

LOG_ROOT = 'log_bccd'
MODEL_DIR = os.path.join(LOG_ROOT,'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('pretrained_model','mask_rcnn_coco.h5')


#%%    
config = BccdConfig()
config.display()


#%% Dataset
dataset_dir = r'D:\YJ\MyDatasets\VOC\bccd'

# Training dataset
dataset_train = BccdDataset()
dataset_train.load_voc(dataset_dir, "trainval")
dataset_train.prepare()

# Validation dataset
dataset_val = BccdDataset()
dataset_val.load_voc(dataset_dir, "trainval")
dataset_val.prepare()


#%% Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 1)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    bbox, class_ids = dataset_train.load_bbox(image_id)
    # visualize.display_image(image)
    visualize.display_instances(image, bbox, class_ids, dataset_train.class_names, ax=get_ax())


#%% Create Model
# Create model in training mode
model = FasterRCNN(mode="training", config=config, model_dir=MODEL_DIR)
tf.keras.utils.plot_model(model.keras_model,
                          to_file=os.path.join(LOG_ROOT,'archi_training.png'),
                          show_shapes=True)


#%% Which weights to start with?
init_with = "imagenet"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox","mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)



#%% Train the head branches
# Only the heads.
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=20, 
            layers='heads')

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='all')


#%% Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually

# model_path = os.path.join(MODEL_DIR, "faster_rcnn.h5")
# model.keras_model.save_weights(model_path)


#%% Detection
class InferenceConfig(BccdConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = FasterRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
tf.keras.utils.plot_model(model.keras_model,
                          to_file=os.path.join(LOG_ROOT,'archi_inference.png'),
                          show_shapes=True)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()
# model_path = r"log_bccd\model\bccd20201213T2201/faster_rcnn_best.h5"

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


#%% Test on a random image
ds = dataset_train
# ds = dataset_val

image_id = random.choice(ds.image_ids)

original_image, image_meta, gt_class_id, gt_bbox =\
    data.load_image_gt(ds, inference_config, image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)

visualize.display_instances(original_image, gt_bbox, gt_class_id, 
                            dataset_train.class_names, ax=get_ax())

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['class_ids'], 
                            ds.class_names, r['scores'], ax=get_ax())


#%% Evaluation
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.

ds = dataset_train
# ds = dataset_val

# image_ids = np.random.choice(ds.image_ids, 10)
image_ids = ds.image_ids
print("Total: ", len(image_ids))

APs = []
t1 = time.time()
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox =\
        data.load_image_gt(ds, inference_config, image_id)
    molded_images = np.expand_dims(common.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
t2 = time.time()
 
print("mAP:", np.mean(APs))
print('Time: %6.2fs'%(t2-t1)) 

