# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Thu Dec 31 13:34:43 2020
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from frcnn import data, utils, visualize
from frcnn.core import common
from frcnn.model import log
from frcnn.model import FasterRCNN

from frcnn.dataset.coco import CocoConfig
from frcnn.dataset.coco import CocoDataset
from frcnn.dataset.coco import evaluate_coco

def get_ax(rows=1, cols=1, size=5):
    return plt.subplots(rows, cols, figsize=(size*cols, size*rows))[1]

LOG_ROOT = 'log_frcnn'


#%%
class CocoConfig(CocoConfig):
    NAME = 'coco'
    BACKBONE_NAME = 'resnet50'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = [128,256,512]
    
    CLASSIF_FC_LAYERS_SIZE = 256
    POOL_SIZE = 7
    
    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.0003
    STEPS_PER_EPOCH = 250

config = CocoConfig()
# config.display()


#%% Dataset
dataset_dir = r'D:\YJ\MyDatasets\COCO\coco2014'
year = '2014'
download = True

# Training dataset
dataset_train = CocoDataset()
dataset_train.load_coco(dataset_dir, "train", year=year, auto_download=download)
if year in '2014':
    dataset_train.load_coco(dataset_dir, "valminusminival", year=year, auto_download=download)
dataset_train.prepare()

# Validation dataset
dataset_val = CocoDataset()
val_type = "val" if year in '2017' else "minival"
dataset_val.load_coco(dataset_dir, val_type, year=year, auto_download=download)
dataset_val.prepare()


# Load and display random samples
# class CheckConfig(CocoConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
# check_config = CheckConfig()
    
# image_ids = np.random.choice(dataset_train.image_ids, 1)
# for image_id in image_ids:
#     original_image, image_meta, gt_class_id, gt_bbox =\
#         data.load_image_gt(dataset_train, check_config, image_id)
        
#     visualize.display_instances(original_image, gt_bbox, gt_class_id, 
#                                 dataset_train.class_names, ax=get_ax())


#%% Create Model
# Create model in training mode
model = FasterRCNN(mode="training", config=config, model_dir=LOG_ROOT)
model.plot_model()
model.print_summary()

model_path = model.find_last()
model.load_weights(model_path, by_name=True)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=160, trainable='+5')


#%% Detection
class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = FasterRCNN(mode="inference", config=inference_config, model_dir=LOG_ROOT)
model.plot_model()
model.print_summary()

model_path = model.find_last()
# model_path = os.path.join("log_frcnn", "shapes20201219T1642",
#                           "weights", "faster_rcnn_shapes_0001.h5")

# Load trained weights
model.load_weights(model_path, by_name=True)


#%% Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox =\
    data.load_image_gt(dataset_val, inference_config, image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)

visualize.display_instances(original_image, gt_bbox, gt_class_id, 
                            dataset_train.class_names, ax=get_ax())

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())
AP, precisions, recalls, overlaps =\
    utils.compute_ap(gt_bbox, gt_class_id, r["rois"], r["class_ids"], r["scores"])
print(AP)


#%% Evaluation
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.

limit = 500
coco = dataset_val.load_coco(dataset_dir, val_type, year=year, return_coco=True, auto_download=download)
dataset_val.prepare()
evaluate_coco(model, dataset_val, coco, "bbox", limit=limit)

from tqdm import tqdm

image_ids = np.random.choice(dataset_val.image_ids, 10)

# image_ids = dataset_val.image_ids

print(len(image_ids))
APs = []
for image_id in tqdm(image_ids):
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

