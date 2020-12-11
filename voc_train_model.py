# -*- coding: utf-8 -*-
"""

已完成工作
* 移除 mask 功能
* 替換原本輸入 mask  改讀取 bbox
* 原本 Shapes 程式  可以完整執行

程式測試紀錄
* 訓練 all 失敗  
    1. 錯誤訊息
    TypeError: in converted code:
    
        D:\YJ\MyRepo\Faster_RCNN-tf2\frcnn\model.py:766 call  *
            window = norm_boxes_graph(m['window'], image_shape[:2])
    
        TypeError: tf__norm_boxes_graph() missing 1 required positional argument: 'num_rows'
    2. GPU不足的情形發生= = (GTX1060 6GB)

未來工作
* 比較 voc 與 coco 差異，嘗試使用原來大小訓練
* 改變 backbone 移除 fpn 減少 head 權重數量

@author: Jacky Gao
@date: Fri Dec 11 21:47:40 2020
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from frcnn import utils
from frcnn import model as modellib
from frcnn import visualize
from frcnn.model import log

from voc import VocConfig
from voc import VocDataset

LOG_ROOT = 'log_voc'
MODEL_DIR = os.path.join(LOG_ROOT,'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Local path to trained weights file
COCO_MODEL_PATH =\
    os.path.join('pretrained_model','mask_rcnn_coco.h5')


#%%
config = VocConfig()
config.display()


#%%
def get_ax(rows=1, cols=1, size=6):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
     
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


#%% Dataset
dataset_dir = r'D:\YJ\MyDatasets\VOC\bccd'

# Training dataset
dataset_train = VocDataset()  # TODO: Check data problem
dataset_train.load_voc(dataset_dir, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = VocDataset()
dataset_val.load_voc(dataset_dir, "val")
dataset_val.prepare()


#%% Load and display random samples
  # TODO: Check data problem
image_ids = np.random.choice(dataset_train.image_ids, 1)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    bbox, class_ids = dataset_train.load_bbox(image_id)
    visualize.display_image(image)
    visualize.display_instances(image, bbox, class_ids, dataset_train.class_names, ax=get_ax())


#%% Create Model
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
tf.keras.utils.plot_model(model.keras_model,
                          to_file=os.path.join(LOG_ROOT,'archi_training.png'),
                          show_shapes=True)


#%% Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox","mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


#%% Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers
#    and training only the randomly initialized layers
#    (i.e. the ones that we didn't use pre-trained weights from MS COCO).
#    To train only the head layers, pass `layers='heads'` to the `train()` function.
# 2. Fine-tune all layers. For this simple example it's not necessary,
#    but we're including it to show the process. 
#    Simply pass `layers="all` to train all layers.
# 
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')


#%% Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10, 
            layers="all")


#%% Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually

# model_path = os.path.join(MODEL_DIR, "faster_rcnn.h5")
# model.keras_model.save_weights(model_path)


#%% Detection
class InferenceConfig(VocConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
tf.keras.utils.plot_model(model.keras_model,
                          to_file=os.path.join(LOG_ROOT,'archi_inference.png'),
                          show_shapes=True)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


#%% Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox =\
    modellib.load_image_gt(dataset_val, inference_config, image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)

visualize.display_instances(original_image, gt_bbox, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


#%%
# TODO: Check data problem, predict bbox is error
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
        modellib.load_image_gt(dataset_val, inference_config, image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))

