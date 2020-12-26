# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Fri Dec 25 21:50:52 2020
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from frcnn import data, utils, visualize
from frcnn.core import common
from frcnn.model import log
from frcnn.model import FasterRCNN

from frcnn.samples.voc import VocConfig
from frcnn.samples.voc import VocDataset

def get_ax(rows=1, cols=1, size=5):
    return plt.subplots(rows, cols, figsize=(size*cols, size*rows))[1]

LOG_ROOT = 'log_frcnn_hand'


#%% Configurations
class HandConfig(VocConfig):
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = [64,128,512]
    
    CLASSIF_FC_LAYERS_SIZE = 256
    POOL_SIZE = 7
    
    IMAGES_PER_GPU = 5
    LEARNING_RATE = 0.0001
    STEPS_PER_EPOCH = 200

    NUM_CLASSES = 1 + 1  # BG + Hand 1 classes
    
config = HandConfig()
# config.display()


#%% Dataset
dataset_dir1 = r'D:\YJ\MyDatasets\VOC\egohands_data'
dataset_dir2 = r'D:\YJ\MyDatasets\VOC\vgg_hands_data'



# Training dataset
# dataset_train = VocDataset()
# dataset_train.load_voc(dataset_dir1, "trainval")
# dataset_train.prepare()

# dataset_train = VocDataset()
# dataset_train.load_voc(dataset_dir2, "trainval")
# dataset_train.prepare()

dataset_train = VocDataset()
dataset_train.load_voc_list([dataset_dir1,dataset_dir2], ["trainval","trainval"])
dataset_train.prepare()



# Validation dataset
# dataset_val = VocDataset()
# dataset_val.load_voc(dataset_dir1, "test")
# dataset_val.prepare()

# dataset_val = VocDataset()
# dataset_val.load_voc(dataset_dir2, "test")
# dataset_val.prepare()

dataset_val = VocDataset()
dataset_val.load_voc_list([dataset_dir1,dataset_dir2], ["test","test"])
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
            epochs=50)


#%% Detection
class InferenceConfig(HandConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = FasterRCNN(mode="inference", config=inference_config, model_dir=LOG_ROOT)
tf.keras.utils.plot_model(model.keras_model, to_file='archi_det.png', show_shapes=True)


# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

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

t1 = time.time()
results = model.detect([original_image], verbose=1)
t2 = time.time()
print('time:', t2-t1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


#%% Evaluation
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.

# image_ids = np.random.choice(dataset_val.image_ids, 10)

image_ids = dataset_val.image_ids


print(len(image_ids))
APs = []
t1 = time.time()
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
t2 = time.time()
print('time:', t2-t1)
print("\nmAP: ", np.mean(APs))

