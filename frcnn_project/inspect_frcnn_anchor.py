# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Thu Dec 17 03:33:59 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from frcnn import utils, visualize

from frcnn.dataset.voc import VocConfig
from frcnn.dataset.voc import VocDataset


def get_ax(rows=1, cols=1, size=6):
    return plt.subplots(rows, cols, figsize=(size*cols, size*rows))[1]


#%% Configurations
class VocConfig(VocConfig):
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
config = VocConfig()
config.display()

dataset_dir = r'D:\YJ\MyDatasets\VOC\voc2007'


#%% Dataset
print('Load dataset')
dataset = VocDataset()
dataset.load_voc(dataset_dir, "trainval")
print('Prepare VOC dataset')
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


#%%
def generate_normal_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at **final feature** . 

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(utils.generate_anchors(scales[i], ratios, feature_shapes,
                                        feature_strides, anchor_stride))
    return np.concatenate(anchors, axis=0)



config.RPN_ANCHOR_SCALES = [128,256,512]
# Generate Anchors
backbone_shapes = config.IMAGE_SHAPE[:2] / config.FEATUREMAP_RATIOS
anchors = generate_normal_anchors(config.RPN_ANCHOR_SCALES, 
                                   config.RPN_ANCHOR_RATIOS,
                                   backbone_shapes,
                                   config.BACKBONE_STRIDES, 
                                   config.RPN_ANCHOR_STRIDE)

# get center anchor
anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
center_cell = backbone_shapes // 2
center_cell_index = (center_cell[0] * backbone_shapes[1] + center_cell[1])
level_center = center_cell_index * anchors_per_cell 
center_anchor = anchors_per_cell * (
        (center_cell[0] * backbone_shapes[1] / config.RPN_ANCHOR_STRIDE**2) \
        + center_cell[1] / config.RPN_ANCHOR_STRIDE)
level_center = int(center_anchor)

colors = visualize.random_colors(3)
# Generate Anchors
fig, ax = plt.subplots(1, figsize=(10, 10))
im = ax.imshow(np.zeros((1024,1024)), cmap='gray')
fig.colorbar(im)
# Draw anchors
for lvl in [-len(anchors)//3 , 0, len(anchors)//3]:
    for i, rect in enumerate(anchors[level_center+lvl:level_center+anchors_per_cell+lvl]):
        y1, x1, y2, x2 = rect
        p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                              edgecolor=np.array(colors[i]))
        ax.add_patch(p)
plt.show()
