# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Thu Dec 10 21:05:32 2020
"""

import os
import time
import numpy as np
from tqdm import  tqdm

from ..data import Dataset
from ..config import Config
from .voc_parser import VOC


############################################################
#  Configurations
############################################################


class VocConfig(Config):
    """ Configuration for training on PASCAL VOC dataset.
    Derives from the base Config class and overrides values specific
    to the VOC dataset.
    """
    # Give the configuration a recognizable name
    NAME = "voc"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Close image resizing when input data to model
    IMAGE_RESIZE_MODE = "square"

    # Learning rate and momentum
    LEARNING_RATE = 0.0005
    
    # Number of Weights of head classifier
    CLASSIF_FC_LAYERS_SIZE = 512

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    NUM_CLASSES = 1 + 20  # BG + VOC 20 classes


############################################################
#  Dataset
############################################################

class VocDataset(Dataset):
    def load_voc(self, dataset_dir, subset, show_num=False):
        """ Load a subset of the VOC dataset.
        dataset_dir: The root directory of the VOC dataset.
        subset: What to load (train, val, minival, valminusminival)
        """
        dataset_dir = os.path.abspath(dataset_dir)
        images_dir = os.path.join(dataset_dir,'JPEGImages')
        
        voc = VOC(dataset_dir)
        image_ids = voc.get_subset(subset)
        
        if show_num:
            voc.subset_count(subset, image_ids)

        # Add classes
        for cLs_name, cls_id in voc.class_map.items():
            self.add_class("voc", cls_id, cLs_name)
        
        time.sleep(0.5)
        # Add images
        for i, image_id in enumerate(tqdm(image_ids, desc="Loading")):
            self.add_image(
                "voc", image_id=image_id,
                path=os.path.join(images_dir, voc.images[image_id]['filename']),
                width=voc.images[image_id]["width"],
                height=voc.images[image_id]["height"],
                bboxes=voc.images[image_id]["bboxes"],
                classes=voc.images[image_id]["classes"],
                classes_id=voc.images[image_id]["classes_id"],
            )
        time.sleep(0.5)
    
    def load_voc_list(self, dataset_dir_list, subset_list, show_num=False):
        """ Load a subset of the VOC dataset.
        dataset_dir: The root directory of the VOC dataset.
        subset: What to load (train, val, minival, valminusminival)
        """
        assert isinstance(dataset_dir_list, list) == True
        assert isinstance(subset_list, list) == True
        
        for dataset_dir, subset in zip(dataset_dir_list, subset_list):
            dataset_dir = os.path.abspath(dataset_dir)
            images_dir = os.path.join(dataset_dir,'JPEGImages')
        
            voc = VOC(dataset_dir)
            image_ids = voc.get_subset(subset)
        
            if show_num:
                voc.subset_count(subset, image_ids)
    
            # Add classes
            for cLs_name, cls_id in voc.class_map.items():
                self.add_class("voc", cls_id, cLs_name)
            
            time.sleep(0.5)
            # Add images
            for i, image_id in enumerate(tqdm(image_ids, desc="Loading")):
                self.add_image(
                    "voc", image_id=image_id,
                    path=os.path.join(images_dir, voc.images[image_id]['filename']),
                    width=voc.images[image_id]["width"],
                    height=voc.images[image_id]["height"],
                    bboxes=voc.images[image_id]["bboxes"],
                    classes=voc.images[image_id]["classes"],
                    classes_id=voc.images[image_id]["classes_id"],
                )
            time.sleep(0.5)

    def load_bbox(self, image_id):
        """ Load annotations for the given image.
        Returns:
        bbox: A 2D float array of counding box [[y1, x1, y2, x2], ...].
        class_ids: a 1D array of class IDs.
        """
        bboxes = self.image_info[image_id]["bboxes"]
        class_ids = self.image_info[image_id]["classes_id"]
        bboxes = np.array(bboxes)
        class_ids = np.array(class_ids)
        return bboxes, class_ids

    def image_reference(self, image_id):  # TODO
        """ Return file path """
        info = self.image_info[image_id]
        if info["source"] == "voc":
            return info["path"]
        else:
            super(self.__class__).image_reference(image_id)