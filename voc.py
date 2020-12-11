# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Thu Dec 10 21:05:32 2020
"""

import os
import numpy as np

import glob
import xml.etree.ElementTree as ET

# Import Mask RCNN
from frcnn.config import Config
from frcnn.utils import Dataset


class VOC:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self._imgs = {}
        self._total = {}
        self._cls_map = {}
        
        annotations = glob.glob(os.path.join(self.dataset_dir,'Annotations','*.xml'))

        for file in annotations:
            parsedXML = ET.parse(file)
            
            filename = parsedXML.getroot().find('filename').text
            height = int(parsedXML.getroot().find('size/height').text)
            width = int(parsedXML.getroot().find('size/width').text)
            
            obj_cnt = 0
            bndboxes = []
            cls_name = []
            cls_id = []
            for node in parsedXML.getroot().iter('object'):
                name = node.find('name').text
                if name not in self._cls_map:
                    self._cls_map[name] = len(self._cls_map) + 1
                self._total[name] = 1 if name not in self._total else self._total[name] + 1
                    
                xmin = int(node.find('bndbox/xmin').text)
                xmax = int(node.find('bndbox/xmax').text)
                ymin = int(node.find('bndbox/ymin').text)
                ymax = int(node.find('bndbox/ymax').text)
        
                bndboxes.append([ymin, xmin, ymax, xmax])
                cls_name.append(name)
                cls_id.append(self._cls_map.get(name,0))
                
                obj_cnt += 1
            
            img_key, _ = os.path.splitext(filename)
            self._imgs[img_key] = {
                'filename': filename, 'height': height, 'width': width, 
                'bboxes': bndboxes, 'classes': cls_name, 'classes_id': cls_id,
            }
    
    @property
    def images(self):
        return self._imgs
    
    @property
    def class_map(self):
        return self._cls_map
    
    def get_subset(self, subset):
        assert subset in ["test", "train", "trainval", "val"]
        filepath = "{}/ImageSets/Main/{}.txt".format(self.dataset_dir,subset)
        with open(filepath, 'r', encoding='UTF-8') as file:
            image_ids = file.read().splitlines()
        return image_ids
    
    def subset_count(self, subset, image_ids):
        print("\nsubset: {}".format(subset))
        cls_cnt = {}
        for k in image_ids:
            classes = self._imgs[k]['classes']
            for name in classes:
                cls_cnt[name] = 1 if name not in cls_cnt else cls_cnt[name] + 1
        for k, v in cls_cnt.items():
            print("{:>15}: {}".format(k,v))



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

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class VocDataset(Dataset):
    def load_voc(self, dataset_dir, subset):
        """ Load a subset of the VOC dataset.
        dataset_dir: The root directory of the VOC dataset.
        subset: What to load (train, val, minival, valminusminival)
        """
        dataset_dir = os.path.abspath(dataset_dir)
        
        voc = VOC(dataset_dir)
        image_ids = voc.get_subset(subset)
        
        voc.subset_count(subset, image_ids)

        # Add classes
        for cLs_name, cls_id in voc.class_map.items():
            self.add_class("voc", cls_id, cLs_name)

        # Add images
        for i in image_ids:
            self.add_image(
                "voc", image_id=i,
                path=os.path.join(dataset_dir, 'JPEGImages', voc.images[i]['filename']),
                width=voc.images[i]["width"],
                height=voc.images[i]["height"],
                bboxes=voc.images[i]["bboxes"],
                classes=voc.images[i]["classes"],
                classes_id=voc.images[i]["classes_id"],
            )

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