# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Mon Dec 14 01:50:03 2020
"""

import os
import glob
import xml.etree.ElementTree as ET



############################################################
#  Parse VOC data
############################################################

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