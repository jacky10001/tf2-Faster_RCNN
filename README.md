# Faster RCNN on TF2

This repo modify [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
For implementation of Faster RCNN model

~~In future, I will try VOC format data.~~  
~~Current VOC format data can train.~~  
**Current can train BCCD data and Shapes data**  

## Done work  
* Remove mask part  
* Inheritance "Dataset" class, **add load bbox function**
Modify "data_generator" in frcnn/model.py  

## Tips  
* Input data  
  * Modify config  
    **IMAGE_RESIZE_MODE = "none"**  
    For closing "resize_image" function  
    
    **NUM_CLASSES**  
    Set number of classes of custom dataset

## Future work  
* Input voc 2017 data  
* Change backbone   
* Head layer reduce weights  

## Refer
BCCD dataset: [here](https://github.com/Shenggan/BCCD_Dataset)
Related work on my [repo](https://github.com/jacky10001/Mask_RCNN-tf2)