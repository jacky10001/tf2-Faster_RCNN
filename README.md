# Faster RCNN on TF2

This repo modify [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
For implementation of Faster RCNN model

**Some error when I input voc 2007 dataset**

## Work list 
[X] Remove mask part  
[X] Training Shapes dataset  
[X] Training BCCD dataset
[ ] Input voc 2017 data  
[ ] Change backbone (Will use Keras application)  
[ ] Head layer reduce weights  

## Tips  
* Input data  
  * Modify config  
    **IMAGE_RESIZE_MODE = "none"**  
    For closing "resize_image" function  
    
    **NUM_CLASSES**  
    Set number of classes of custom dataset
* Install package can follow [here](https://github.com/jacky10001/Faster_RCNN-tf2/blob/main/requirements.txt)  
  I build environment to Python 3.6.6 by Anaconda  

## Refer
BCCD dataset: [here](https://github.com/Shenggan/BCCD_Dataset)  
Related work on my [repo](https://github.com/jacky10001/Mask_RCNN-tf2)  