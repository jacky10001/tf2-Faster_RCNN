# Faster RCNN on TF2

This repo modify [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
For implementation of Faster RCNN model

## Work list 
- [x] Remove mask part  
- [x] Input Shapes dataset  
- [x] Input BCCD dataset  
- [x] Input voc 2017 data  
- [x] Change backbone (Will use Keras application)  
- [x] Head layer reduce weights  
- [ ] Inspect training results (performance)

## Tips  
* Install package can follow [here](https://github.com/jacky10001/Faster_RCNN-tf2/blob/main/requirements.txt)  
  I build environment to Python 3.6.6 by Anaconda  
* Input data  
  * Modify config  
    **IMAGE_RESIZE_MODE = "none"**  
    For closing "resize_image" function  
    **IMAGE_RESIZE_MODE = "square"**  
    For different size image  
    
    **NUM_CLASSES**  
    Set number of classes of custom dataset  

## Refer
PSACAL VOC 2007 dataset: [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)  
BCCD dataset: [here](https://github.com/Shenggan/BCCD_Dataset)  
Related work on my [repo](https://github.com/jacky10001/Mask_RCNN-tf2)  
