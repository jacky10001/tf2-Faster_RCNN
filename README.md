# Faster RCNN on TF2
This repo modify [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
For implementation of Faster RCNN model  


## Folder
**frcnn_project (main work)**: for change backbone network from Keras Applications model  
**frcnnfpn_project**: use pre-training model from matterport/Mask_RCNN [weights](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)  


## setup
* Anaconda on Windows 10  
* Python 3.6.6  
* Use Spyder IDE edit my code  
* GTX 1060 (Laptop)
* CUDA 10.1 + cudnn v8.0.4.30  
* RAM: 16GB
The code can run on tf2.3  


## Work list 
- [x] Remove mask part  
- [x] Input Shapes dataset  
- [x] Input BCCD dataset  
- [x] Input voc 2007 data  
- [x] Change backbone (use ResNet50 from Keras application)  
- [x] Head layer reduce weights  
- [ ] Change backbone (use MobileNetV2 from Keras application)  
- [ ] Inspect training results (calculate mAP)


## Tips  
* Install package can follow [here](https://github.com/jacky10001/Faster_RCNN-tf2/blob/main/requirements.txt)   
* Input data  
  * Modify config  
    **IMAGE_RESIZE_MODE = "none" or "square"**  
    "square" can handle different size image  
    
    **NUM_CLASSES**  
    Set number of classes of custom dataset  
  * Use VOC annotation format (.xml)


## Results  
Showing prediced results of VOC dataset and BCCD dataset by **frcnn**  
![alt text](https://github.com/jacky10001/Faster_RCNN-tf2/blob/main/images/image-1.png "Train VOC dataset")
![alt text](https://github.com/jacky10001/Faster_RCNN-tf2/blob/main/images/image-2.png "Train BCCD dataset")


## Refer
PSACAL VOC 2007 dataset: [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)  
BCCD dataset: [here](https://github.com/Shenggan/BCCD_Dataset)  
Related work on my [repo](https://github.com/jacky10001/Mask_RCNN-tf2)  
