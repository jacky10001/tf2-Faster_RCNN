# Faster RCNN on TF2

This repo only Shapes dataset can sucessfully run....

~~In future, I will try VOC format data.~~  
** Current VOC format data can not train ...**  
Some log I write in bccd_train_model.py  

## Done work  
* Remove mask part  
* Inheritance "Dataset" class, ** add load bbox function **
Modify "data_generator" in frcnn/model.py  

## Tips  
* Input data  
  * Modify config  
    ** IMAGE_RESIZE_MODE = "none" **  
    For closing "resize_image" function  
    
    ** NUM_CLASSES **  
    Set number of classes of custom dataset

## Future work  
* Input voc 2017 data  
* Change backbone   
* Head layer reduce weights  

Related work on my [repo](https://github.com/jacky10001/Mask_RCNN-tf2)