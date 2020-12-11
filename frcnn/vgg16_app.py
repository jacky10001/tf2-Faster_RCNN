# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Tue Dec  1 03:10:03 2020
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16


nn_dict = {
    'backbone': VGG16(include_top=False, weights='imagenet'),
    'finetune': [
        'block5_conv1',
        'block5_conv2',
        'block5_conv3',
    ]
}


class VGG16_App(tf.keras.Model):
    def __init__(self, training=False, **kwargs):
        super().__init__(**kwargs)
        self.nn_base = nn_dict['backbone']
        self.nn_tune = nn_dict['finetune']
        
        if training:
            print('\n\nFinetune backbone!!!')
            for layer in self.nn_base.layers:
                if layer.name in self.nn_tune:
                    print('    layer trainable,', layer.name)
                    layer.trainable = training
                else: layer.trainable = False
        else:
            print('\n\nFreeze backbone!!!')
            self.nn_base.trainable = False
        
        print('Backbone network finished initialization.\n\n')
        
    def call(self, inputs):
        x = inputs
        for layer in self.nn_base.layers[:-1]:
            x = layer(x, training=False)
        return x