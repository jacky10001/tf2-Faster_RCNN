# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Thu Dec  3 01:59:52 2020
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50


nn_dict = {
    'backbone': ResNet50(include_top=False, weights='imagenet'),
    'finetune': [
        # 'conv5_block3_1_conv', 'conv5_block3_1_bn',
        # 'conv5_block3_2_conv', 'conv5_block3_2_bn',
        'conv5_block3_3_conv', 'conv5_block3_3_bn',
        'conv5_block3_out',
    ]
}


class ResNet50_App(tf.keras.Model):
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
        return self.nn_base(inputs)