# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Thu Dec 31 00:55:20 2020
"""

from tensorflow import keras



BACKBONE = {
    'vgg16': {
        'net': keras.applications.VGG16,
        'featuremap_kernel': 512,
    },
    'resnet50': {
        'net': keras.applications.ResNet50,
        'featuremap_kernel': 2048,
    },
    'resnet101': {
        'net': keras.applications.ResNet101,
        'featuremap_kernel': 2048,
    },
    'mobilenetv2': {
        'net': keras.applications.MobileNetV2,
        'featuremap_kernel': 1280,
    },
}



TRAINABLE = {
    'vgg16': {
        '+all': r".*",
        '+frcnn': r"(frcnn\_.*)",
        '+head': r"(frcnn\_.*)|(rpn\_.*)",
        '+5': "(block5\_.*)|(frcnn\_.*)|(rpn\_.*)",
        '+4': "(block4\_.*)|(block5\_.*)|(frcnn\_.*)|(rpn\_.*)",
    },
    'resnet50': {
        '+all': r".*",
        '+frcnn': r"(frcnn\_.*)",
        '+head': r"(frcnn\_.*)|(rpn\_.*)",
        '+5': r"(conv5\_.*)|(frcnn\_.*)|(rpn\_.*)",
        '+4': r"(conv4\_.*)|(conv5\_.*)|(frcnn\_.*)|(rpn\_.*)",
    },
    'resnet101': {
        '+all': r".*",
        '+frcnn': r"(frcnn\_.*)",
        '+head': r"(frcnn\_.*)|(rpn\_.*)",
        '+5': r"(conv5\_.*)|(frcnn\_.*)|(rpn\_.*)",
        '+4': r"(conv4\_.*)|(conv5\_.*)|(frcnn\_.*)|(rpn\_.*)",
    },
    'mobilenetv2': {
        '+all': r".*",
        '+frcnn': r"(frcnn\_.*)",
        '+head': r"(frcnn\_.*)|(rpn\_.*)",
        '+16': r"(block\_16.*)|(Conv\_)|(out_relu)|(frcnn\_.*)|(rpn\_.*)",
        '+15': r"(block\_15.*)|(block\_16.*)|(Conv\_)|(out_relu)|(frcnn\_.*)|(rpn\_.*)",
    },
}