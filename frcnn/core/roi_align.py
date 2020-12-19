# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Sun Dec 20 01:31:51 2020
"""

import tensorflow as tf
from tensorflow.keras import layers as KL


class ROIAlign(KL.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(ROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2]
        CH = feature_maps.shape[-1]

        # box index
        N = tf.shape(boxes)[0]
        R = tf.shape(boxes)[1]
        box_indices = tf.expand_dims(tf.range(N), axis=1)
        box_indices = tf.tile(box_indices, [1, R])
        box_indices = tf.reshape(box_indices, [-1])  # [0,0,0,..,1,1,1...]
        # boxes
        boxes = tf.reshape(boxes, [-1, 4])
        
        # Stop gradient propogation to ROI proposals
        boxes = tf.stop_gradient(boxes)
        box_indices = tf.stop_gradient(box_indices)

        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        pooled = tf.image.crop_and_resize(
                feature_maps, boxes, box_indices, self.pool_shape,
                method="bilinear")

        # Re-add the batch dimension
        pooled = tf.reshape(pooled, (N, R, self.pool_shape[0], self.pool_shape[1], CH))
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )