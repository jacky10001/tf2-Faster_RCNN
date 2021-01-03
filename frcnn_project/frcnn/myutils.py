# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Sun Jan  3 22:17:11 2021
"""

import numpy as np


# copy tf.keras function from source code
def count_params(weights):
    """Count the total number of scalars composing the weights.
    Args:
        weights: An iterable containing the weights on which to compute params
    Returns:
        The total number of scalars composing the weights
    """
    unique_weights = {id(w): w for w in weights}.values()
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))


def show_params(model):
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)
    
    non_trainable_count = count_params(model.non_trainable_weights)
    
    print('\n----------------------------------------')
    print('    Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('    Trainable params: {:,}'.format(trainable_count))
    print('    Non-trainable params: {:,}'.format(non_trainable_count))
    print('----------------------------------------\n')