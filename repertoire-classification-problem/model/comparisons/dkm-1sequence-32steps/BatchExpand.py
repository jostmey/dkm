##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-12-30
# Purpose: Expand single feature to the shape of the batch
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

class BatchExpand(Layer):
  def __init__(self, **kwargs):
    super(__class__, self).__init__(**kwargs)
  def call(self, inputs, mask=None):
    x, y = inputs
    outputs = x*K.ones_like(y, dtype=x.dtype)
    return outputs

