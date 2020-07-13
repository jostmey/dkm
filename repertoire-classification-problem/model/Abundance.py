##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-12-30
# Purpose: Alignment layer for keras
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

class Abundance(Layer):
  def __init__(self, **kwargs):
    super(__class__, self).__init__(**kwargs)
  def compute_mask(self, inputs, mask=None):
    return mask
  def call(self, inputs, mask=None):
    inputs_expand = tf.expand_dims(inputs, axis=1)
    outputs = K.log(inputs_expand)
    return outputs
