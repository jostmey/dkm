##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-05-04
# Purpose: Global pooling with masking
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow as tf

class GlobalPoolWithMask(Layer):
  def __init__(self, **kwargs):
    super(__class__, self).__init__(**kwargs)
  def compute_mask(self, inputs, mask=None):
    return tf.reduce_any(mask, axis=1)
  def call(self, inputs, mask=None):
    indicators = tf.expand_dims(tf.cast(mask, dtype=inputs.dtype), axis=2)
    penalties = -1.0E16*(1.0-indicators)
    outputs = tf.reduce_max(inputs+penalties, axis=1)
    return outputs

