##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-14
# Purpose: Flatten tensor for single binary output
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow as tf

class Aggregate(Layer):
  def compute_mask(self, inputs, mask=None):
    return None
  def call(self, inputs, mask=None):
    outputs = tf.reduce_max(inputs, axis=0, keepdims=True)
    return outputs

