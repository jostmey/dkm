##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-05-04
# Purpose: Copy masks
##########################################################################################

from tensorflow.keras.layers import *

class MaskCopy(Layer):
  def __init__(self, trim_front=0, **kwargs):
    self.trim_front = trim_front
    super(__class__, self).__init__(**kwargs)
  def compute_mask(self, inputs, mask=None):
    return mask[1][:,self.trim_front:]
  def call(self, inputs, mask=None):
    return inputs[0]

