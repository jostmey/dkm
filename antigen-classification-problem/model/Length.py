##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-12-30
# Purpose: Alignment layer for keras
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from alignment_score import *

class Length(Layer):
  def __init__(self, **kwargs):
    super(__class__, self).__init__(**kwargs)
  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return mask
    return K.any(mask, axis=1)
  def call(self, inputs, mask=None):
    lengths = K.sum(K.cast(mask, dtype=inputs.dtype), axis=1, keepdims=True)
    return lengths

