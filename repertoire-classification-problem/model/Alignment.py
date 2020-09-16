##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-12-30
# Purpose: Alignment layer for keras
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from alignment_score import *

class Alignment(Layer):
  def __init__(
      self,
      filters, weight_steps,
      penalties_feature=0.0, penalties_filter=0.0, length_normalize=False,
      kernel_initializer='uniform', bias_initializer='zeros',
      kernel_regularizer=None, bias_regularizer=None,
      kernel_constraint=None, bias_constraint=None,
      **kwargs
    ):
    self.filters = filters
    self.weight_steps = weight_steps
    self.penalties_feature = penalties_feature
    self.penalties_filter = penalties_filter
    self.length_normalize = length_normalize
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    super(__class__, self).__init__(**kwargs)
  def build(self, input_shape):
    self.kernel = self.add_weight(
      name='kernel',
      shape=[self.weight_steps, int(input_shape[2]), self.filters],
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      trainable=True
    )
    self.bias = self.add_weight(
      name='bias',
      shape=[self.filters],
      initializer=self.bias_initializer,
      regularizer=self.bias_regularizer,
      constraint=self.bias_constraint,
      trainable=True
    )
    super(__class__, self).build(input_shape)
  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return mask
    return K.any(mask, axis=1)
  def call(self, inputs, mask=None):
    scores = alignment_score(
      inputs, mask, self.kernel,
      penalties_feature=self.penalties_feature, penalties_weight=self.penalties_filter
    )
    if self.length_normalize:
      lengths_feature = K.sum(K.cast(mask, dtype=inputs.dtype), axis=1, keepdims=True)
      lengths_weight = K.cast(self.weight_steps, inputs.dtype)
      lengths = K.minimum(lengths_feature, lengths_weight)
      scores = scores/K.sqrt(lengths)
    logits = scores+self.bias
    return logits

