##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-15
# Purpose: Weighted normalization layer over all samples (without a running average like batch normalization)
##########################################################################################

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K
import tensorflow as tf

class NormalizeInitializationByAggregation(Layer):   # Scale at initialization to zero mean and unit variance, assuming we can only fit one immune repetoire a time on the GPU
  def __init__(self, level, epsilon=1.0E-5, **kwargs):
    self.level = level
    self.epsilon = epsilon
    super(__class__, self).__init__(**kwargs)
  def build(self, input_shape):
    input_shape, _, _ = input_shape
    self.numerator = self.add_weight(
      name='mean',
      shape=input_shape[1:],
      initializer=Zeros(),
      trainable=False
    )
    self.numerator_sq = self.add_weight(
      name='numerator_sq',
      shape=input_shape[1:],
      initializer=Zeros(),
      trainable=False
    )
    self.denominator = self.add_weight(
      name='denominator',
      shape=[1],
      initializer=Constant(1.0E-5),
      trainable=False
    )
    super(__class__, self).build(input_shape)
  def compute_mask(self, inputs, mask=None):
    return None
  def call(self, inputs):
    inputs, weights, level_ = inputs
    level = tf.reshape(tf.cast(self.level, level_.dtype), [1])

    weights_expand = tf.expand_dims(weights, axis=1)

    numerator_block = tf.reduce_sum(weights_expand*inputs, axis=0)
    numerator_sq_block = tf.reduce_sum(weights_expand*inputs**2, axis=0)
    denominator_block = tf.reduce_sum(weights_expand, axis=0)

    indicator = tf.cast(tf.equal(level, level_), numerator_block.dtype)
    numerator = K.update_add(self.numerator, indicator*numerator_block)
    numerator_sq = K.update_add(self.numerator_sq, indicator*numerator_sq_block)
    denominator = K.update_add(self.denominator, indicator*denominator_block)

    mean = numerator/denominator
    variance = numerator_sq/denominator-mean**2

    mean_expand = tf.expand_dims(mean, axis=0)
    variance_expand = tf.expand_dims(variance, axis=0)

    outputs = (inputs-mean_expand)/tf.sqrt(variance_expand+self.epsilon)

    return outputs

