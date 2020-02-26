##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-15
# Purpose: Weighted normalization layer over all samples (without a running average like batch normalization)
##########################################################################################

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K
import tensorflow as tf

class NormalizeInitialization(Layer):  # Scale at initialization to zero mean and unit variance
  def __init__(self, epsilon=1.0E-5, **kwargs):
    self.epsilon = epsilon
    super(__class__, self).__init__(**kwargs)
  def build(self, input_shape):
    input_shape, _ = input_shape
    self.counter = self.add_weight(
      name='counter',
      shape=[1],
      initializer=Zeros(),
      trainable=False
    )
    self.mean = self.add_weight(
      name='mean',
      shape=input_shape[1:],
      initializer=Zeros(),
      trainable=False
    )
    self.variance = self.add_weight(
      name='variance',
      shape=input_shape[1:],
      initializer=Ones(),
      trainable=False
    )
    super(__class__, self).build(input_shape)
  def compute_mask(self, inputs, mask=None):
    return None
  def call(self, inputs):
    inputs, weights = inputs

    weights = weights/tf.reduce_sum(weights)
    weights_expand = tf.expand_dims(weights, axis=1)

    mean, variance = tf.nn.weighted_moments(inputs, [0], weights_expand)

    counter = K.update_add(self.counter, K.ones_like(self.counter))
    init = K.sign(counter-K.ones_like(counter))

    mean = K.update(self.mean, init*self.mean+(1.0-init)*mean)
    variance = K.update(self.variance, init*self.variance+(1.0-init)*variance)

    mean_expand = tf.expand_dims(mean, axis=0)
    variance_expand = tf.expand_dims(variance, axis=0)

    outputs = (inputs-mean_expand)/tf.sqrt(variance_expand+self.epsilon)

    return outputs

