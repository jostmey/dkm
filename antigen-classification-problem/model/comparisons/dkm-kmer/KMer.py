##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-07-30
# Purpose: Compute every possible overlapping k-length subsequence from each input
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow as tf

class KMer(Layer):
  def __init__(self, k, **kwargs):
    self.k = k
    super(__class__, self).__init__(**kwargs)
  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return mask
    return tf.concat(
      [
        tf.expand_dims(mask[:,0], axis=1),  # Make sure the first element is not masked 
        mask[:,self.k:]
      ],
      axis=1
    )
  def call(self, inputs, mask=None):
    N = int(inputs.shape[1])
    kmers = []
    for i in range(N-self.k+1):
      kmers.append(
        tf.concat(
          [ inputs[:,i+j] for j in range(self.k)  ],
          axis=1
        )
      )
    outputs = tf.stack(kmers, axis=1)
    return outputs
