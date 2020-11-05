##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-14
# Purpose: Aggregate instances
##########################################################################################

from tensorflow.keras.layers import *
import tensorflow as tf

class Aggregate(Layer):
  def call(self, inputs, mask=None):
    outputs = tf.reduce_max(inputs, axis=0, keepdims=True)
    return outputs

class Aggregate2Instances(Layer):
  def call(self, inputs, mask=None):
    num = int(inputs.shape[1])
    half = int(num/2)

    matches_trans, indices_trans = tf.nn.top_k(
      tf.transpose(inputs), k=2, sorted=True
    )

    matches_trans_0 = matches_trans[:half,:]
    matches_trans_1 = matches_trans[half:,:]

    indices_trans_0 = indices_trans[:half,:]
    indices_trans_1 = indices_trans[half:,:]

    penalties_trans = -1.0E16*tf.cast(
      tf.equal(indices_trans_0[:,0], indices_trans_1[:,0]),
      inputs.dtype
    )  # Penalty is 0 if different instances are used, otherwise, the penalty is -1.0E16

    outputs = tf.reduce_max(
      tf.stack(
        [
          matches_trans_0[:,0]+matches_trans_0[:,0]+penalties_trans,  # Penalty is 0 if different instances are used
          matches_trans_0[:,0]+matches_trans_0[:,1],  # Otherwise, try 0th then 1st values
          matches_trans_0[:,1]+matches_trans_0[:,0],  # Then try the 1st then 0th values
        ],
        axis=0
      ),
      axis=0, keepdims=True
    )

    return outputs
