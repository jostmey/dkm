##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-12-31 (the date when I had the idea)
# Purpose: Align features to weights
##########################################################################################

import tensorflow as tf

def alignment_score(features, masks, weights, penalties_feature=0.0, penalties_weight=0.0):

  # Settings
  #
  M = int(features.get_shape()[1])
  N = int(weights.get_shape()[0])
  shape = [ tf.shape(features)[0] ]+list(weights.get_shape()[2:])
  dtype = features.dtype

  # Initialize the alignment table
  #
  scores = []
  for i in range(0, M+1):
    scores.append([])
    for j in range(0, N+1):
      scores[i].append(
        tf.zeros(shape, dtype=dtype)
      )

  # Tabulate alignment scores along each edge
  #
  for i in range(1, M+1):
    scores[i][0] = tf.where(
      masks[:,i-1],
      scores[i-1][0]+penalties_feature,
      scores[i-1][0]
    )
  for j in range(1, N+1):
    scores[0][j] = scores[0][j-1]+penalties_weight

  # Tabulate alignment scores everywhere else
  #
  for i in range(1, M+1):
    for j in range(1, N+1):
      similarities = tf.tensordot(features[:,i-1], weights[j-1], [[1], [0]])
      scores_update = tf.reduce_max(
        tf.stack(
          [
            scores[i-1][j-1]+similarities,
            scores[i  ][j-1]+penalties_weight,
            scores[i-1][j  ]+penalties_feature
          ],
          axis=1
        ),
        axis=1
      )
      scores[i][j] = tf.where(
        masks[:,i-1],  # Indicates if the position is padding
        scores_update,  # Use the new scores from aggregating the transitions into the cell
        scores[i-1][j]  # Otherwise, skip down whenever the position is padding
      )

  return scores[-1][-1]

