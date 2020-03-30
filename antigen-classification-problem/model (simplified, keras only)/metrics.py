##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-15
# Purpose: Custom metrics for training and evaluating a model
##########################################################################################

import tensorflow as tf

def crossentropy(labels, logits, weights):
  weights = weights/tf.reduce_sum(weights)
  costs = -tf.reduce_sum(labels*logits, axis=1)+tf.reduce_logsumexp(logits, axis=1)
  cost = tf.reduce_sum(weights*costs)
  return cost

def accuracy(labels, logits, weights):
  probabilities = tf.math.softmax(logits)
  weights = weights/tf.reduce_sum(weights)
  corrects = tf.cast(
    tf.equal(
      tf.argmax(labels, axis=1),
      tf.argmax(probabilities, axis=1)
    ),
    probabilities.dtype
  )
  accuracy = tf.reduce_sum(weights*corrects)
  return accuracy

def find_threshold(labels, logits, weights, target_accuracy):

  probabilities = tf.math.softmax(logits)
  weights = weights/tf.reduce_sum(weights)

  entropies = -tf.reduce_sum(probabilities*logits, axis=1)+tf.reduce_logsumexp(logits, axis=1)
  corrects = tf.cast(
    tf.equal(
      tf.argmax(labels, axis=1),
      tf.argmax(probabilities, axis=1)
    ),
    probabilities.dtype
  )

  indices_sorted = tf.argsort(entropies, axis=0)
  entropies_sorted = tf.gather(entropies, indices_sorted)
  corrects_sorted = tf.gather(corrects, indices_sorted)
  weights_sorted = tf.gather(weights, indices_sorted)

  numerators_sorted = tf.math.cumsum(weights_sorted*corrects_sorted, axis=0)
  denominators_sorted = tf.math.cumsum(weights_sorted, axis=0)
  accuracies_sorted = numerators_sorted/denominators_sorted

  range = tf.math.cumsum(tf.ones_like(accuracies_sorted, dtype=tf.int64), axis=0)-1  # Subtract one so the range starts as zero
  indices_threshold = tf.where(
    accuracies_sorted > tf.constant(target_accuracy, accuracies_sorted.dtype),
    range,
    tf.zeros_like(range)
  )
  index_threshold = tf.reduce_max(indices_threshold)

  entropy_threshold = tf.gather(entropies_sorted, index_threshold)

  return entropy_threshold

def accuracy_with_threshold(labels, logits, weights, threshold):

  probabilities = tf.math.softmax(logits)
  weights = weights/tf.reduce_sum(weights)

  entropies = -tf.reduce_sum(probabilities*logits, axis=1)+tf.reduce_logsumexp(logits, axis=1)
  corrects = tf.cast(
    tf.equal(
      tf.argmax(labels, axis=1),
      tf.argmax(probabilities, axis=1)
    ),
    probabilities.dtype
  )

  masks = tf.where(
    entropies <= threshold,
    tf.ones_like(entropies),
    tf.zeros_like(entropies)
  )

  accuracy_mask = tf.math.divide(
    tf.reduce_sum(weights*masks*corrects),
    tf.reduce_sum(weights*masks)
  )

  return accuracy_mask

def crossentropy_with_threshold(labels, logits, weights, threshold):

  probabilities = tf.math.softmax(logits)
  weights = weights/tf.reduce_sum(weights)

  entropies = -tf.reduce_sum(probabilities*logits, axis=1)+tf.reduce_logsumexp(logits, axis=1)
  costs = -tf.reduce_sum(labels*logits, axis=1)+tf.reduce_logsumexp(logits, axis=1)

  masks = tf.where(
    entropies <= threshold,
    tf.ones_like(entropies),
    tf.zeros_like(entropies)
  )

  cost_mask = tf.math.divide(
    tf.reduce_sum(weights*masks*costs),
    tf.reduce_sum(weights*masks)
  )

  return cost_mask

def fraction_with_threshold(logits, weights, threshold):

  probabilities = tf.math.softmax(logits)
  weights = weights/tf.reduce_sum(weights)

  entropies = -tf.reduce_sum(probabilities*logits, axis=1)+tf.reduce_logsumexp(logits, axis=1)

  masks = tf.where(
    entropies <= threshold,
    tf.ones_like(entropies),
    tf.zeros_like(entropies)
  )

  fraction_mask = tf.reduce_sum(weights*masks)

  return fraction_mask

