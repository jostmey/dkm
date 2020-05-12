#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Test model classifier for T-cell receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import os
from dataset import *
from model import *
import tensorflow as tf
import numpy as np

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--database', help='Path to the database', type=str, required=True)
parser.add_argument('--cohort_test', help='Name of the test cohort', type=str, required=True)
parser.add_argument('--split_test', help='Name of the test samples', type=str, required=True)
parser.add_argument('--input', help='Input basename', type=str, required=True)
parser.add_argument('--index', help='Index of the bestfit model', type=int, required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--path_shuffle_test', help='Path where CSV file will store shuffled data', type=str, default=None)
parser.add_argument('--gpu', help='GPU ID', type=int, default=0)

args = parser.parse_args()

##########################################################################################
# Environment
##########################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##########################################################################################
# Load datasets
##########################################################################################

# Settings
#
max_steps = 32

# Load representation of the features
#
aminoacids_dict = load_aminoacid_embedding_dict('../../aminoacid-representation/atchley_factors_normalized.csv')

# Load the samples
#
xs_test, ys_test, ws_test = load_dataset(
  args.database, args.cohort_test, args.split_test, aminoacids_dict, path_shuffle=args.path_shuffle_test
)

##########################################################################################
# Model
##########################################################################################

# Settings
#
learning_rate = 0.001
filter_size = 8
num_levels = 3
num_fits = 16

first = list(ys_test.keys())[0]

# Inputs
#
features_cdr3_block = tf.placeholder(tf.float32, [None]+list(xs_test[first]['cdr3'].shape[1:]))
features_quantity_block = tf.placeholder(tf.float32, [None])
features_age_block = tf.placeholder(tf.float32)
weight_block = tf.placeholder(tf.float32)
label_block = tf.placeholder(tf.float32)
level_block = tf.placeholder(tf.int32)

# Format inputs
#
features_quantity_block_ = features_quantity_block/tf.reduce_sum(features_quantity_block)
features_age_block_ = tf.reshape(features_age_block, [1])
labels_block = tf.tile(
  tf.reshape(label_block, [1]),
  [ num_fits ]
)
weight_block_ = tf.reshape(weight_block, [1])

# Define the model
#
model = generate_model(list(xs_test[first]['cdr3'].shape[1:]), filter_size)

# Run model
#
logits_block = model(
  [
    features_cdr3_block, features_quantity_block_, features_age_block_,
    weight_block_, level_block
  ]
)

probabilities_block = tf.math.sigmoid(logits_block)

# Metrics
#
error_block = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_block, labels=labels_block)
costs_block = weight_block_*error_block

corrects_block = tf.cast(
  tf.equal(
    tf.round(labels_block),
    tf.round(probabilities_block)
  ),
  logits_block.dtype
)
accuracies_block = weight_block_*corrects_block

# Aggregate metrics
#
costs = tf.get_variable(
  'costs', shape=costs_block.get_shape(),
  initializer=tf.constant_initializer(0.0),
  dtype=costs_block.dtype, trainable=False
)
accuracies = tf.get_variable(
  'accuracies', shape=accuracies_block.get_shape(),
  initializer=tf.constant_initializer(0.0),
  dtype=accuracies_block.dtype, trainable=False
)

accumulate_costs = costs.assign_add(costs_block)
accumulate_accuracies = accuracies.assign_add(accuracies_block)

reset_costs = costs.assign(tf.zeros_like(costs))
reset_accuracies = accuracies.assign(tf.zeros_like(accuracies))

index_bestfit = tf.argmin(costs, axis=0)

# Aggregate gradients
#
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_params_sample = optimizer.compute_gradients(tf.reduce_sum(costs_block), var_list=tf.trainable_variables())

grads = [
  tf.Variable(tf.zeros_like(param.initialized_value()), dtype=param.initialized_value().dtype, trainable=False) \
  for grad, param in grads_params_sample
]

accumulate_gradients = tf.group(*[
  grads[index].assign_add(grad) for index, (grad, param) in enumerate(grads_params_sample)
])
reset_gradients = tf.group(*[
  grad.assign(tf.zeros_like(grad)) for grad in grads
])

apply_gradients = optimizer.apply_gradients([
  (grads[index], param) for index, (grad, param) in enumerate(grads_params_sample)
])

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

##########################################################################################
# Session
##########################################################################################

# Settings
#
cutoff = 131072

# Open session
#
with tf.Session() as session:

  # Initialize variables
  #
  session.run(initializer)

  # Save the parameters
  #
  model.load_weights(args.input)

  # Test the model
  #
  session.run((reset_costs, reset_accuracies))
  for sample in ys_test.keys():
    session.run(
      (accumulate_costs, accumulate_accuracies),
      feed_dict={
        features_cdr3_block: xs_test[sample]['cdr3'][:cutoff],
        features_quantity_block: xs_test[sample]['quantity'][:cutoff],
        features_age_block: xs_test[sample]['age'],
        label_block: ys_test[sample],
        weight_block: ws_test[sample],
        level_block: num_levels
      }
    )
  cs_test, as_test = session.run((costs, accuracies))

  # Print report
  #
  print(
#    np.mean(cs_test)/np.log(2.0),
#    100.0*np.mean(as_test),
    cs_test[args.index]/np.log(2.0),
    100.0*as_test[args.index],
    sep='\t', flush=True
  )

  # Save the predictions on test data
  #
  with open(args.output+'_ps_test.csv', 'w') as stream:
    print('Sample', 'Weight', 'Label', ','.join([ 'Prediction_'+str(i) for i in range(num_fits) ]), sep=',', file=stream)
    for sample, y in ys_test.items():
      ps = session.run(
        probabilities_block,
        feed_dict={
          features_cdr3_block: xs_test[sample]['cdr3'][:cutoff],
          features_quantity_block: xs_test[sample]['quantity'][:cutoff],
          features_age_block: xs_test[sample]['age'],
          weight_block: ws_test[sample],
          level_block: num_levels
        }
      )
      print(sample, ws_test[sample], y, ','.join([ str(ps[i]) for i in range(num_fits) ]), sep=',', file=stream)
