#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Train and validate model classifier for T-cell receptor sequences
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
parser.add_argument('--cohort_train', help='Name of the training cohort', type=str, required=True)
parser.add_argument('--split_train', help='Name of the training samples', type=str, required=True)
parser.add_argument('--cohort_val', help='Name of the validation cohort', type=str, required=True)
parser.add_argument('--split_val', help='Name of the validation samples', type=str, required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--path_shuffle_train', help='Path where CSV file will store shuffled data', type=str, default=None)
parser.add_argument('--path_shuffle_val', help='Path where CSV file will store shuffled data', type=str, default=None)

parser.add_argument('--gpu', help='GPU ID', type=int, default=0)

args = parser.parse_args()

# Settings
#
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

##########################################################################################
# Load datasets
##########################################################################################

# Settings
#
max_steps = 32

# Load representation of the features
#
aminoacids_dict = load_aminoacid_embedding_dict('../lib/atchley_factors_normalized.csv')

# Load the samples
#
xs_train, ys_train, ws_train = load_dataset(
  args.database, args.cohort_train, args.split_train, aminoacids_dict, path_shuffle=args.path_shuffle_train
)
xs_val, ys_val, ws_val = load_dataset(
  args.database, args.cohort_val, args.split_val, aminoacids_dict, path_shuffle=args.path_shuffle_val
)

##########################################################################################
# Model
##########################################################################################

# Settings
#
learning_rate = 0.001
num_steps = 8
num_levels = 3
num_fits = 16

first = list(ys_train.keys())[0]

# Inputs
#
features_cdr3_block = tf.placeholder(tf.float32, [None]+list(xs_train[first]['cdr3'].shape[1:]))
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
model = generate_model([num_steps]+list(xs_train[first]['cdr3'].shape[2:]), num_fits)

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
num_epochs = 1024
cutoff = 131072

# Open session
#
with tf.Session() as session:

  # Initialize variables
  #
  session.run(initializer)

  # Initialize the model
  #
  for level_ in range(0, num_levels):
    for sample in ys_train.keys():
      session.run(
        probabilities_block,
        feed_dict={
          features_cdr3_block: xs_train[sample]['cdr3'][:cutoff],
          features_quantity_block: xs_train[sample]['quantity'][:cutoff],
          features_age_block: xs_train[sample]['age'],
          weight_block: ws_train[sample],
          level_block: level_
        }
      )  

  # Each iteration represents one batch
  #
  for epoch in range(0, num_epochs):

    # Train the model
    #
    session.run((reset_costs, reset_accuracies, reset_gradients))
    for sample in ys_train.keys():
      session.run(
        (accumulate_costs, accumulate_accuracies, accumulate_gradients),
        feed_dict={
          features_cdr3_block: xs_train[sample]['cdr3'][:cutoff],
          features_quantity_block: xs_train[sample]['quantity'][:cutoff],
          features_age_block: xs_train[sample]['age'],
          label_block: ys_train[sample],
          weight_block: ws_train[sample],
          level_block: num_levels
        }
      )
    cs_train, as_train, i_bestfit = session.run((costs, accuracies, index_bestfit))

    # Validate the model
    #
    session.run((reset_costs, reset_accuracies))
    for sample in ys_val.keys():
      session.run(
        (accumulate_costs, accumulate_accuracies),
        feed_dict={
          features_cdr3_block: xs_val[sample]['cdr3'][:cutoff],
          features_quantity_block: xs_val[sample]['quantity'][:cutoff],
          features_age_block: xs_val[sample]['age'],
          label_block: ys_val[sample],
          weight_block: ws_val[sample],
          level_block: num_levels
        }
      )
    cs_val, as_val = session.run((costs, accuracies))

    # Update the parameters
    #
    session.run(apply_gradients)

    # Print report
    #
    print(
      epoch,
      np.mean(cs_train)/np.log(2.0),
      100.0*np.mean(as_train),
      np.mean(cs_val)/np.log(2.0),
      100.0*np.mean(as_val),
      i_bestfit,
      cs_train[i_bestfit]/np.log(2.0),
      100.0*as_train[i_bestfit],
      cs_val[i_bestfit]/np.log(2.0),
      100.0*as_val[i_bestfit],
      sep='\t', flush=True
    )

    # Periodically save results
    #
    if epoch%64 == 0:

      # Save the predictions on training data
      #
      with open(args.output+'_ps_train_'+str(epoch)+'.csv', 'w') as stream:
        print('Sample', 'Weight', 'Label', ','.join([ 'Prediction_'+str(i) for i in range(num_fits) ]), sep=',', file=stream)
        for sample, y in ys_train.items():
          ps = session.run(
            probabilities_block,
            feed_dict={
              features_cdr3_block: xs_train[sample]['cdr3'][:cutoff],
              features_quantity_block: xs_train[sample]['quantity'][:cutoff],
              features_age_block: xs_train[sample]['age'],
              weight_block: ws_train[sample],
              level_block: num_levels
            }
          )
          print(sample, ws_train[sample], y, ','.join([ str(ps[i]) for i in range(num_fits) ]), sep=',', file=stream)

      # Save the predictions on validation data
      #
      with open(args.output+'_ps_val_'+str(epoch)+'.csv', 'w') as stream:
        print('Sample', 'Weight', 'Label', ','.join([ 'Prediction_'+str(i) for i in range(num_fits) ]), sep=',', file=stream)
        for sample, y in ys_val.items():
          ps = session.run(
            probabilities_block,
            feed_dict={
              features_cdr3_block: xs_val[sample]['cdr3'][:cutoff],
              features_quantity_block: xs_val[sample]['quantity'][:cutoff],
              features_age_block: xs_val[sample]['age'],
              weight_block: ws_val[sample],
              level_block: num_levels
            }
          )
          print(sample, ws_val[sample], y, ','.join([ str(ps[i]) for i in range(num_fits) ]), sep=',', file=stream)

      # Save the parameters
      #
      model.save_weights(args.output+'_'+str(epoch))

  # Save the predictions on training data
  #
  with open(args.output+'_ps_train.csv', 'w') as stream:
    print('Sample', 'Weight', 'Label', ','.join([ 'Prediction_'+str(i) for i in range(num_fits) ]), sep=',', file=stream)
    for sample, y in ys_train.items():
      ps = session.run(
        probabilities_block,
        feed_dict={
          features_cdr3_block: xs_train[sample]['cdr3'][:cutoff],
          features_quantity_block: xs_train[sample]['quantity'][:cutoff],
          features_age_block: xs_train[sample]['age'],
          weight_block: ws_train[sample],
          level_block: num_levels
        }
      )
      print(sample, ws_train[sample], y, ','.join([ str(ps[i]) for i in range(num_fits) ]), sep=',', file=stream)

  # Save the predictions on validation data
  #
  with open(args.output+'_ps_val.csv', 'w') as stream:
    print('Sample', 'Weight', 'Label', ','.join([ 'Prediction_'+str(i) for i in range(num_fits) ]), sep=',', file=stream)
    for sample, y in ys_val.items():
      ps = session.run(
        probabilities_block,
        feed_dict={
          features_cdr3_block: xs_val[sample]['cdr3'][:cutoff],
          features_quantity_block: xs_val[sample]['quantity'][:cutoff],
          features_age_block: xs_val[sample]['age'],
          weight_block: ws_val[sample],
          level_block: num_levels
        }
      )
      print(sample, ws_val[sample], y, ','.join([ str(ps[i]) for i in range(num_fits) ]), sep=',', file=stream)

  # Save the parameters
  #
  model.save_weights(args.output)

