#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Train and validate classifier for T-cell receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
from dataset import *
from model import *
from metrics import *
import tensorflow as tf
import numpy as np

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--database', help='Path to the database', type=str, required=True)
parser.add_argument('--table_train', help='Path to the training table', type=str, required=True)
parser.add_argument('--table_val', help='Path to the validation table', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--permute', help='Randomly permute the relationship between features and labels', type=bool, default=False)
args = parser.parse_args()

##########################################################################################
# Load datasets
##########################################################################################

# Settings
#
max_steps = 32

# Load representation of the features
#
aminoacids_dict = load_aminoacid_embedding_dict('../lib/atchley_factors_normalized.csv')
tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict = load_genes_embedding_dict(args.database)

# Load the samples
#
features_tra_cdr3_train, features_tra_vgene_train, features_tra_jgene_train, \
features_trb_cdr3_train, features_trb_vgene_train, features_trb_jgene_train, \
labels_train, weights_train = \
  load_dataset(
    args.database, args.table_train, args.tags, aminoacids_dict,
    tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
    max_steps=max_steps, permute=args.permute
  )
features_tra_cdr3_val, features_tra_vgene_val, features_tra_jgene_val, \
features_trb_cdr3_val, features_trb_vgene_val, features_trb_jgene_val, \
labels_val, weights_val = \
  load_dataset(
    args.database, args.table_val, args.tags, aminoacids_dict,
    tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
    max_steps=max_steps, permute=args.permute
  )

##########################################################################################
# Model
##########################################################################################

# Settings
#
learning_rate = 0.001
target_accuracy = 0.9

# Define the model
#
model = generate_model(
  features_tra_cdr3_train.shape[1:], features_tra_vgene_train.shape[1:], features_tra_jgene_train.shape[1:],
  features_trb_cdr3_train.shape[1:], features_trb_vgene_train.shape[1:], features_trb_jgene_train.shape[1:],
  labels_train.shape[1]
)

# Run model on training data
#
logits_train = model(
  [
    features_tra_cdr3_train, features_tra_vgene_train, features_tra_jgene_train,
    features_trb_cdr3_train, features_trb_vgene_train, features_trb_jgene_train,
    weights_train
  ]
)
probabilities_train = tf.math.softmax(logits_train)

# Metrics for the training data
#
cost_train = crossentropy(labels_train, logits_train, weights_train)
accuracy_train = accuracy(labels_train, logits_train, weights_train)
threshold_train = find_threshold(labels_train, logits_train, weights_train, target_accuracy)
cost_threshold_train = crossentropy_with_threshold(labels_train, logits_train, weights_train, threshold_train)
accuracy_threshold_train = accuracy_with_threshold(labels_train, logits_train, weights_train, threshold_train)
fraction_threshold_train = fraction_with_threshold(logits_train, weights_train, threshold_train)

# Optimizer
#
index_step = tf.Variable(0, trainable=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_train, global_step=index_step)

# Run the model on validation data
#
logits_val = model(
  [
    features_tra_cdr3_val, features_tra_vgene_val, features_tra_jgene_val,
    features_trb_cdr3_val, features_trb_vgene_val, features_trb_jgene_val,
    weights_val
  ]
)
probabilities_val = tf.math.softmax(logits_val)

# Metrics for the validation data
#
cost_val = crossentropy(labels_val, logits_val, weights_val)
accuracy_val = accuracy(labels_val, logits_val, weights_val)
threshold_val = tf.placeholder(dtype=accuracy_val.dtype)
cost_threshold_val = crossentropy_with_threshold(labels_val, logits_val, weights_val, threshold_val)
accuracy_threshold_val = accuracy_with_threshold(labels_val, logits_val, weights_val, threshold_val)
fraction_threshold_val = fraction_with_threshold(logits_val, weights_val, threshold_val)

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

##########################################################################################
# Session
##########################################################################################

# Settings
#
num_epochs = 1024

# Open session
#
with tf.Session() as session:

  # Initialize variables
  #
  session.run(initializer)

  # Each iteration represents one batch
  #
  for epoch in range(num_epochs):

    # Train the model
    #
    c_train, a_train, \
    t_train, c_t_train, a_t_train, f_t_train, \
    i_step, _ = \
      session.run(
        (
          cost_train, accuracy_train,
          threshold_train, cost_threshold_train, accuracy_threshold_train, fraction_threshold_train,
          index_step, optimizer
        )
      )

    # Evaluate the model
    #
    c_val, a_val, \
    c_t_val, a_t_val, f_t_val = \
      session.run(
        (
          cost_val, accuracy_val,
          cost_threshold_val, accuracy_threshold_val, fraction_threshold_val
        ),
        feed_dict={
          threshold_val: t_train
        }
      )

    # Print report
    #
    print(
      i_step,
      '%4.3f'%(c_train/np.log(2.0)),
      '%4.3f'%(100.0*a_train),
      '%4.3f'%(c_t_train/np.log(2.0)),
      '%4.3f'%(100.0*a_t_train),
      '%4.3f'%(100.0*f_t_train),
      '%4.3f'%(c_val/np.log(2.0)),
      '%4.3f'%(100.0*a_val),
      '%4.3f'%(c_t_val/np.log(2.0)),
      '%4.3f'%(100.0*a_t_val),
      '%4.3f'%(100.0*f_t_val),
      sep='\t', flush=True
    )

  # Save the model and cutoffs
  #
  model.save_weights(args.output)
  with open(args.output+'_cutoff.txt', 'w') as stream:
    print(t_train, file=stream)

  # Save the model's predictions
  #
  ps_train = session.run(probabilities_train)
  np.save(args.output+'_ps_train.npy', ps_train)

  ps_val = session.run(probabilities_val)
  np.save(args.output+'_ps_val.npy', ps_val)

