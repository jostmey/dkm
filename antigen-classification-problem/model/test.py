#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Test classifier for T-cell receptor sequences
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
parser.add_argument('--table', help='Path to the test table', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--cutoff', help='Cutoff threshold for including row', type=float, required=True)
parser.add_argument('--input', help='Input basename', type=str, required=True)
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
aminoacids_dict = load_aminoacid_embedding_dict('../../aminoacid-representation/atchley_factors_normalized.csv')
tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict = load_genes_embedding_dict(args.database)

# Load the samples
#
features_tra_cdr3_test, features_tra_vgene_test, features_tra_jgene_test, \
features_trb_cdr3_test, features_trb_vgene_test, features_trb_jgene_test, \
labels_test, weights_test = \
  load_dataset(
    args.database, args.table, args.tags, aminoacids_dict,
    tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
    max_steps=max_steps, permute=args.permute
  )

#ys = labels_test
#ys_ = np.where(ys > 0, ys, np.ones_like(ys))
#hs = np.sum(ys*np.log(ys_), axis=1)
#h = np.mean(hs, axis=0)
#print(h)
#exit()

##########################################################################################
# Model
##########################################################################################

# Define the model
#
model = generate_model(
  features_tra_cdr3_test.shape[1:], features_tra_vgene_test.shape[1:], features_tra_jgene_test.shape[1:],
  features_trb_cdr3_test.shape[1:], features_trb_vgene_test.shape[1:], features_trb_jgene_test.shape[1:],
  labels_test.shape[1]
)

# Run model on test data
#
logits_test = model(
  [
    features_tra_cdr3_test, features_tra_vgene_test, features_tra_jgene_test,
    features_trb_cdr3_test, features_trb_vgene_test, features_trb_jgene_test,
    weights_test
  ]
)
probabilities_test = tf.math.softmax(logits_test)

# Metrics for the test data
#
cost_test = crossentropy(labels_test, logits_test, weights_test)
accuracy_test = accuracy(labels_test, logits_test, weights_test)
threshold_test = tf.placeholder(dtype=logits_test.dtype)
cost_threshold_test = crossentropy_with_threshold(labels_test, logits_test, weights_test, threshold_test)
accuracy_threshold_test = accuracy_with_threshold(labels_test, logits_test, weights_test, threshold_test)
fraction_threshold_test = fraction_with_threshold(logits_test, weights_test, threshold_test)

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

  # Load the model
  #
  model.load_weights(args.input)

  # Test the model
  #
  c_test, a_test, \
  t_test, c_t_test, a_t_test, f_t_test = \
    session.run(
      (
        cost_test, accuracy_test,
        threshold_test, cost_threshold_test, accuracy_threshold_test, fraction_threshold_test
      ),
      feed_dict={
        threshold_test: args.cutoff
      }
    )

  # Print report
  #
  print(
    '%4.3f'%(c_test/np.log(2.0)),
    '%4.3f'%(100.0*a_test),
#    '%4.3f'%(c_t_test/np.log(2.0)),
#    '%4.3f'%(100.0*a_t_test),
#    '%4.3f'%(100.0*f_t_test),
    sep='\t', flush=True
  )

  # Save the model's predictions
  #
  ps_test = session.run(probabilities_test)
  np.save(args.output+'_ps_test.npy', ps_test)
  np.save(args.output+'_ys_test.npy', labels_test)
  np.save(args.output+'_ws_test.npy', weights_test)

