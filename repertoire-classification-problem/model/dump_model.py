#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Dump parameters of the statistical classifier
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
parser.add_argument('--cohort', help='Name of the cohort', type=str, required=True)
parser.add_argument('--split', help='Name of the samples', type=str, required=True)
parser.add_argument('--input', help='Input basename', type=str, required=True)
parser.add_argument('--index', help='Index of the fit', type=int, required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)

args = parser.parse_args()

# Settings
#
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
xs, ys, ws = load_dataset(
  args.database, args.cohort, args.split, aminoacids_dict
)

##########################################################################################
# Model
##########################################################################################

# Settings
#
learning_rate = 0.001
num_steps = 8
num_fits = 16

first = list(ys.keys())[0]

# Inputs
#
features_cdr3_block = tf.placeholder(tf.float32, [None]+list(xs[first]['cdr3'].shape[1:]))
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
model = generate_model(list(xs[first]['cdr3'].shape[1:]), num_fits, num_steps)

# Run model
#
logits_block = model(
  [
    features_cdr3_block, features_quantity_block_, features_age_block_,
    weight_block_, level_block
  ]
)

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

###########################################################################################
# Session
##########################################################################################

# Settings
#
eps = 1.0E-5

# Open session
#
with tf.Session() as session:

  # Initialize variables
  #
  session.run(initializer)

  # Load the model
  #
  model.load_weights(args.input)

  # Save the embeddings for the alignment layer
  #
  with open(args.output+'_alignment_embedding.csv', 'w') as stream:
    for aminoacid, values in aminoacids_dict.items():
      print(aminoacid, ','.join([ str(value) for value in values ]), sep=',', file=stream)

  print(model.summary())

  # Get weights for global init normalization
  #
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_7').get_weights()
  ms_global = ns/ds
  vs_global = ns2/ds-ms_global**2

  # Divide by three because thats how many times we will reuse this term
  #
  ms_global /= 4

  # Save the weights of the alignment layer
  #
  ws, bs = model.get_layer('alignment').get_weights()
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation').get_weights()
  ms = ns/ds
  vs = ns2/ds-ms**2

  vs_ = np.expand_dims(np.expand_dims(vs, axis=0), axis=0)
  ws_ = ws/np.sqrt(vs_+eps)
  bs_ = (bs-ms)/np.sqrt(vs+eps)

  vs_global_ = np.expand_dims(np.expand_dims(vs_global, axis=0), axis=0)
  ws_ = ws_/np.sqrt(vs_global_+eps)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global+eps)

  np.save(args.output+'_alignment_weights.npy', ws_[:,:,args.index])
  np.save(args.output+'_alignment_biases.npy', bs_[args.index])

  # Save the weights of the length layer
  #
  ws, bs = model.get_layer('dense').get_weights()
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_1').get_weights()
  ms_feature = ns/ds
  vs_feature = ns2/ds-ms_feature**2
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_2').get_weights()
  ms = ns/ds
  vs = ns2/ds-ms**2

  ms_feature_ = np.expand_dims(ms_feature, axis=0)
  vs_feature_ = np.expand_dims(vs_feature, axis=0)
  vs_ = np.expand_dims(vs, axis=0)
  ws_ = (ws/np.sqrt(vs_feature_+eps))/np.sqrt(vs_+eps)
  bs_ = (bs-np.sum(ws*ms_feature_/np.sqrt(vs_feature_+eps), axis=0)-ms)/np.sqrt(vs+eps)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_+eps)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global+eps)

  np.save(args.output+'_length_weights.npy', ws_[:,args.index])
  np.save(args.output+'_length_biases.npy', bs_[args.index])

  # Save the weights of the abundance layer
  #
  ws, bs = model.get_layer('dense_1').get_weights()
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_3').get_weights()
  ms_feature = ns/ds
  vs_feature = ns2/ds-ms_feature**2
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_4').get_weights()
  ms = ns/ds
  vs = ns2/ds-ms**2

  ms_feature_ = np.expand_dims(ms_feature, axis=0)
  vs_feature_ = np.expand_dims(vs_feature, axis=0)
  vs_ = np.expand_dims(vs, axis=0)
  ws_ = (ws/np.sqrt(vs_feature_+eps))/np.sqrt(vs_+eps)
  bs_ = (bs-np.sum(ws*ms_feature_/np.sqrt(vs_feature_+eps), axis=0)-ms)/np.sqrt(vs+eps)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_+eps)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global+eps)

  np.save(args.output+'_abundance_weights.npy', ws_[:,args.index])
  np.save(args.output+'_abundance_biases.npy', bs_[args.index])

  # Save the weights of the age layer
  #
  ws, bs = model.get_layer('dense_2').get_weights()
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_5').get_weights()
  ms_feature = ns/ds
  vs_feature = ns2/ds-ms_feature**2
  ns, ns2, ds = model.get_layer('normalize_initialization_by_aggregation_6').get_weights()
  ms = ns/ds
  vs = ns2/ds-ms**2

  ms_feature_ = np.expand_dims(ms_feature, axis=0)
  vs_feature_ = np.expand_dims(vs_feature, axis=0)
  vs_ = np.expand_dims(vs, axis=0)
  ws_ = (ws/np.sqrt(vs_feature_+eps))/np.sqrt(vs_+eps)
  bs_ = (bs-np.sum(ws*ms_feature_/np.sqrt(vs_feature_+eps), axis=0)-ms)/np.sqrt(vs+eps)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_+eps)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global+eps)

  np.save(args.output+'_age_weights.npy', ws_[:,args.index])
  np.save(args.output+'_age_biases.npy', bs_[args.index])
