#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Dump the model's weights
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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--database', help='Path to the database', type=str, required=True)
parser.add_argument('--table', help='Path to the table', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--input', help='Input basename', type=str, required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
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
features_tra_cdr3, features_tra_vgene, features_tra_jgene, \
features_trb_cdr3, features_trb_vgene, features_trb_jgene, \
labels, weights = \
  load_dataset(
    args.database, args.table, args.tags, aminoacids_dict,
    tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
    max_steps=max_steps
  )

##########################################################################################
# Model
##########################################################################################

# Define the model
#
model = generate_model(
  features_tra_cdr3.shape[1:], features_tra_vgene.shape[1:], features_tra_jgene.shape[1:],
  features_trb_cdr3.shape[1:], features_trb_vgene.shape[1:], features_trb_jgene.shape[1:],
  labels.shape[1]
)

# Run model on training data
#
logits = model(
  [
    features_tra_cdr3, features_tra_vgene, features_tra_jgene,
    features_trb_cdr3, features_trb_vgene, features_trb_jgene,
    weights
  ]
)

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

##########################################################################################
# Session
##########################################################################################

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

  # Save the embeddings for the vgene layer
  #
  with open(args.output+'_tra_vgene_embedding.csv', 'w') as stream:
    for vgene, values in tra_vgenes_dict.items():
      print(vgene, ','.join([ str(value) for value in values ]), sep=',', file=stream)

  # Save the embeddings for the jgene layer
  #
  with open(args.output+'_tra_jgene_embedding.csv', 'w') as stream:
    for jgene, values in tra_jgenes_dict.items():
      print(jgene, ','.join([ str(value) for value in values ]), sep=',', file=stream)

  # Save the embeddings for the vgene layer
  #
  with open(args.output+'_trb_vgene_embedding.csv', 'w') as stream:
    for vgene, values in trb_vgenes_dict.items():
      print(vgene, ','.join([ str(value) for value in values ]), sep=',', file=stream)

  # Save the embeddings for the jgene layer
  #
  with open(args.output+'_trb_jgene_embedding.csv', 'w') as stream:
    for jgene, values in trb_jgenes_dict.items():
      print(jgene, ','.join([ str(value) for value in values ]), sep=',', file=stream)


  # Get weights for global init normalization
  #
  _, ms_global, vs_global = model.get_layer('normalize_initialization_8').get_weights()

  # Save the weights of the alignment layer
  #
  ws, bs = model.get_layer('alignment').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization').get_weights()

  vs_ = np.expand_dims(np.expand_dims(vs, axis=0), axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(np.expand_dims(vs_global, axis=0), axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_tra_alignment_weights.npy', ws_)
  np.save(args.output+'_tra_alignment_biases.npy', bs_)

  # Save the weights of the length layer
  #
  ws, bs = model.get_layer('dense').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_1').get_weights()

  vs_ = np.expand_dims(vs, axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_tra_length_weights.npy', ws_)
  np.save(args.output+'_tra_length_biases.npy', bs_)

  # Save the weights of the vgene layer
  #
  ws, bs = model.get_layer('dense_1').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_2').get_weights()

  vs_ = np.expand_dims(vs, axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_tra_vgene_weights.npy', ws_)
  np.save(args.output+'_tra_vgene_biases.npy', bs_)

  # Save the weights of the jgene layer
  #
  ws, bs = model.get_layer('dense_2').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_3').get_weights()

  vs_ = np.expand_dims(vs, axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_tra_jgene_weights.npy', ws_)
  np.save(args.output+'_tra_jgene_biases.npy', bs_)

  # Save the weights of the alignment layer
  #
  ws, bs = model.get_layer('alignment_1').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_4').get_weights()

  vs_ = np.expand_dims(np.expand_dims(vs, axis=0), axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(np.expand_dims(vs_global, axis=0), axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_trb_alignment_weights.npy', ws_)
  np.save(args.output+'_trb_alignment_biases.npy', bs_)

  # Save the weights of the length layer
  #
  ws, bs = model.get_layer('dense_3').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_5').get_weights()

  vs_ = np.expand_dims(vs, axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_trb_length_weights.npy', ws_)
  np.save(args.output+'_trb_length_biases.npy', bs_)

  # Save the weights of the vgene layer
  #
  ws, bs = model.get_layer('dense_4').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_6').get_weights()

  vs_ = np.expand_dims(vs, axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_trb_vgene_weights.npy', ws_)
  np.save(args.output+'_trb_vgene_biases.npy', bs_)

  # Save the weights of the jgene layer
  #
  ws, bs = model.get_layer('dense_5').get_weights()
  _, ms, vs = model.get_layer('normalize_initialization_7').get_weights()

  vs_ = np.expand_dims(vs, axis=0)
  ws_ = ws/np.sqrt(vs_)
  bs_ = (bs-ms)/np.sqrt(vs)

  vs_global_ = np.expand_dims(vs_global, axis=0)
  ws_ = ws_/np.sqrt(vs_global_)
  bs_ = (bs_-ms_global)/np.sqrt(vs_global)

  np.save(args.output+'_trb_jgene_weights.npy', ws_)
  np.save(args.output+'_trb_jgene_biases.npy', bs_)
