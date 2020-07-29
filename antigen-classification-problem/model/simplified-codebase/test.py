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
parser.add_argument('--table', help='Path to the test table', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--input', help='Input basename', type=str, required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--permute', help='Randomly permute the relationship between features and labels', type=bool, default=False)
parser.add_argument('--gpu', help='GPU ID', type=int, default=0)
args = parser.parse_args()

##########################################################################################
# Environment
##########################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
xs_tra_cdr3_test, xs_tra_vgene_test, xs_tra_jgene_test, \
xs_trb_cdr3_test, xs_trb_vgene_test, xs_trb_jgene_test, \
ys_test, fs_test = load_dataset(
  args.database, args.table, args.tags, aminoacids_dict,
  tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
  max_steps=max_steps, permute=args.permute
)

##########################################################################################
# Model
##########################################################################################

model = generate_model(
  xs_tra_cdr3_test.shape[1:], xs_tra_vgene_test.shape[1:], xs_tra_jgene_test.shape[1:],
  xs_trb_cdr3_test.shape[1:], xs_trb_vgene_test.shape[1:], xs_trb_jgene_test.shape[1:],
  ys_test.shape[1]
)

##########################################################################################
# Optimization and Compilation
##########################################################################################

#learning_rate = 0.001

#optimizer = optimizers.Adam(lr=learning_rate)
model.compile(
#  optimizer=optimizer,
  loss='categorical_crossentropy',
  metrics=[ 'categorical_accuracy' ]
)

##########################################################################################
# Load Model
##########################################################################################

model.load_weights(args.input)

##########################################################################################
# Evaluate
##########################################################################################

batch_size = ys_test.shape[0]
num_test = ys_test.shape[0]

def balanced_sampling(xs, ys, ws, batch_size):
  rs = np.arange(xs[0].shape[0])
  ws_ = ws/np.sum(ws)
  while True:
    js = np.random.choice(rs, size=batch_size, p=ws_)  # Balanced sampling from the categories
    yield (
      (
        xs[0][js], xs[1][js], xs[2][js],
        xs[3][js], xs[4][js], xs[5][js]
      ),
      ys[js]
    )

c_test, a_test = model.evaluate_generator(
  generator=balanced_sampling(
    (
      xs_tra_cdr3_test, xs_tra_vgene_test, xs_tra_jgene_test, 
      xs_trb_cdr3_test, xs_trb_vgene_test, xs_trb_jgene_test
    ),
    ys_test, fs_test, batch_size
  ),
  steps=int(np.ceil(num_test/batch_size))
)

print(
  '%4.3f'%(c_test/np.log(2.0)),
  '%4.3f'%(100.0*a_test)
)

##########################################################################################
# Save Results
##########################################################################################

ps_test = model.predict(
  (
    xs_tra_cdr3_test, xs_tra_vgene_test, xs_tra_jgene_test, 
    xs_trb_cdr3_test, xs_trb_vgene_test, xs_trb_jgene_test
  )
)
np.save(args.output+'_ps_test.npy', ps_test)
np.save(args.output+'_ys_test.npy', ys_test)
np.save(args.output+'_ws_test.npy', fs_test)
