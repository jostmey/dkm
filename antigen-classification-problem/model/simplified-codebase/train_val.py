#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-03-17
# Purpose: Train and validate classifier for T-cell receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import os
from dataset import *
from model import *
import tensorflow as tf
from tensorflow.keras import *
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
xs_tra_cdr3_train, xs_tra_vgene_train,xs_tra_jgene_train, \
xs_trb_cdr3_train, xs_trb_vgene_train, xs_trb_jgene_train, \
ys_train, fs_train = load_dataset(
  args.database, args.table_train, args.tags, aminoacids_dict,
  tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
  max_steps=max_steps, permute=args.permute
)
xs_tra_cdr3_val, xs_tra_vgene_val, xs_tra_jgene_val, \
xs_trb_cdr3_val, xs_trb_vgene_val, xs_trb_jgene_val, \
ys_val, fs_val = load_dataset(
  args.database, args.table_val, args.tags, aminoacids_dict,
  tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict,
  max_steps=max_steps, permute=args.permute
)

##########################################################################################
# Model
##########################################################################################

num_steps = max_steps

model = generate_model(
  xs_tra_cdr3_train.shape[1:], xs_tra_vgene_train.shape[1:], xs_tra_jgene_train.shape[1:],
  xs_trb_cdr3_train.shape[1:], xs_trb_vgene_train.shape[1:], xs_trb_jgene_train.shape[1:],
  ys_train.shape[1], num_steps
)

##########################################################################################
# Optimization and Compilation
##########################################################################################

learning_rate = 0.001

optimizer = optimizers.Adam(lr=learning_rate)
model.compile(
  optimizer=optimizer,
  loss='categorical_crossentropy',
  metrics=[ 'categorical_accuracy' ]
)

##########################################################################################
# Fit
##########################################################################################

batch_size = ys_train.shape[0]
num_train = ys_train.shape[0]
num_val = ys_val.shape[0]

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

early_stopping = callbacks.EarlyStopping(patience=128, restore_best_weights=True)
model.fit_generator(
  generator=balanced_sampling(
    (
      xs_tra_cdr3_train, xs_tra_vgene_train, xs_tra_jgene_train, 
      xs_trb_cdr3_train, xs_trb_vgene_train, xs_trb_jgene_train
    ),
    ys_train, fs_train, batch_size
  ),
  steps_per_epoch=int(np.ceil(num_train/batch_size)), epochs=1024,
  validation_data=balanced_sampling(
    (
      xs_tra_cdr3_val, xs_tra_vgene_val, xs_tra_jgene_val, 
      xs_trb_cdr3_val, xs_trb_vgene_val, xs_trb_jgene_val
    ),
    ys_val, fs_val, batch_size
  ),
  validation_steps=int(np.ceil(num_val/batch_size)),
  callbacks=[ early_stopping ]
)

##########################################################################################
# Save Results
##########################################################################################

model.save_weights(args.output)

ps_train = model.predict(
  (
    xs_tra_cdr3_train, xs_tra_vgene_train, xs_tra_jgene_train, 
    xs_trb_cdr3_train, xs_trb_vgene_train, xs_trb_jgene_train
  )
)
np.save(args.output+'_ps_train.npy', ps_train)
np.save(args.output+'_ys_train.npy', ys_train)
np.save(args.output+'_ws_train.npy', fs_train)

ps_val = model.predict(
  (
    xs_tra_cdr3_val, xs_tra_vgene_val, xs_tra_jgene_val, 
    xs_trb_cdr3_val, xs_trb_vgene_val, xs_trb_jgene_val
  )
)
np.save(args.output+'_ps_val.npy', ps_val)
np.save(args.output+'_ys_val.npy', ys_val)
np.save(args.output+'_ws_val.npy', fs_val)
