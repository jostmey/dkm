#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-12-27
# Purpose: Classify TCR sequences using TCR-Dist
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import pandas as pd
import numpy as np
from tcrdist.repertoire import TCRrep

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to the CSV file', type=str, required=True)
parser.add_argument('--output', help='Basename for output files', type=str, required=True)
args = parser.parse_args()

##########################################################################################
# Load datasets
##########################################################################################

df = pd.read_csv(args.input)

##########################################################################################
# Run TCR Dist
##########################################################################################

tr = TCRrep(cell_df=df, organism='human', chains=[ 'alpha', 'beta' ], db_file='alphabeta_gammadelta_db.tsv')

df = tr.clone_df
ds = tr.pw_alpha+tr.pw_beta

##########################################################################################
# Splits
##########################################################################################

is_train = []
is_val = []
is_test = []
for i, split in enumerate(df.split):
  if 'train' == split:
    is_train.append(i)
  elif 'validate' == split:
    is_val.append(i)
  elif 'test' == split:
    is_test.append(i)
is_train = np.array(is_train)
is_val = np.array(is_val)
is_test = np.array(is_test)

##########################################################################################
# Balanced classification accuracy
##########################################################################################

ms_val = []
fs_val = []
for i_val in is_val:
  ds_ = ds[i_val,is_train]
  d_ = np.min(ds_)
  is_match = np.argwhere(ds[i_val,:] == d_).flatten()
  # Frequency by epitope
  epitopes = {}
  for i_match in is_match:
    if df.split[i_match] == 'train':
      epitope = df.epitope[i_match]
      if epitope not in epitopes:
        epitopes[epitope] = 0.0
      epitopes[epitope] += df.frequency[i_match]
  # Identify match with greated frequency
  epitope_best = None
  f_best = -1.0
  for epitope, f in epitopes.items():
    if f >= f_best:
      epitope_best = epitope
      f_best = f
  # Check if best match is correct
  if epitope_best == df.epitope[i_val]:
    ms_val.append(1.0)
  else:
    ms_val.append(0.0)
  fs_val.append(df.frequency[i_val])
ms_val = np.array(ms_val)
fs_val = np.array(fs_val)
a_val = np.sum(fs_val*ms_val)

ms_test = []
fs_test = []
for i_test in is_test:
  ds_ = ds[i_test,is_train]
  d_ = np.min(ds_)
  is_match = np.argwhere(ds[i_test,:] == d_).flatten()
  # Frequency by epitope
  epitopes = {}
  for i_match in is_match:
    if df.split[i_match] == 'train':
      epitope = df.epitope[i_match]
      if epitope not in epitopes:
        epitopes[epitope] = 0.0
      epitopes[epitope] += df.frequency[i_match]
  # Identify match with greated frequency
  epitope_best = None
  f_best = -1.0
  for epitope, f in epitopes.items():
    if f >= f_best:
      epitope_best = epitope
      f_best = f
  # Check if best match is correct
  if epitope_best == df.epitope[i_test]:
    ms_test.append(1.0)
  else:
    ms_test.append(0.0)
  fs_test.append(df.frequency[i_test])
ms_test = np.array(ms_test)
fs_test = np.array(fs_test)
a_test = np.sum(fs_test*ms_test)

##########################################################################################
# Save results
##########################################################################################

with open(args.output+'_a_val.txt', 'w') as stream:
  print('Accuracy:', 100.0*a_val, '%', file=stream)
np.save(args.output+'_fs_val.npy', fs_val)
np.save(args.output+'_ms_val.npy', ms_val)

with open(args.output+'_a_test.txt', 'w') as stream:
  print('Accuracy:', 100.0*a_test, '%', file=stream)
np.save(args.output+'_fs_test.npy', fs_test)
np.save(args.output+'_ms_test.npy', ms_test)
