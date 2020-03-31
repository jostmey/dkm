#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Dump the model's preformance over a range of target cutoffs
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import csv
import numpy as np

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--predictions_val', help='Base path to the files with the prediction results for the validation data', type=str, required=True)
parser.add_argument('--output', help='Path to CSV file where cutoffs will be stored', type=str, required=True)
args = parser.parse_args()

##########################################################################################
# Load data
##########################################################################################

ps_val =  np.load(args.predictions_val+'_ps_val.npy')
ys_val =  np.load(args.predictions_val+'_ys_val.npy')
ws_val =  np.load(args.predictions_val+'_ws_val.npy')

ws_val /= np.sum(ws_val)

##########################################################################################
# Determine cutoffs for a range of target accuracies
##########################################################################################

hs_val = -np.sum(ps_val*np.log(ps_val), axis=1)
ms_val = np.equal(np.argmax(ys_val, axis=1), np.argmax(ps_val, axis=1)).astype(ps_val.dtype)

is_sorted = np.argsort(hs_val, axis=0)
hs_sorted = hs_val[is_sorted]
ms_sorted = ms_val[is_sorted]
ws_sorted = ws_val[is_sorted]

ns_sorted = np.cumsum(ws_sorted*ms_sorted, axis=0)
ds_sorted = np.cumsum(ws_sorted, axis=0)
as_sorted = ns_sorted/ds_sorted

cutoffs = {}
for target_accuracy in np.arange(0.5, 1.0, 0.01):

  is_range = np.arange(as_sorted.size, dtype=is_sorted.dtype)
  is_cutoff = is_range[as_sorted >= target_accuracy]
  if is_cutoff.size == 0:
    print('WARNING: Could not achieve target accuracy of '+str(100.0*target_accuracy)+'%')
    break
  i_cutoff = is_cutoff[-1]
  h_cutoff = hs_sorted[i_cutoff]

  cutoffs[target_accuracy] = h_cutoff

##########################################################################################
# Save preformance of model at each cutoff
##########################################################################################

with open(args.output, 'w') as stream:

  print(
    'target_accuracy',
    'cutoff',
    'cost_val',
    'accuracy_val',
    'fraction_val',
    sep=',', file=stream
  )

  for target_accuracy in sorted(cutoffs.keys()):

    cutoff = cutoffs[target_accuracy]

    hs_val = -np.sum(ps_val*np.log(ps_val), axis=1)
    cs_val = -np.sum(ys_val*np.log(ps_val), axis=1)
    ms_val = np.equal(np.argmax(ys_val, axis=1), np.argmax(ps_val, axis=1)).astype(ps_val.dtype)

    cs_val = cs_val[hs_val <= cutoff]
    ms_val = ms_val[hs_val <= cutoff]
    ws_val_ = ws_val[hs_val <= cutoff]
    ys_val_ = ys_val[hs_val <= cutoff]

    c_val = np.sum(ws_val_*cs_val)/np.sum(ws_val_)
    a_val = np.sum(ws_val_*ms_val)/np.sum(ws_val_)
    f_val = np.sum(ws_val_)
    n_val = np.sum(hs_val <= cutoff)
    m_val = np.mean(ys_val_)

    print(
      target_accuracy,
      cutoff,
      c_val/np.log(2.0),
      100.0*a_val,
      100.0*f_val,
      n_val,
      m_val,
      sep=',', file=stream
    )

