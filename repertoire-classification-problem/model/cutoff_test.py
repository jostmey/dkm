#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-01-08
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
parser.add_argument('--predictions_test', help='Path to the CSV file of predictions on the test data', type=str, required=True)
parser.add_argument('--cutoff', help='Entropy cutoff', type=float, required=True)
parser.add_argument('--index', help='Index of best fit', type=str, required=True)
parser.add_argument('--output', help='Path to CSV file where cutoffs will be stored', type=str, required=True)
args = parser.parse_args()

##########################################################################################
# Load data
##########################################################################################

ys_val = []
ws_val = []
ps_val = []
with open(args.predictions_test, 'r') as stream:
  reader = csv.DictReader(stream, delimiter=',')
  for row in reader:
    ys_val.append(row['Label'])
    ws_val.append(row['Weight'])
    ps_val.append(row['Prediction_'+str(args.index)])
ys_val = np.array(ys_val, np.float32)
ws_val = np.array(ws_val, np.float32)
ps_val = np.array(ps_val, np.float32)

ws_val /= np.sum(ws_val)

##########################################################################################
# Save preformance of model at each cutoff
##########################################################################################

with open(args.output, 'w') as stream:

  print(
    'cutoff',
    'cost_val',
    'accuracy_val',
    'fraction_val',
    sep=',', file=stream
  )

  hs_val = -ps_val*np.log(ps_val)-(1.0-ps_val)*np.log(1.0-ps_val)
  cs_val = -ys_val*np.log(ps_val)-(1.0-ys_val)*np.log(1.0-ps_val)
  ms_val = np.equal(np.round(ys_val), np.round(ps_val)).astype(ps_val.dtype)

  cs_val = cs_val[hs_val <= args.cutoff]
  ms_val = ms_val[hs_val <= args.cutoff]
  ws_val_ = ws_val[hs_val <= args.cutoff]
  ys_val_ = ys_val[hs_val <= args.cutoff]

  c_val = np.sum(ws_val_*cs_val)/np.sum(ws_val_)
  a_val = np.sum(ws_val_*ms_val)/np.sum(ws_val_)
  f_val = np.sum(ws_val_)
  n_val = np.sum(hs_val <= args.cutoff)
  m_val = np.mean(ys_val_)

  print(
    args.cutoff,
    c_val/np.log(2.0),
    100.0*a_val,
    100.0*f_val,
    n_val,
    m_val,
    sep=',', file=stream
  )

