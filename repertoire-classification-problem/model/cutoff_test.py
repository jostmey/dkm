#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-01-08
# Purpose: Dump the model's preformance for a given confidence cutoff
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

ys_test = []
ws_test = []
ps_test = []
with open(args.predictions_test, 'r') as stream:
  reader = csv.DictReader(stream, delimiter=',')
  for row in reader:
    ys_test.append(row['Label'])
    ws_test.append(row['Weight'])
    ps_test.append(row['Prediction_'+str(args.index)])
ys_test = np.array(ys_test, np.float32)
ws_test = np.array(ws_test, np.float32)
ps_test = np.array(ps_test, np.float32)

ws_test /= np.sum(ws_test)

##########################################################################################
# Save preformance of model at each cutoff
##########################################################################################

with open(args.output, 'w') as stream:

  print(
    'cutoff',
    'cost_test',
    'accuracy_test',
    'fraction_test',
    sep=',', file=stream
  )

  hs_test = -ps_test*np.log(ps_test)-(1.0-ps_test)*np.log(1.0-ps_test)
  cs_test = -ys_test*np.log(ps_test)-(1.0-ys_test)*np.log(1.0-ps_test)
  ms_test = np.equal(np.round(ys_test), np.round(ps_test)).astype(ps_test.dtype)

  cs_test = cs_test[hs_test <= args.cutoff]
  ms_test = ms_test[hs_test <= args.cutoff]
  ws_test_ = ws_test[hs_test <= args.cutoff]
  ys_test_ = ys_test[hs_test <= args.cutoff]

  c_test = np.sum(ws_test_*cs_test)/np.sum(ws_test_)
  a_test = np.sum(ws_test_*ms_test)/np.sum(ws_test_)
  f_test = np.sum(ws_test_)
  n_test = np.sum(hs_test <= args.cutoff)
  m_test = np.mean(ys_test_)

  print(
    args.cutoff,
    c_test/np.log(2.0),
    100.0*a_test,
    100.0*f_test,
    sep=',', file=stream
  )

