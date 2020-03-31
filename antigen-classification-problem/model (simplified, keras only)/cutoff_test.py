#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-01-08
# Purpose: Calculate the model's performance for the given confidence cutoff
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
parser.add_argument('--output', help='Path to CSV file where cutoffs will be stored', type=str, required=True)
args = parser.parse_args()

##########################################################################################
# Load data
##########################################################################################

ps_test =  np.load(args.predictions_test+'_ps_test.npy')
ys_test =  np.load(args.predictions_test+'_ys_test.npy')
ws_test =  np.load(args.predictions_test+'_ws_test.npy')

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

  hs_test = -np.sum(ps_test*np.log(ps_test), axis=1)
  cs_test = -np.sum(ys_test*np.log(ps_test), axis=1)
  ms_test = np.equal(np.argmax(ys_test, axis=1), np.argmax(ps_test, axis=1)).astype(ps_test.dtype)

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

