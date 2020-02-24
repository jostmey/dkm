#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-11-26
# Purpose: Find best fit to the training data
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import csv
import os

##########################################################################################
# Settings
##########################################################################################

num_fits = 128

##########################################################################################
# Find best fits
##########################################################################################

bestfits = {}
for fit in range(1, num_fits+1):
  path = 'bin/train_val_'+str(fit)+'.out'
  if os.path.isfile(path):
    with open(path, 'r') as stream:
      reader = csv.reader(stream, delimiter='\t')
      for row in reader:
        i_step = int(row[0])
        i_best = int(row[5])
        c_train = float(row[6])
        a_train = float(row[7])
        c_val = float(row[8])
        a_val = float(row[9])
        if i_step not in bestfits:
          bestfits[i_step] = {
            'fit': fit,
            'best': i_best,
            'c_train': c_train,
            'a_train': a_train,
            'c_val': c_val,
            'a_val': a_val
          }
        elif c_train < bestfits[i_step]['c_train']:
          bestfits[i_step]['fit'] = fit
          bestfits[i_step]['best'] = i_best
          bestfits[i_step]['c_train'] = c_train
          bestfits[i_step]['a_train'] = a_train
          bestfits[i_step]['c_val'] = c_val
          bestfits[i_step]['a_val'] = a_val

##########################################################################################
# Print results
##########################################################################################

steps = sorted(list(bestfits.keys()))

for step in steps:
  print(
    step,
    bestfits[step]['fit'],
    bestfits[step]['best'],
    '%4.3f'%(bestfits[step]['c_train']),
    '%4.3f'%(bestfits[step]['a_train']),
    '%4.3f'%(bestfits[step]['c_val']),
    '%4.3f'%(bestfits[step]['a_val']),
    sep='\t'
  )




