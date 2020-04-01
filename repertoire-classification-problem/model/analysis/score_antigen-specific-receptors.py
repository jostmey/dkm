#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-10-12
# Purpose: Score antigen specific receptors
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import numpy as np
import h5py
from alignment import *

##########################################################################################
# Load model
##########################################################################################

# Load weights and bias terms
#
#ws_alignment = np.load('model_alignment_weights.npy')
bs_alignment = np.load('model_alignment_biases.npy')

ws_length = np.load('model_length_weights.npy')
bs_length = np.load('model_length_biases.npy')

#ws_abundance = np.load('model_abundance_weights.npy')
bs_abundance = np.load('model_abundance_biases.npy')

#ws_age = np.load('model_age_weights.npy')
bs_age = np.load('model_age_biases.npy')

# Load the similarity table
#
sm = load_similarity_matrix('similarity_table.csv')

##########################################################################################
# Features
##########################################################################################

with h5py.File('../../../antigen-classification-problem/dataset/database.h5', 'r') as db:

  receptors = db['Receptor-PMHC-Complex/all']

  with open('score_antigen-specific-receptors.csv', 'w') as stream:
    print('Source', 'pMHC', 'Weighted Score', 'Max Score', sep=',', file=stream, flush=True)
    for name in receptors[0].dtype.names:
      if 'frequency' in name and 'CMV' in name:
        numer = 0.0
        denom = 0.0
        logit_max = -1.0E16
        for receptor in receptors:
          frequency = receptor[name]
          if frequency > 0.0:
            cdr3 = receptor['trb_cdr3'].decode('utf-8')
            length = len(cdr3)
            align = do_alignment(sm, cdr3)
            score = align[0][-1][-1]
            logit = score+bs_alignment+np.dot(ws_length, length)+bs_length+bs_abundance+bs_age
            numer += frequency*logit
            denom += frequency
            if logit > logit_max:
              logit_max = logit
        name_ = name.split('_')
        name_ = name_[2]+':'+name_[1]
        print('CMV', name_, numer[0]/denom, logit_max[0], sep=',', file=stream, flush=True)
    for name in receptors[0].dtype.names:
      if 'frequency' in name and 'CMV' not in name:
        numer = 0.0
        denom = 0.0
        logit_max = -1.0E16
        for receptor in receptors:
          frequency = receptor[name]
          if frequency > 0.0:
            cdr3 = receptor['trb_cdr3'].decode('utf-8')
            length = len(cdr3)
            align = do_alignment(sm, cdr3)
            score = align[0][-1][-1]
            logit = score+bs_alignment+np.dot(ws_length, length)+bs_length+bs_abundance+bs_age
            numer += frequency*logit
            denom += frequency
            if logit > logit_max:
              logit_max = logit
        name_ = name.split('_')
        name_ = name_[2]+':'+name_[1]
        print('Not CMV', name_, numer[0]/denom, logit_max[0], sep=',', file=stream, flush=True)

