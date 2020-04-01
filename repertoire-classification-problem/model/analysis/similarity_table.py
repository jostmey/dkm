#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-10-02
# Purpose: Dump similarity table
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
parser.add_argument('--input', help='Input basename for dumping weights', type=str, required=True)
parser.add_argument('--output', help='Filename for similarity table', type=str, required=True)
args = parser.parse_args()

##########################################################################################
# Load model
##########################################################################################

# Load the embeddings for the alignment layer
#
aminoacid_embeddings = {}
with open(args.input+'_alignment_embedding.csv', 'r') as stream:
  for line in stream:
    data = line.split(',')
    aminoacid = data[0]
    values = np.array([ float(value) for value in data[1:] ])
    aminoacid_embeddings[aminoacid] = values

# Load the alignment weights
#
weights = np.load(args.input+'_alignment_weights.npy')

##########################################################################################
# Compute similarity values and save results
##########################################################################################

with open(args.output, 'w') as stream:

  string = ''
  for aminoacid in sorted(aminoacid_embeddings.keys()):
    string += ','+aminoacid
  print(string, sep=',', file=stream)

  for i in range(weights.shape[0]):
    string = ''
    for aminoacid in sorted(aminoacid_embeddings.keys()):
      values = aminoacid_embeddings[aminoacid]
      similarity = np.dot(values, weights[i,:])
      string += ','+str(similarity)
    string = string[1:]
    print(i+1, string, sep=',', file=stream)

