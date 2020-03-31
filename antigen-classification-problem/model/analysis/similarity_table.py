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
parser.add_argument('--output', help='Output basename for similarity tables', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
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
tra_weights = np.load(args.input+'_tra_alignment_weights.npy')
trb_weights = np.load(args.input+'_trb_alignment_weights.npy')

##########################################################################################
# Compute similarity values and save results
##########################################################################################

for k in range(tra_weights.shape[2]):
  with open(args.output+'_'+args.tags[k]+'_tra.csv', 'w') as stream:

    string = ''
    for aminoacid in sorted(aminoacid_embeddings.keys()):
      string += ','+aminoacid
    print(string, sep=',', file=stream)

    for i in range(tra_weights.shape[0]):
      string = ''
      for aminoacid in sorted(aminoacid_embeddings.keys()):
        values = aminoacid_embeddings[aminoacid]
        similarity = np.dot(values, tra_weights[i,:,k])
        string += ','+str(similarity)
      string = string[1:]
      print(i+1, string, sep=',', file=stream)

for k in range(trb_weights.shape[2]):
  with open(args.output+'_'+args.tags[k]+'_trb.csv', 'w') as stream:

    string = ''
    for aminoacid in sorted(aminoacid_embeddings.keys()):
      string += ','+aminoacid
    print(string, sep=',', file=stream)

    for i in range(trb_weights.shape[0]):
      string = ''
      for aminoacid in sorted(aminoacid_embeddings.keys()):
        values = aminoacid_embeddings[aminoacid]
        similarity = np.dot(values, trb_weights[i,:,k])
        string += ','+str(similarity)
      string = string[1:]
      print(i+1, string, sep=',', file=stream)
