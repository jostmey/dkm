#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-10-12
# Purpose: Preform in-silico alanine scan
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import csv
import numpy as np
from alignment import *

import scipy.special as ss

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input basename for weights', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--tables', help='Basename to tables with similarity values', type=str, required=True)
parser.add_argument('--output', help='Output basename for CSV files', type=str, required=True)
parser.add_argument('--tra_vgene', type=str, required=True)
parser.add_argument('--tra_cdr3', type=str, required=True)
parser.add_argument('--tra_jgene', type=str, required=True)
parser.add_argument('--trb_vgene', type=str, required=True)
parser.add_argument('--trb_cdr3', type=str, required=True)
parser.add_argument('--trb_jgene', type=str, required=True)
args = parser.parse_args()

##########################################################################################
# Settings
##########################################################################################

max_steps = 32

##########################################################################################
# Load model
##########################################################################################

# Load embeddings
#
aminoacids = {}
with open(args.input+'_alignment_embedding.csv', 'r') as stream:
  reader = csv.reader(stream, delimiter=',')
  for row in reader:
    key = row[0]
    values = np.array([ float(v) for v in row[1:] ], dtype=np.float32)
    aminoacids[key] = values
aminoacids['.'] = np.zeros(aminoacids['A'].shape, dtype=np.float32)

tra_vgenes = {}
with open(args.input+'_tra_vgene_embedding.csv', 'r') as stream:
  reader = csv.reader(stream, delimiter=',')
  for row in reader:
    key = row[0]
    values = np.array([ float(v) for v in row[1:] ], dtype=np.float32)
    tra_vgenes[key] = values

tra_jgenes = {}
with open(args.input+'_tra_jgene_embedding.csv', 'r') as stream:
  reader = csv.reader(stream, delimiter=',')
  for row in reader:
    key = row[0]
    values = np.array([ float(v) for v in row[1:] ], dtype=np.float32)
    tra_jgenes[key] = values

trb_vgenes = {}
with open(args.input+'_trb_vgene_embedding.csv', 'r') as stream:
  reader = csv.reader(stream, delimiter=',')
  for row in reader:
    key = row[0]
    values = np.array([ float(v) for v in row[1:] ], dtype=np.float32)
    trb_vgenes[key] = values

trb_jgenes = {}
with open(args.input+'_trb_jgene_embedding.csv', 'r') as stream:
  reader = csv.reader(stream, delimiter=',')
  for row in reader:
    key = row[0]
    values = np.array([ float(v) for v in row[1:] ], dtype=np.float32)
    trb_jgenes[key] = values

# Load weights and bias terms
#
tra_ws_alignment = np.load(args.input+'_tra_alignment_weights.npy')
tra_bs_alignment = np.load(args.input+'_tra_alignment_biases.npy')

tra_ws_length = np.load(args.input+'_tra_length_weights.npy')
tra_bs_length = np.load(args.input+'_tra_length_biases.npy')

tra_ws_vgene = np.load(args.input+'_tra_vgene_weights.npy')
tra_bs_vgene = np.load(args.input+'_tra_vgene_biases.npy')

tra_ws_jgene = np.load(args.input+'_tra_jgene_weights.npy')
tra_bs_jgene = np.load(args.input+'_tra_jgene_biases.npy')

trb_ws_alignment = np.load(args.input+'_trb_alignment_weights.npy')
trb_bs_alignment = np.load(args.input+'_trb_alignment_biases.npy')

trb_ws_length = np.load(args.input+'_trb_length_weights.npy')
trb_bs_length = np.load(args.input+'_trb_length_biases.npy')

trb_ws_vgene = np.load(args.input+'_trb_vgene_weights.npy')
trb_bs_vgene = np.load(args.input+'_trb_vgene_biases.npy')

trb_ws_jgene = np.load(args.input+'_trb_jgene_weights.npy')
trb_bs_jgene = np.load(args.input+'_trb_jgene_biases.npy')

##########################################################################################
# Features
##########################################################################################

# Align CDR3 sequences
#
tra_cdr3_aligned = []
trb_cdr3_aligned = []
for tag in args.tags:
  sm_tra = load_similarity_matrix(args.tables+'_'+tag+'_tra.csv')
  tra_align = do_alignment(sm_tra, args.tra_cdr3)
  tra_cdr3_aligned.append(
    print_alignment(tra_align[1], args.tra_cdr3)
  )
  sm_trb = load_similarity_matrix(args.tables+'_'+tag+'_trb.csv')
  trb_align = do_alignment(sm_trb, args.trb_cdr3)
  trb_cdr3_aligned.append(
    print_alignment(trb_align[1], args.trb_cdr3)
  )

# Convert to features
#
x_tra_vgene = tra_vgenes[args.tra_vgene]
x_tra_cdr3 = []
for seq in tra_cdr3_aligned:
  x_tra_cdr3.append(
    np.array([ aminoacids[aa] for aa in seq ])
  )
x_tra_L = np.array([ len(args.tra_cdr3) ], dtype=np.float32)
x_tra_jgene = tra_jgenes[args.tra_jgene]

x_trb_vgene = trb_vgenes[args.trb_vgene]
x_trb_cdr3 = []
for seq in trb_cdr3_aligned:
  x_trb_cdr3.append(
    np.array([ aminoacids[aa] for aa in seq ])
  )
x_trb_L = np.array([ len(args.trb_cdr3) ], dtype=np.float32)
x_trb_jgene = trb_jgenes[args.trb_jgene]

##########################################################################################
# Compute model
##########################################################################################

ls_tra_cdr3 = []
for i, x in enumerate(x_tra_cdr3):
  ls_tra_cdr3.append(
    np.sum(x*tra_ws_alignment[:,:,i])/np.sqrt(x_tra_L[0])+tra_bs_alignment[i]
  )
ls_tra_cdr3 = np.array(ls_tra_cdr3)
ls_trb_cdr3 = []
for i, x in enumerate(x_trb_cdr3):
  ls_trb_cdr3.append(
    np.sum(x*trb_ws_alignment[:,:,i])/np.sqrt(x_trb_L[0])+trb_bs_alignment[i]
  )
ls_trb_cdr3 = np.array(ls_trb_cdr3)

ls_tra_L = np.dot(x_tra_L, tra_ws_length)+tra_bs_length
ls_trb_L = np.dot(x_trb_L, trb_ws_length)+trb_bs_length

ls_tra_vgene = np.dot(x_tra_vgene, tra_ws_vgene)+tra_bs_vgene
ls_tra_jgene = np.dot(x_tra_jgene, tra_ws_jgene)+tra_bs_jgene

ls_trb_vgene = np.dot(x_trb_vgene, trb_ws_vgene)+trb_bs_vgene
ls_trb_jgene = np.dot(x_trb_jgene, trb_ws_jgene)+trb_bs_jgene

ls = ls_tra_vgene+ls_tra_cdr3+ls_tra_L+ls_tra_jgene+ \
     ls_trb_vgene+ls_trb_cdr3+ls_trb_L+ls_trb_jgene

ls_offset = ls-np.max(ls)
ps = np.exp(ls_offset)/np.sum(np.exp(ls_offset))

#print(ps)

##########################################################################################
# Save results
##########################################################################################

with open(args.output+'_tra.csv', 'w') as stream:
  print('Position', 'CDR3', 'Delta Log Probablity', 'Delta Logit', sep=',', file=stream)
  for k in range(len(args.tra_cdr3)):
    tra_cdr3_ = args.tra_cdr3[:k]+'A'+args.tra_cdr3[k+1:]

    # Align CDR3 sequences
    #
    tra_cdr3_aligned_ = []
    for tag in args.tags:
      sm_tra = load_similarity_matrix(args.tables+'_'+tag+'_tra.csv')
      tra_align = do_alignment(sm_tra, tra_cdr3_)
      tra_cdr3_aligned_.append(
        print_alignment(tra_align[1], tra_cdr3_)
      )

    # Recompute features
    #
    x_tra_cdr3_ = []
    for seq in tra_cdr3_aligned_:
      x_tra_cdr3_.append(
        np.array([ aminoacids[aa] for aa in seq ])
      )

    # Recompute logits
    ls_tra_cdr3_ = []
    for i, x in enumerate(x_tra_cdr3_):
      ls_tra_cdr3_.append(
        np.sum(x*tra_ws_alignment[:,:,i])/np.sqrt(x_tra_L[0])+tra_bs_alignment[i]
      )

    ls_ = ls_tra_vgene+ls_tra_cdr3_+ls_tra_L+ls_tra_jgene+ \
          ls_trb_vgene+ls_trb_cdr3+ls_trb_L+ls_trb_jgene

    ls_offset_ = ls_-np.max(ls_)
    ps_ = np.exp(ls_offset_)/np.sum(np.exp(ls_offset_))

    print(k, tra_cdr3_, np.log(ps_[0])-np.log(ps[0]), ls_[0]-ls[0], sep=',', file=stream)

with open(args.output+'_trb.csv', 'w') as stream:
  print('Position', 'CDR3', 'Delta Log Probablity', 'Delta Logit', sep=',', file=stream)
  for k in range(len(args.trb_cdr3)):
    trb_cdr3_ = args.trb_cdr3[:k]+'A'+args.trb_cdr3[k+1:]

    # Align CDR3 sequences
    #
    trb_cdr3_aligned_ = []
    for tag in args.tags:
      sm_trb = load_similarity_matrix(args.tables+'_'+tag+'_trb.csv')
      trb_align = do_alignment(sm_trb, trb_cdr3_)
      trb_cdr3_aligned_.append(
        print_alignment(trb_align[1], trb_cdr3_)
      )

    # Recompute features
    #
    x_trb_cdr3_ = []
    for seq in trb_cdr3_aligned_:
      x_trb_cdr3_.append(
        np.array([ aminoacids[aa] for aa in seq ])
      )

    # Recompute logits
    ls_trb_cdr3_ = []
    for i, x in enumerate(x_trb_cdr3_):
      ls_trb_cdr3_.append(
        np.sum(x*trb_ws_alignment[:,:,i])/np.sqrt(x_trb_L[0])+trb_bs_alignment[i]
      )

    ls_ = ls_tra_vgene+ls_tra_cdr3+ls_tra_L+ls_tra_jgene+ \
          ls_trb_vgene+ls_trb_cdr3_+ls_trb_L+ls_trb_jgene

    ls_offset_ = ls_-np.max(ls_)
    ps_ = np.exp(ls_offset_)/np.sum(np.exp(ls_offset_))

    print(k, trb_cdr3_, np.log(ps_[0])-np.log(ps[0]), ls_[0]-ls[0], sep=',', file=stream)
