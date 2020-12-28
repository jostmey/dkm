#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-12-27
# Purpose: Convert database of TCR sequences to a CSV file for TCR-Dist
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
from dataset import *
import numpy as np

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--database', help='Path to the database', type=str, required=True)
parser.add_argument('--project', help='Name of the project in the database', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--permute', help='Randomly permute the relationship between features and labels', type=bool, default=False)
args = parser.parse_args()

##########################################################################################
# Load datasets
##########################################################################################

# Settings
#
max_steps = 32

# Load the samples
#
tras_train, trbs_train, ys_train, fs_train = \
  load_dataset(args.database, args.project+'/train', args.tags, max_steps=max_steps, permute=args.permute)
tras_val, trbs_val, ys_val, fs_val = \
  load_dataset(args.database, args.project+'/validate', args.tags, max_steps=max_steps, permute=args.permute)
tras_test, trbs_test, ys_test, fs_test = \
  load_dataset(args.database, args.project+'/test', args.tags, max_steps=max_steps, permute=args.permute)

##########################################################################################
# Save results
##########################################################################################

tags = [ ':'.join(tag.split('_')[:2]) for tag in args.tags ]

with open(args.output, 'w') as stream:
  print(
    'v_a_gene', 'j_a_gene', 'cdr3_a_aa',
    'v_b_gene', 'j_b_gene', 'cdr3_b_aa',
    'clone_id', 'epitope',
    'count', 'frequency',
    'split',
    sep=',', file=stream
  )

  i = 0

  f_min = np.min(fs_train)
  for tra, trb, y, f in zip(tras_train, trbs_train, ys_train, fs_train):
    tra_vgene, tra_cdr3, tra_jgene = tra.split(':')
    trb_vgene, trb_cdr3, trb_jgene = trb.split(':')
    j = np.argmax(y)
    tag = tags[j]
    print(
      tra_vgene.replace('DV', '/DV')+'*01', tra_jgene.replace('DV', '/DV')+'*01', tra_cdr3,
      trb_vgene.replace('DV', '/DV')+'*01', trb_jgene.replace('DV', '/DV')+'*01', trb_cdr3,
      i, tag,
      int(f/f_min), f,
      'train',
      sep=',', file=stream
    )
    i += 1

  f_min = np.min(fs_val)
  for tra, trb, y, f in zip(tras_val, trbs_val, ys_val, fs_val):
    tra_vgene, tra_cdr3, tra_jgene = tra.split(':')
    trb_vgene, trb_cdr3, trb_jgene = trb.split(':')
    j = np.argmax(y)
    tag = tags[j]
    print(
      tra_vgene.replace('DV', '/DV')+'*01', tra_jgene.replace('DV', '/DV')+'*01', tra_cdr3,
      trb_vgene.replace('DV', '/DV')+'*01', trb_jgene.replace('DV', '/DV')+'*01', trb_cdr3,
      i, tag,
      int(f/f_min), f,
      'validate',
      sep=',',file=stream
    )
    i += 1

  f_min = np.min(fs_test)
  for tra, trb, y, f in zip(tras_test, trbs_test, ys_test, fs_test):
    tra_vgene, tra_cdr3, tra_jgene = tra.split(':')
    trb_vgene, trb_cdr3, trb_jgene = trb.split(':')
    j = np.argmax(y)
    tag = tags[j]
    print(
      tra_vgene.replace('DV', '/DV')+'*01', tra_jgene.replace('DV', '/DV')+'*01', tra_cdr3,
      trb_vgene.replace('DV', '/DV')+'*01', trb_jgene.replace('DV', '/DV')+'*01', trb_cdr3,
      i, tag,
      int(f/f_min), f,
      'test',
      sep=',', file=stream
    )
    i += 1

