#########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-04-17
# Environment: Python3
# Purpose: Utilities for creating a database of immune receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import csv
import numpy as np
import os.path
import h5py

##########################################################################################
# Load sequences
##########################################################################################

def load_receptors(path_tsv, min_cdr3_length=8, max_cdr3_length=32):
  receptors = {}
  with open(path_tsv, 'r') as stream:
    reader = csv.DictReader(stream, delimiter='\t')
    for row in reader:
      nns = row['nucleotide']  # See "docs" pages 24 and 25 for definition of row entries
      cdr3 = row['aminoAcid']
      vgene = row['vGeneName']
      dgene = row['dGeneName']
      jgene = row['jGeneName']
      quantity = np.float64(row['frequencyCount (%)'])
      status = row['sequenceStatus']
      if 'In' in status and min_cdr3_length <= len(cdr3) and len(cdr3) <= max_cdr3_length:
        if cdr3 not in receptors:
          receptors[cdr3] = quantity
        else:
          receptors[cdr3] += quantity
  return receptors

def normalize_receptors(receptors):
  total_quantity = np.float64(0.0)
  for quantity in sorted(receptors.values()):  # Add the smallest values together first preserves precision
    total_quantity += quantity
  for receptor in receptors.keys():
    receptors[receptor] /= total_quantity
  return receptors

##########################################################################################
# Database interface
##########################################################################################

def insert_receptors(path_db, name, receptors, max_cdr3_length=32):
  dtype = [
    ('cdr3', 'S'+str(max_cdr3_length)),
    ('frequency', 'f8')
  ]
  rs = np.zeros(len(receptors), dtype=dtype)
  for i, cdr3 in enumerate(sorted(receptors, key=receptors.get, reverse=True)):
    rs[i]['cdr3'] = cdr3
    rs[i]['frequency'] = receptors[cdr3]
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    rs_db = db.create_dataset(name, (rs.size,), dtype)
    rs_db[:] = rs

def insert_samples(path_db, name, samples):
  dtype = [
    ('sample', 'S32'),
    ('age', 'f8'),
    ('label', 'f8'),
    ('weight', 'f8')
  ]
  ss = np.zeros(len(samples), dtype=dtype)
  num_pos = 0.0
  for i, sample in enumerate(sorted(samples.keys())):
    if samples[sample]['diagnosis'] > 0.5:
      num_pos += 1.0
  num_neg = len(samples)-num_pos
  for i, sample in enumerate(sorted(samples.keys())):
    ss[i]['sample'] = sample
    ss[i]['age'] = samples[sample]['age']
    ss[i]['label'] = 1.0 if samples[sample]['diagnosis'] else 0.0
    ss[i]['weight'] = 0.5/num_pos if samples[sample]['diagnosis'] else 0.5/num_neg
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    ss_db = db.create_dataset(name, (ss.size,), dtype)
    ss_db[:] = ss

