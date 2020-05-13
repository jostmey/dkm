##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-07-30
# Purpose: Methods for loading the dataset
##########################################################################################

import numpy as np
import h5py
import tensorflow as tf
import os
import csv

def load_aminoacid_embedding_dict(path_embedding):

  # Amino acid factors
  #
  names = []
  factors = []
  with open(path_embedding, 'r') as stream:
    for line in stream:
      rows = line.split(',')
      names.append(rows[0])
      factors.append(np.array(rows[1:], dtype=np.float32))
  names = np.array(names)
  factors = np.array(factors)

  # Convert into a dictionary
  #
  aminoacids_dict = { name: factors[i,:] for i, name in enumerate(names) }

  return aminoacids_dict

def load_dataset(path_db, cohort, split, aminoacids_dict, max_steps=32, path_shuffle=None):

  # Database
  #
  with h5py.File(path_db, 'r') as db:

    # Load the overviews
    #
    samples = db[cohort+'/'+split][...]

    # List of samples
    #
    features = {}
    labels = {}
    weights = {}

    # Loop over every sample
    #
    for sample in samples:

      # Load the receptors
      #
      sample_ = sample['sample'].decode('utf-8')
      receptors = db[cohort+'/features/'+sample_][...]

      # Settings
      #
      num_samples = receptors.size
      num_features_cdr3 = len(list(aminoacids_dict.values())[0])

      # Format data
      #
      features_cdr3 = np.zeros([num_samples, max_steps, num_features_cdr3], dtype=np.float32)
      features_quantity = np.zeros([num_samples], dtype=np.float32)

      for i, receptor in enumerate(receptors):
        for j, aa in enumerate(receptor['cdr3'].decode('utf-8')):
          features_cdr3[i,j,:] = aminoacids_dict[aa]
        features_quantity[i] = receptor['frequency']

      # Update list of samples
      #
      features[sample_] = {
        'cdr3': features_cdr3,
        'quantity': features_quantity,
        'age': np.float32(sample['age'])
      }
      labels[sample_] = sample['label']
      weights[sample_] = sample['weight']

  # Permute samples
  #
  if path_shuffle is not None:
    if os.path.isfile(path_shuffle):
      keys = []
      keys_ = []
      with open(path_shuffle, 'r') as stream:
        reader = csv.reader(stream, delimiter=',')
        for row in reader:
          keys.append(row[0])
          keys_.append(row[1])
    else:
      keys = list(features.keys())
      keys_ = list(features.keys())
      np.random.shuffle(keys_)
      with open(path_shuffle, 'w') as stream:
        for key, key_ in zip(keys, keys_):
          print(key, key_, sep=',', file=stream)
    features = { key: features[key_] for key, key_ in zip(keys, keys_) }

  return features, labels, weights
