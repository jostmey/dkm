##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-07-30
# Purpose: Methods for loading the dataset
##########################################################################################

import numpy as np
import h5py
import tensorflow as tf

def load_dataset(path_db, split_name, tags, max_steps=32, uniform=False, permute=False):

  # Database
  #
  with h5py.File(path_db, 'r') as db:
    receptors = db[split_name][...]

  # Remove samples where the weight will be zero
  #
  weights = 0.0
  for tag in tags:
    weights += receptors['frequency_'+tag]
  indices = np.argwhere(weights > 0.0).flatten()
  receptors = receptors[indices]

  # Uniformly distribute the receptor abundances
  #
  if uniform:
    for tag in tags:
      receptors['frequency_'+tag] = np.sign(receptors['frequency_'+tag])

  # Normalize sample weights
  #
  for tag in tags:
    receptors['frequency_'+tag] /= np.sum(receptors['frequency_'+tag])

  # Settings
  #
  num_samples = receptors.size
  num_categories = len(tags)

  # Format data
  #
  tras = []
  trbs = []
  labels = np.zeros([num_samples, num_categories], dtype=np.float32)
  weights = np.zeros([num_samples], dtype=np.float32)

  for i, receptor in enumerate(receptors):
    tras.append(
      receptor['tra_vgene'].decode('utf-8')+':'+receptor['tra_cdr3'].decode('utf-8')+':'+receptor['tra_jgene'].decode('utf-8')
    )
    trbs.append(
      receptor['trb_vgene'].decode('utf-8')+':'+receptor['trb_cdr3'].decode('utf-8')+':'+receptor['trb_jgene'].decode('utf-8')
    )
  for tag in tags:
    weights += receptors['frequency_'+tag]
  weights /= num_categories
  for j, tag in enumerate(tags):
    labels[:,j] = receptors['frequency_'+tag]/(num_categories*weights)
  tras = np.array(tras)
  trbs = np.array(trbs)

  # Permute samples
  #
  if permute:
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    tras = tras[indices]
    trbs = trbs[indices]

  return tras, trbs, labels, weights

