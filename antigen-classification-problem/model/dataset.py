##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-07-30
# Purpose: Methods for loading the dataset
##########################################################################################

import numpy as np
import h5py
import tensorflow as tf

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

def load_genes_embedding_dict(path_db):

  # Database
  #
  with h5py.File(path_db, 'r') as db:

    # Load gene calls
    #
    tra_vgenes_list = []
    tra_jgenes_list = []
    trb_vgenes_list = []
    trb_jgenes_list = []
    for folder_name in db:
      db_folder = db[folder_name]
      for subfolder_name in db_folder:
        tra_vgenes_list.append(
          db_folder[subfolder_name]['tra_vgene'][...]
        )
        tra_jgenes_list.append(
          db_folder[subfolder_name]['tra_jgene'][...]
        )
        trb_vgenes_list.append(
          db_folder[subfolder_name]['trb_vgene'][...]
        )
        trb_jgenes_list.append(
          db_folder[subfolder_name]['trb_jgene'][...]
        )

  # Get unique, sorted name calls as a string
  #
  tra_vgenes = np.concatenate(tra_vgenes_list, axis=0)
  tra_vgenes_unique = np.unique(tra_vgenes)
  tra_vgenes_sorted = np.sort(tra_vgenes_unique)
  tra_vgenes_str = tra_vgenes_sorted.astype('U')

  tra_jgenes = np.concatenate(tra_jgenes_list, axis=0)
  tra_jgenes_unique = np.unique(tra_jgenes)
  tra_jgenes_sorted = np.sort(tra_jgenes_unique)
  tra_jgenes_str = tra_jgenes_sorted.astype('U')

  trb_vgenes = np.concatenate(trb_vgenes_list, axis=0)
  trb_vgenes_unique = np.unique(trb_vgenes)
  trb_vgenes_sorted = np.sort(trb_vgenes_unique)
  trb_vgenes_str = trb_vgenes_sorted.astype('U')

  trb_jgenes = np.concatenate(trb_jgenes_list, axis=0)
  trb_jgenes_unique = np.unique(trb_jgenes)
  trb_jgenes_sorted = np.sort(trb_jgenes_unique)
  trb_jgenes_str = trb_jgenes_sorted.astype('U')

  # Create embeddings
  #
  tra_vgenes_dict = {}
  for i, vgene in enumerate(tra_vgenes_str):
    embedding = np.zeros(len(tra_vgenes_str), dtype=np.float32)
    embedding[i] = 1.0
    tra_vgenes_dict[vgene] = embedding

  tra_jgenes_dict = {}
  for i, jgene in enumerate(tra_jgenes_str):
    embedding = np.zeros(len(tra_jgenes_str), dtype=np.float32)
    embedding[i] = 1.0
    tra_jgenes_dict[jgene] = embedding

  trb_vgenes_dict = {}
  for i, vgene in enumerate(trb_vgenes_str):
    embedding = np.zeros(len(trb_vgenes_str), dtype=np.float32)
    embedding[i] = 1.0
    trb_vgenes_dict[vgene] = embedding

  trb_jgenes_dict = {}
  for i, jgene in enumerate(trb_jgenes_str):
    embedding = np.zeros(len(trb_jgenes_str), dtype=np.float32)
    embedding[i] = 1.0
    trb_jgenes_dict[jgene] = embedding

  return tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict

def load_dataset(path_db, split_name, tags, aminoacids_dict, tra_vgenes_dict, tra_jgenes_dict, trb_vgenes_dict, trb_jgenes_dict, max_steps=32, uniform=False, permute=False, convert_to_tensors=False):

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
  num_features_cdr3 = len(list(aminoacids_dict.values())[0])
  num_features_tra_vgene = len(list(tra_vgenes_dict.values())[0])
  num_features_tra_jgene = len(list(tra_jgenes_dict.values())[0])
  num_features_trb_vgene = len(list(trb_vgenes_dict.values())[0])
  num_features_trb_jgene = len(list(trb_jgenes_dict.values())[0])
  num_categories = len(tags)

  # Format data
  #
  features_tra_cdr3 = np.zeros([num_samples, max_steps, num_features_cdr3], dtype=np.float32)
  features_tra_vgene = np.zeros([num_samples, num_features_tra_vgene], dtype=np.float32)
  features_tra_jgene = np.zeros([num_samples, num_features_tra_jgene], dtype=np.float32)
  features_trb_cdr3 = np.zeros([num_samples, max_steps, num_features_cdr3], dtype=np.float32)
  features_trb_vgene = np.zeros([num_samples, num_features_trb_vgene], dtype=np.float32)
  features_trb_jgene = np.zeros([num_samples, num_features_trb_jgene], dtype=np.float32)
  labels = np.zeros([num_samples, num_categories], dtype=np.float32)
  weights = np.zeros([num_samples], dtype=np.float32)

  for i, receptor in enumerate(receptors):
    for j, aa in enumerate(receptor['tra_cdr3'].decode('utf-8')):
      features_tra_cdr3[i,j,:] = aminoacids_dict[aa]
    vgene = receptor['tra_vgene'].decode('utf-8')
    features_tra_vgene[i,:] = tra_vgenes_dict[vgene]
    jgene = receptor['tra_jgene'].decode('utf-8')
    features_tra_jgene[i,:] = tra_jgenes_dict[jgene]
    for j, aa in enumerate(receptor['trb_cdr3'].decode('utf-8')):
      features_trb_cdr3[i,j,:] = aminoacids_dict[aa]
    vgene = receptor['trb_vgene'].decode('utf-8')
    features_trb_vgene[i,:] = trb_vgenes_dict[vgene]
    jgene = receptor['trb_jgene'].decode('utf-8')
    features_trb_jgene[i,:] = trb_jgenes_dict[jgene]
  for tag in tags:
    weights += receptors['frequency_'+tag]
  weights /= num_categories
  for j, tag in enumerate(tags):
    labels[:,j] = receptors['frequency_'+tag]/(num_categories*weights)

  # Permute samples
  #
  if permute:
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    features_tra_cdr3 = features_tra_cdr3[indices]
    features_tra_vgene = features_tra_vgene[indices]
    features_tra_jgene = features_tra_jgene[indices]
    features_trb_cdr3 = features_trb_cdr3[indices]
    features_trb_vgene = features_trb_vgene[indices]
    features_trb_jgene = features_trb_jgene[indices]

  # Convert to tensors
  #
  if convert_to_tensors:
    features_tra_cdr3 = tf.convert_to_tensor(features_tra_cdr3, dtype=tf.float32)
    features_tra_vgene = tf.convert_to_tensor(features_tra_vgene, dtype=tf.float32)
    features_tra_jgene = tf.convert_to_tensor(features_tra_jgene, dtype=tf.float32)
    features_trb_cdr3 = tf.convert_to_tensor(features_trb_cdr3, dtype=tf.float32)
    features_trb_vgene = tf.convert_to_tensor(features_trb_vgene, dtype=tf.float32)
    features_trb_jgene = tf.convert_to_tensor(features_trb_jgene, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

  return features_tra_cdr3, features_tra_vgene, features_tra_jgene, features_trb_cdr3, features_trb_vgene, features_trb_jgene, labels, weights

