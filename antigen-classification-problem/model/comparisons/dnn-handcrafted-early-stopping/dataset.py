##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-07-30
# Purpose: Methods for loading the dataset
##########################################################################################

import numpy as np
import h5py
import pandas as pd
from pyteomics import electrochem, mass, parser
from sklearn import feature_extraction, metrics

def handcrafted_features(data, tags):

  #
  # DOI 10.1007/s00251-017-1023-5
  # Code from https://github.com/bittremieux/TCR-Classifier/blob/master/tcr_classifier.ipynb
  # Modified to apply handcrafted features twice, once to the alpha chain and again to the beta chain
  # Modified to handle split for training, validation, and test cohorts
  # Modified for multinomial classification
  #

  # physicochemical amino acid properties
  basicity = {
    'A': 206.4, 'B': 210.7, 'C': 206.2, 'D': 208.6, 'E': 215.6, 'F': 212.1, 'G': 202.7,
    'H': 223.7, 'I': 210.8, 'K': 221.8, 'L': 209.6, 'M': 213.3, 'N': 212.8, 'P': 214.4,
    'Q': 214.2, 'R': 237.0, 'S': 207.6, 'T': 211.7, 'V': 208.7, 'W': 216.1, 'X': 210.2,
    'Y': 213.1, 'Z': 214.9
  }

  hydrophobicity = {
    'A': 0.16, 'B': -3.14, 'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31,
    'H': -4.63, 'I': 4.41, 'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92,
    'Q': -2.76, 'R': -2.77, 'S': -2.85, 'T': -1.08, 'V': 3.02, 'W': 4.88, 'X': 4.59,
    'Y': 2.00, 'Z': -2.13
  }

  helicity = {
    'A': 1.24, 'B': 0.92, 'C': 0.79, 'D': 0.89, 'E': 0.85, 'F': 1.26, 'G': 1.15, 'H': 0.97,
    'I': 1.29, 'K': 0.88, 'L': 1.28, 'M': 1.22, 'N': 0.94, 'P': 0.57, 'Q': 0.96, 'R': 0.95,
    'S': 1.00, 'T': 1.09, 'V': 1.27, 'W': 1.07, 'X': 1.29, 'Y': 1.11, 'Z': 0.91
  }

  mutation_stability = {
    'A': 13, 'C': 52, 'D': 11, 'E': 12, 'F': 32, 'G': 27, 'H': 15, 'I': 10,
    'K': 24, 'L': 34, 'M':  6, 'N':  6, 'P': 20, 'Q': 10, 'R': 17, 'S': 10,
    'T': 11, 'V': 17, 'W': 55, 'Y': 31
  }

  # feature conversion and generation
  features_list = []

  for chain in [ 'tra', 'trb' ]:

    onehot_encoder = feature_extraction.DictVectorizer(sparse=False)
    features_list.append(
      pd.DataFrame(
        onehot_encoder.fit_transform(data[[chain+'_vgene', chain+'_jgene']].to_dict(orient='records')),
        columns=onehot_encoder.feature_names_
      )
    )

    # sequence length
    features_list.append(data[chain+'_cdr3'].apply(
      lambda sequence: parser.length(sequence)).to_frame().rename(columns={chain+'_cdr3': 'length'})
    )

    # number of occurences of each amino acid
    aa_counts = pd.DataFrame.from_records(
      [parser.amino_acid_composition(sequence) for sequence in data[chain+'_cdr3']]
    ).fillna(0)
    aa_counts.columns = [chain+'_count_{}'.format(column) for column in aa_counts.columns]
    features_list.append(aa_counts)

    # physicochemical properties: (average) basicity, (average) hydrophobicity,
    #                             (average) helicity, pI, (average) mutation stability
    features_list.append(data[chain+'_cdr3'].apply(
      lambda seq: sum([basicity[aa] for aa in seq])/parser.length(seq)
    ).to_frame().rename(columns={chain+'_cdr3': 'avg_basicity'}))
    features_list.append(data[chain+'_cdr3'].apply(
      lambda seq: sum([hydrophobicity[aa] for aa in seq])/parser.length(seq)
    ).to_frame().rename(columns={chain+'_cdr3': 'avg_hydrophobicity'}))
    features_list.append(data[chain+'_cdr3'].apply(
      lambda seq: sum([helicity[aa] for aa in seq])/parser.length(seq)
    ).to_frame().rename(columns={chain+'_cdr3': 'avg_helicity'}))
    features_list.append(data[chain+'_cdr3'].apply(
      lambda seq: electrochem.pI(seq)).to_frame().rename(columns={chain+'_cdr3': 'pI'}))
    features_list.append(data[chain+'_cdr3'].apply(
      lambda seq: sum([mutation_stability[aa] for aa in seq])/parser.length(seq)
    ).to_frame().rename(columns={chain+'_cdr3': 'avg_mutation_stability'}))

    # peptide mass
    features_list.append(data[chain+'_cdr3'].apply(
      lambda seq: mass.fast_mass(seq)).to_frame().rename(columns={chain+'_cdr3': 'mass'})
    )

    # positional features
    # amino acid occurence and physicochemical properties at a given position from the center
    pos_aa, pos_basicity, pos_hydro, pos_helicity, pos_pI, pos_mutation = [[] for _ in range(6)]
    for sequence in data[chain+'_cdr3']:
      length = parser.length(sequence)
      start_pos = -1 * (length // 2)
      pos_range = list(range(start_pos, start_pos + length)) if length % 2 == 1 else\
        list(range(start_pos, 0)) + list(range(1, start_pos + length + 1))
    
      pos_aa.append(
        {chain+'_pos_{}_{}'.format(pos, aa): 1 for pos, aa in zip(pos_range, sequence)}
      )
      pos_basicity.append(
        {chain+'_pos_{}_basicity'.format(pos): basicity[aa] for pos, aa in zip(pos_range, sequence)}
      )
      pos_hydro.append(
        {chain+'_pos_{}_hydrophobicity'.format(pos): hydrophobicity[aa] for pos, aa in zip(pos_range, sequence)}
      )
      pos_helicity.append(
        {chain+'_pos_{}_helicity'.format(pos): helicity[aa] for pos, aa in zip(pos_range, sequence)}
      )
      pos_pI.append(
        {chain+'_pos_{}_pI'.format(pos): electrochem.pI(aa) for pos, aa in zip(pos_range, sequence)}
      )
      pos_mutation.append(
        {chain+'_pos_{}_mutation_stability'.format(pos): mutation_stability[aa] for pos, aa in zip(pos_range, sequence)}
      )

    features_list.append(pd.DataFrame.from_records(pos_aa).fillna(0))
    features_list.append(pd.DataFrame.from_records(pos_basicity).fillna(0))
    features_list.append(pd.DataFrame.from_records(pos_hydro).fillna(0))
    features_list.append(pd.DataFrame.from_records(pos_helicity).fillna(0))
    features_list.append(pd.DataFrame.from_records(pos_pI).fillna(0))
    features_list.append(pd.DataFrame.from_records(pos_mutation).fillna(0))

  features_list.append(data['weights'])
  for tag in tags:
    features_list.append(data['labels_'+tag])
  features_list.append(data['split'])

  # combine all features
  data_processed = pd.concat(features_list, axis=1)

  return data_processed

def load_datasets(path_db, splits, tags, uniform=False, permute=False):

  # Settings
  #
  num_categories = len(tags)

  # For transferring the hdf5 tables to a dataframe
  #
  receptors_dict = {}

  # Loop over each split of the data
  #
  for split in splits:

    # Database
    #
    with h5py.File(path_db, 'r') as db:
      receptors = db[split][...]

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

    if 'tra_vgene' not in receptors_dict:
      receptors_dict['tra_vgene'] = np.char.decode(receptors['tra_vgene'])
      receptors_dict['tra_cdr3'] = np.char.decode(receptors['tra_cdr3'])
      receptors_dict['tra_jgene'] = np.char.decode(receptors['tra_jgene'])
      receptors_dict['trb_vgene'] = np.char.decode(receptors['trb_vgene'])
      receptors_dict['trb_cdr3'] = np.char.decode(receptors['trb_cdr3'])
      receptors_dict['trb_jgene'] = np.char.decode(receptors['trb_jgene'])
      weights = 0.0
      for tag in tags:
        weights += receptors['frequency_'+tag]
      weights /= num_categories
      receptors_dict['weights'] = weights
      for j, tag in enumerate(tags):
        receptors_dict['labels_'+tag] = receptors['frequency_'+tag]/(num_categories*weights)
      receptors_dict['split'] = [ split for i in range(receptors.size) ]
    else:
      receptors_dict['tra_vgene'] = np.concatenate(
        [ receptors_dict['tra_vgene'], np.char.decode(receptors['tra_vgene']) ],
        axis=0
      )
      receptors_dict['tra_cdr3'] = np.concatenate(
        [ receptors_dict['tra_cdr3'], np.char.decode(receptors['tra_cdr3']) ],
        axis=0
      )
      receptors_dict['tra_jgene'] = np.concatenate(
        [ receptors_dict['tra_jgene'], np.char.decode(receptors['tra_jgene']) ],
        axis=0
      )
      receptors_dict['trb_vgene'] = np.concatenate(
        [ receptors_dict['trb_vgene'], np.char.decode(receptors['trb_vgene']) ],
        axis=0
      )
      receptors_dict['trb_cdr3'] = np.concatenate(
        [ receptors_dict['trb_cdr3'], np.char.decode(receptors['trb_cdr3']) ],
        axis=0
      )
      receptors_dict['trb_jgene'] = np.concatenate(
        [ receptors_dict['trb_jgene'], np.char.decode(receptors['trb_jgene']) ],
        axis=0
      )
      weights = 0.0
      for tag in tags:
        weights += receptors['frequency_'+tag]
      weights /= num_categories
      receptors_dict['weights'] = np.concatenate(
        [ receptors_dict['weights'], weights ],
        axis=0
      )
      for j, tag in enumerate(tags):
        receptors_dict['labels_'+tag] = np.concatenate(
          [ receptors_dict['labels_'+tag], receptors['frequency_'+tag]/(num_categories*weights) ],
          axis=0
        )
      receptors_dict['split'] = np.concatenate(
        [ receptors_dict['split'], [ split for i in range(receptors.size) ] ],
        axis=0
      )

  # Build dataframe
  #
  data = pd.DataFrame(receptors_dict)

  # Handcrafted features
  #
  data_processed = handcrafted_features(data, tags)

  # Split samples
  #
  outputs_list = []
  for split in splits:

    conditions = data_processed['split'] == split
    data_split = data_processed[conditions]
    data_split.drop('split', axis=1)

    features_split = data_split.drop(
      [ 'weights', 'split' ]+[ 'labels_'+tag for tag in tags ],
      axis=1
    )
    labels_split = data_split[
      [ 'labels_'+tag for tag in tags ]
    ]
    weights_split = data_split['weights']

    xs_split = features_split.to_numpy()
    ys_split = labels_split.to_numpy()
    ws_split = weights_split.to_numpy()

    # Permute samples
    #
    if permute:
      indices = np.arange(xs_split.shape[0])
      np.random.shuffle(indices)
      xs_split = xs_split[indices]

    outputs_list.append(xs_split)
    outputs_list.append(ys_split)
    outputs_list.append(ws_split)

  return outputs_list

