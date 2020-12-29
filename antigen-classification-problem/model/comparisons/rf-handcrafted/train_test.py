#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Train and test model classifier for T-cell receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
from dataset import *
import numpy as np
from sklearn import ensemble, metrics, pipeline
#from boruta import BorutaPy
import pickle

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--database', help='Path to the database', type=str, required=True)
parser.add_argument('--table_train', help='Path to the training table', type=str, required=True)
parser.add_argument('--table_test', help='Path to the test table', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--permute', help='Randomly permute the relationship between features and labels', type=bool, default=False)
args = parser.parse_args()

##########################################################################################
# Load datasets
##########################################################################################

# Settings
#
num_tags = len(args.tags)

# Load the samples
#
xs_train, ys_train, fs_train, xs_test, ys_test, fs_test = \
  load_datasets(
    args.database, [ args.table_train, args.table_test ],
    args.tags, permute=args.permute
  )

##########################################################################################
# Model
##########################################################################################

# Converts soft labels to hard labels
#
def label_float2int(ys, num_classes):
  ys_index = np.argmax(ys, axis=1)
  ys_onehot = np.squeeze(np.eye(num_classes)[ys_index.reshape(-1)])
  ys_hard = ys_onehot.astype(np.int64)
  return ys_hard

# Instantiate model with feature selection (no sample weighting)
#
#classifier = ensemble.RandomForestClassifier(n_estimators=200)
#predictor = pipeline.Pipeline(
#  [
#    ('feature_selection', BorutaPy(ensemble.ExtraTreesClassifier(n_jobs=-1), n_estimators='auto')),
#    ('classification', classifier)
#  ]
#)

# Instantiate model
#
predictor = ensemble.RandomForestClassifier(n_estimators=200)

# Fit model (no sample weighting)
#
#predictor.fit(xs_train, label_float2int(ys_train, num_tags))

# Fit model
#
predictor.fit(xs_train, label_float2int(ys_train, num_tags), sample_weight=fs_train)

# Generate predictions
#
ps_train = predictor.predict(xs_train)
ps_test = predictor.predict(xs_test)

# Metrics
#
a_train = metrics.accuracy_score(
  label_float2int(ys_train, num_tags),
  ps_train,
  sample_weight=fs_train
)
a_test = metrics.accuracy_score(
  label_float2int(ys_test, num_tags),
  ps_test,
  sample_weight=fs_test
)

##########################################################################################
# Report and save model
##########################################################################################

# Print report
#
print(
  'Multinomial-Accuracy (Train):', a_train,
  'Multinomial-Accuracy (Test):', a_test
)

# Save model
#
with open(args.output, 'wb') as stream:
  pickle.dump(predictor, stream)
