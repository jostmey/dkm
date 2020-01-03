#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-05
# Purpose: Insert samples into database
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import csv
import dataplumbing as dp
import random

##########################################################################################
# Settings
##########################################################################################

directory = 'downloads/v2/'
path_overview = 'downloads/SampleOverview_08-19-2017_12-59-19_PM.tsv'
path_db = 'database.h5'

##########################################################################################
# Insert samples
##########################################################################################

samples_I = {}
samples_II = {}

with open(path_overview, 'r') as stream:
  reader = csv.DictReader(stream, delimiter='\t')
  for row in reader:
    sample_name = row['sample_name']
    sample_tags = row['sample_tags'].split(',')
    cmv_positive = None
    age = None
    cohort = None
    for sample_tag in sample_tags:
      if 'Years' in sample_tag:
        age = float(sample_tag.split(' ')[0])
      if 'Cytomegalovirus -' in sample_tag:
        cmv_positive = False
      elif 'Cytomegalovirus +' in sample_tag:
        cmv_positive = True
      elif 'Cohort 01' in sample_tag:
        cohort = 1
      elif 'Cohort 02' in sample_tag:
        cohort = 2
    if cmv_positive is not None and age is not None:
      if cohort == 1:
        samples_I[sample_name] = {
          'diagnosis': cmv_positive,
          'age': age
        }
      else:
        samples_II[sample_name] = {
          'diagnosis': cmv_positive,
          'age': age
        }

for sample in sorted(samples_I.keys()):
  receptors = dp.load_receptors(directory+sample+'.tsv')
  receptors = dp.normalize_receptors(receptors)
  dp.insert_receptors(path_db, 'Cohort_I/features/'+sample, receptors)
dp.insert_samples(path_db, 'Cohort_I/samples', samples_I)

keys = list(samples_I.keys())
random.shuffle(keys)
samples_I_train = { key: samples_I[key] for key in keys[:-120] } 
samples_I_val = { key: samples_I[key] for key in keys[-120:] } 
dp.insert_samples(path_db, 'Cohort_I/samples_train', samples_I_train)
dp.insert_samples(path_db, 'Cohort_I/samples_validate', samples_I_val)

for sample in sorted(samples_II.keys()):
  receptors = dp.load_receptors(directory+sample+'.tsv')
  receptors = dp.normalize_receptors(receptors)
  dp.insert_receptors(path_db, 'Cohort_II/features/'+sample, receptors)
dp.insert_samples(path_db, 'Cohort_II/samples', samples_II)

