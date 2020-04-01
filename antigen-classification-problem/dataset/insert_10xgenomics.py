#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-07-11
# Purpose: Insert samples into database
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import dataplumbing as dp

##########################################################################################
# Settings
##########################################################################################

base_dir = 'downloads'
path_db = 'database.h5'

##########################################################################################
# Insert samples
##########################################################################################

pmhcs = dp.list_pmhc_types()

pmhcs_subject = []
receptors_subject = []
for pmhc in pmhcs:
  receptors = dp.load_receptors(base_dir, pmhc)
  print('CATEGORY:', pmhc, 'NUMBER:', len(receptors))
  receptors = dp.normalize_sample(receptors)
  pmhcs_subject.append(pmhc)
  receptors_subject.append(receptors)
receptors = dp.collapse_samples(receptors_subject, pmhcs_subject)

receptors_train, receptors_val, receptors_test = dp.split_dataset(receptors, [0.6, 0.2, 0.2])
dp.insert_receptors(path_db, 'Receptor-PMHC-Complex/train', receptors_train)
dp.insert_receptors(path_db, 'Receptor-PMHC-Complex/validate', receptors_val)
dp.insert_receptors(path_db, 'Receptor-PMHC-Complex/test', receptors_test)
dp.insert_receptors(path_db, 'Receptor-PMHC-Complex/all', receptors)
