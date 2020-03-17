#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-07-11
# Purpose: Split dataset into a training, validation, and test set.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import csv
import numpy as np
import os
import h5py

##########################################################################################
# Load data
##########################################################################################

def list_pmhc_types():
  return [
    'A0101_VTEHDTLLY_IE-1_CMV_binder', 'A0201_KTWGQYWQV_gp100_Cancer_binder', 'A0201_ELAGIGILTV_MART-1_Cancer_binder',
    'A0201_CLLWSFQTSA_Tyrosinase_Cancer_binder', 'A0201_IMDQVPFSV_gp100_Cancer_binder', 'A0201_SLLMWITQV_NY-ESO-1_Cancer_binder',
    'A0201_KVAELVHFL_MAGE-A3_Cancer_binder', 'A0201_KVLEYVIKV_MAGE-A1_Cancer_binder', 'A0201_CLLGTYTQDV_Kanamycin-B-dioxygenase_binder',
    'A0201_LLDFVRFMGV_EBNA-3B_EBV_binder', 'A0201_LLMGTLGIVC_HPV-16E7_82-91_binder', 'A0201_CLGGLLTMV_LMP-2A_EBV_binder',
    'A0201_YLLEMLWRL_LMP1_EBV_binder', 'A0201_FLYALALLL_LMP2A_EBV_binder', 'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
    'A0201_GLCTLVAML_BMLF1_EBV_binder', 'A0201_NLVPMVATV_pp65_CMV_binder', 'A0201_ILKEPVHGV_RT_HIV_binder',
    'A0201_FLASKIGRLV_Ca2-indepen-Plip-A2_binder', 'A2402_CYTWNQMNL_WT1-(235-243)236M_Y_binder', 'A0201_RTLNAWVKV_Gag-protein_HIV_binder',
    'A0201_KLQCVDLHV_PSA146-154_binder', 'A0201_LLFGYPVYV_HTLV-1_binder', 'A0201_SLFNTVATL_Gag-protein_HIV_binder',
    'A0201_SLYNTVATLY_Gag-protein_HIV_binder', 'A0201_SLFNTVATLY_Gag-protein_HIV_binder', 'A0201_RMFPNAPYL_WT-1_binder',
    'A0201_YLNDHLEPWI_BCL-X_Cancer_binder', 'A0201_MLDLQPETT_16E7_HPV_binder', 'A0301_KLGGALQAK_IE-1_CMV_binder',
    'A0301_RLRAEAQVK_EMNA-3A_EBV_binder', 'A0301_RIAAWMATY_BCL-2L1_Cancer_binder', 'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
    'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder', 'B3501_IPSINVHHY_pp65_CMV_binder', 'A2402_AYAQKIFKI_IE-1_CMV_binder',
    'A2402_QYDPVAALF_pp65_CMV_binder', 'B0702_QPRAPIRPI_EBNA-6_EBV_binder', 'B0702_TPRVTGGGAM_pp65_CMV_binder',
    'B0702_RPPIFIRRL_EBNA-3A_EBV_binder', 'B0702_RPHERNGFTVL_pp65_CMV_binder', 'B0801_RAKFKQLL_BZLF1_EBV_binder',
    'B0801_ELRRKMMYM_IE-1_CMV_binder', 'B0801_FLRGRAYGL_EBNA-3A_EBV_binder', 'A0101_SLEGGGLGY_NC_binder',
    'A0101_STEGGGLAY_NC_binder', 'A0201_ALIAPVHAV_NC_binder', 'A2402_AYSSAGASI_NC_binder',
    'B0702_GPAESAAGL_NC_binder', 'NR(B0801)_AAKGRGAAL_NC_binder'
  ]

def load_receptors(base_dir, pmhc):
  receptors = {}
  for subject in ['1', '2', '3', '4']:
    barcodes = {}
    path_csv = base_dir+'/'+'vdj_v1_hs_aggregated_donor'+subject+'_all_contig_annotations.csv'
    with open(path_csv, 'r') as stream:
      reader = csv.DictReader(stream, delimiter=',')
      for row in reader:
        barcode = row['barcode']
        if barcode not in barcodes:
          barcodes[barcode] = []
        cdr3 = row['cdr3']
        vgene = row['v_gene']
        jgene = row['j_gene']
        if 'None' not in cdr3 and '*' not in cdr3 and 'None' not in vgene and 'None' not in jgene:
          barcodes[barcode].append(
            {
              'chain': row['chain'],
              'cdr3': cdr3,
              'vgene': vgene,
              'jgene': jgene,
              'full': True if 'TRUE' in row['full_length'] else False
            }
          )
    path_csv = base_dir+'/'+'vdj_v1_hs_aggregated_donor'+subject+'_binarized_matrix.csv'
    with open(path_csv, 'r') as stream:
      reader = csv.DictReader(stream, delimiter=',')
      for row in reader:
        if 'True' in row[pmhc]:
          pairings = []
          barcode = row['barcode']
          for sequence_tra in barcodes[barcode]:
            if 'TRA' in sequence_tra['chain']:
              for sequence_trb in barcodes[barcode]:
                if 'TRB' in sequence_trb['chain']:
                  pairings.append(
                    sequence_tra['vgene']+':'+sequence_tra['cdr3']+':'+sequence_tra['jgene']+':'+ \
                    sequence_trb['vgene']+':'+sequence_trb['cdr3']+':'+sequence_trb['jgene']
                  )
          for pairing in pairings:
            if pairing not in receptors:
              receptors[pairing] = 1.0
            else:
              receptors[pairing] += 1.0
  return receptors

def normalize_sample(receptors):
  total_count = np.float64(0.0)
  for quantity in receptors.values():
    total_count += quantity
  for receptor in receptors.keys():
    receptors[receptor] /= total_count
  return receptors

def collapse_samples(samples, labels):
  receptors_collapse = {}
  for i, (receptors, label) in enumerate(zip(samples, labels)):
    for receptor, quantity in receptors.items():
      if receptor not in receptors_collapse:
        receptors_collapse[receptor] = {}
      if label not in receptors_collapse[receptor]:
        receptors_collapse[receptor][label] = quantity
      else:
        receptors_collapse[receptor][label] += quantity
        print('WARNING: Duplicate label for the same receptor')
  return receptors_collapse

def split_dataset(receptors, ratios):
  rs = np.array(ratios, dtype=np.float64)
  ss = rs/np.sum(rs)
  cs = np.cumsum(ss)
  ps = np.pad(cs, [1, 0], 'constant', constant_values=0)
  keys = list(receptors.keys())
  np.random.shuffle(keys)
  keys_split = []
  for i in range(len(ratios)):
    j1, j2 = (len(keys)*ps[i:i+2]).astype(int)
    keys_split.append(keys[j1:j2])
  receptors_split = []
  for keys in keys_split:
    receptor_split = {}
    for key in keys:
      receptor_split[key] = receptors[key]
    receptors_split.append(receptor_split)
  return receptors_split

##########################################################################################
# Database interface
##########################################################################################

def insert_receptors(path_db, name, receptors, max_cdr3_length=32):
  labels = set()
  for quantities in receptors.values():
    labels.update(quantities.keys())
  labels = sorted(list(labels))
  dtype_receptor = [
    ('tra_vgene', 'S16'),
    ('tra_cdr3', 'S'+str(max_cdr3_length)),
    ('tra_jgene', 'S16'),
    ('trb_vgene', 'S16'),
    ('trb_cdr3', 'S'+str(max_cdr3_length)),
    ('trb_jgene', 'S16')
  ]+[
    ('frequency_'+label, 'f8') for label in labels
  ]
  rs = np.zeros(len(receptors), dtype=dtype_receptor)
  for i, (receptor, quantities) in enumerate(receptors.items()):
    tra_vgene, tra_cdr3, tra_jgene, trb_vgene, trb_cdr3, trb_jgene = receptor.split(':')
    rs[i]['tra_vgene'] = tra_vgene
    rs[i]['tra_cdr3'] = tra_cdr3
    rs[i]['tra_jgene'] = tra_jgene
    rs[i]['trb_vgene'] = trb_vgene
    rs[i]['trb_cdr3'] = trb_cdr3
    rs[i]['trb_jgene'] = trb_jgene
    for label in quantities.keys():
      rs[i]['frequency_'+label] = quantities[label]
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    rs_db = db.create_dataset(name, (rs.size,), dtype_receptor)
    rs_db[:] = rs
