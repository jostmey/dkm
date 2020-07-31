##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-15
# Purpose: Define the model
##########################################################################################

from tensorflow.keras import *
from tensorflow.keras.layers import *
from Length import *
from KMer import *
from Alignment import *
from NormalizeInitialization import *

def generate_model(
    input_shape_tra_cdr3, input_shape_tra_vgene, input_shape_tra_jgene,
    input_shape_trb_cdr3, input_shape_trb_vgene, input_shape_trb_jgene,
    num_outputs, num_steps
  ):

  kmer_size = 5

  features_tra_cdr3 = Input(shape=input_shape_tra_cdr3)
  features_tra_vgene = Input(shape=input_shape_tra_vgene)
  features_tra_jgene = Input(shape=input_shape_tra_jgene)
  features_trb_cdr3 = Input(shape=input_shape_trb_cdr3)
  features_trb_vgene = Input(shape=input_shape_trb_vgene)
  features_trb_jgene = Input(shape=input_shape_trb_jgene)
  weights = Input(shape=[])

  features_tra_mask = Masking(mask_value=0.0)(features_tra_cdr3)
  features_tra_length = Length()(features_tra_mask)

  features_tra_kmer = KMer(kmer_size)(features_tra_mask)
  logits_tra_kmer = Alignment(num_outputs, num_steps-kmer_size+1, penalties_feature=-1.0E16, penalties_filter=0.0, length_normalize=True)(features_tra_kmer)
  logits_tra_kmer_norm = NormalizeInitialization(epsilon=0.0)([ logits_tra_kmer, weights ])

  logits_tra_length = Dense(num_outputs)(features_tra_length)
  logits_tra_length_norm = NormalizeInitialization(epsilon=0.0)([ logits_tra_length, weights ])

  logits_tra_vgene = Dense(num_outputs)(features_tra_vgene)
  logits_tra_vgene_norm = NormalizeInitialization(epsilon=0.0)([ logits_tra_vgene, weights ])

  logits_tra_jgene = Dense(num_outputs)(features_tra_jgene)
  logits_tra_jgene_norm = NormalizeInitialization(epsilon=0.0)([ logits_tra_jgene, weights ])

  features_trb_mask = Masking(mask_value=0.0)(features_trb_cdr3)
  features_trb_length = Length()(features_trb_mask)

  features_trb_kmer = KMer(kmer_size)(features_trb_mask)
  logits_trb_kmer = Alignment(num_outputs, num_steps-kmer_size+1, penalties_feature=-1.0E16, penalties_filter=0.0, length_normalize=True)(features_trb_kmer)
  logits_trb_kmer_norm = NormalizeInitialization(epsilon=0.0)([ logits_trb_kmer, weights ])

  logits_trb_length = Dense(num_outputs)(features_trb_length)
  logits_trb_length_norm = NormalizeInitialization(epsilon=0.0)([ logits_trb_length, weights ])

  logits_trb_vgene = Dense(num_outputs)(features_trb_vgene)
  logits_trb_vgene_norm = NormalizeInitialization(epsilon=0.0)([ logits_trb_vgene, weights ])

  logits_trb_jgene = Dense(num_outputs)(features_trb_jgene)
  logits_trb_jgene_norm = NormalizeInitialization(epsilon=0.0)([ logits_trb_jgene, weights ])

  logits = Add()(
    [
      logits_tra_kmer_norm, logits_tra_length_norm, logits_tra_vgene_norm, logits_tra_jgene_norm,
      logits_trb_kmer_norm, logits_trb_length_norm, logits_trb_vgene_norm, logits_trb_jgene_norm
    ]
  )
  logits_norm = NormalizeInitialization(epsilon=0.0)([ logits, weights ])

  model = Model(
    inputs=[
      features_tra_cdr3, features_tra_vgene, features_tra_jgene,
      features_trb_cdr3, features_trb_vgene, features_trb_jgene,
      weights 
    ],
    outputs=logits_norm
  )

  return model
