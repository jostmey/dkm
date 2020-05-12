##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-08-15
# Purpose: Define the model
##########################################################################################

from tensorflow.keras import *
from tensorflow.keras.layers import *
from Length import *
from BatchExpand import *
from Abundance import *
from Alignment import *
from NormalizeInitialization import *
from Aggregate import *
from FullFlatten import *

def generate_model(input_shape_cdr3, num_outputs, filter_size):

  features_cdr3 = Input(shape=input_shape_cdr3)
  features_quantity = Input(shape=[])
  feature_age = Input(batch_shape=[1])
  weight = Input(batch_shape=[1])
  level = Input(batch_shape=[1])

  features_mask = Masking(mask_value=0.0)(features_cdr3)
  features_length = Length()(features_mask)
  features_abundance = Abundance()(features_quantity)
  features_age = BatchExpand()([ feature_age, features_abundance ])
  weights_instance = Multiply()([weight, features_quantity])

  logits_cdr3 = Alignment(num_outputs, filter_size, penalties_feature=0.0, penalties_filter=-1.0E16, length_normalize=False)(features_mask)
  logits_cdr3_norm = NormalizeInitializationByAggregation(1, epsilon=1.0E-5)([ logits_cdr3, weights_instance, level ])

  feature_length_norm = NormalizeInitializationByAggregation(0, epsilon=1.0E-5)([ features_length, weights_instance, level ])
  logits_length = Dense(num_outputs)(feature_length_norm)
  logits_length_norm = NormalizeInitializationByAggregation(1, epsilon=1.0E-5)([ logits_length, weights_instance, level ])

  features_abundance_norm = NormalizeInitializationByAggregation(0, epsilon=1.0E-5)([ features_abundance, weights_instance, level ])
  logits_abundance = Dense(num_outputs)(features_abundance_norm)
  logits_abundance_norm = NormalizeInitializationByAggregation(1, epsilon=1.0E-5)([ logits_abundance, weights_instance, level ])

  features_age_norm = NormalizeInitializationByAggregation(0, epsilon=1.0E-5)([ features_age, weights_instance, level ])
  logits_age = Dense(num_outputs)(features_age_norm)
  logits_age_norm = NormalizeInitializationByAggregation(1, epsilon=1.0E-5)([ logits_age, weights_instance, level ])

  logits = Add()(
    [ logits_cdr3_norm, logits_length_norm, logits_abundance_norm, logits_age_norm ]
  )
  logits_aggregate = Aggregate()(logits)
  logits_aggregate_norm = NormalizeInitializationByAggregation(2, epsilon=1.0E-5)([ logits_aggregate, weight, level ])

  logits_flat = FullFlatten()(logits_aggregate_norm)

  model = Model(
    inputs=[ features_cdr3, features_quantity, feature_age, weight, level ],
    outputs=logits_flat
  )

  return model
