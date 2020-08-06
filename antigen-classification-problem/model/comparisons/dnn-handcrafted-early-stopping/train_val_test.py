#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2019-01-28
# Purpose: Train and validate model classifier for T-cell receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import os
from dataset import *
import numpy as np
from tensorflow.keras import *
from tensorflow.keras.layers import *
from sklearn import metrics

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--database', help='Path to the database', type=str, required=True)
parser.add_argument('--table_train', help='Path to the training table', type=str, required=True)
parser.add_argument('--table_val', help='Path to the validation table', type=str, required=True)
parser.add_argument('--table_test', help='Path to the test table', type=str, required=True)
parser.add_argument('--tags', help='Tag name of the categories', type=str, nargs='+', required=True)
parser.add_argument('--output', help='Output basename', type=str, required=True)
parser.add_argument('--permute', help='Randomly permute the relationship between features and labels', type=bool, default=False)
parser.add_argument('--gpu', help='GPU ID', type=int, default=0)
args = parser.parse_args()

##########################################################################################
# Environment
##########################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

##########################################################################################
# Load datasets
##########################################################################################

# Load the samples
#
xs_train, ys_train, fs_train, xs_val, ys_val, fs_val, xs_test, ys_test, fs_test = \
  load_datasets(
    args.database, [ args.table_train, args.table_val, args.table_test ],
    args.tags, permute=args.permute
  )

##########################################################################################
# Model
##########################################################################################

num_hiddens = 128
num_categories = ys_train.shape[1]

model = Sequential(
  [
    Dense(num_hiddens),
    BatchNormalization(momentum=0.5),
    Activation('relu'),
    Dense(num_categories),
    Dropout(0.5),
    BatchNormalization(momentum=0.5),
    Activation('softmax')
  ]
)

##########################################################################################
# Optimization and Compilation
##########################################################################################

learning_rate = 0.001

optimizer = optimizers.Adam(lr=learning_rate)
model.compile(
  optimizer=optimizer,
  loss='categorical_crossentropy',
  metrics=[ 'categorical_accuracy' ]
)

##########################################################################################
# Fit and Evaluate
##########################################################################################

batch_size = ys_train.shape[0]
num_train = ys_train.shape[0]
num_val = ys_val.shape[0]
num_test = ys_test.shape[0]

def balanced_sampling(xs, ys, ws, batch_size):
  rs = np.arange(xs.shape[0])
  ws_ = ws/np.sum(ws)
  while True:
    js = np.random.choice(rs, size=batch_size, p=ws_)  # Balanced sampling from the categories
    yield ( xs[js], ys[js] )

early_stopping = callbacks.EarlyStopping(patience=128, restore_best_weights=True)
model.fit_generator(
  generator=balanced_sampling(
    xs_train, ys_train, fs_train, batch_size
  ),
  steps_per_epoch=int(np.ceil(num_train/batch_size)), epochs=1024,
  validation_data=balanced_sampling(
    xs_val, ys_val, fs_val, batch_size
  ),
  validation_steps=int(np.ceil(num_val/batch_size)),
  callbacks=[ early_stopping ],
  use_multiprocessing=True
)

c_test, a_test = model.evaluate_generator(
  generator=balanced_sampling(
    xs_test, ys_test, fs_test, batch_size
  ),
  steps=int(np.ceil(num_test/batch_size)),
  use_multiprocessing=True
)
print(
  'Cost (Test):', '%4.3f'%(c_test/np.log(2.0)),
  'Accuracy (Test):', '%4.3f'%(100.0*a_test)
)

##########################################################################################
# Save Results
##########################################################################################

model.save_weights(args.output)

ps_train = model.predict(xs_train)
np.save(args.output+'_ps_train.npy', ps_train)
np.save(args.output+'_ys_train.npy', ys_train)
np.save(args.output+'_ws_train.npy', fs_train)

ps_val = model.predict(xs_val)
np.save(args.output+'_ps_val.npy', ps_val)
np.save(args.output+'_ys_val.npy', ys_val)
np.save(args.output+'_ws_val.npy', fs_val)

ps_test = model.predict(xs_val)
np.save(args.output+'_ps_test.npy', ps_test)
np.save(args.output+'_ys_test.npy', ys_test)
np.save(args.output+'_ws_test.npy', fs_test)
