## Comparison of DKM to another variant of DKM

This folder contains a script for using an alternative variant of DKM on the repertoire classification
problem. The model has 32 instead of 8 steps of weights. To use this code, simply move these python
scripts into the parent directory, overwriting the existing DKM model.

```
mv model.py ../../
mv train_val.py ../../
```

Then follow the README in parent folder. On the test cohort, we achieve a classification
accuracy of 73.5%.
