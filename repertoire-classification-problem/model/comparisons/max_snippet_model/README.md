## Comparison of DKM to another variant of DKM

This folder contains a script for using the Max Snippet Model on the repertoire classification
problem. To use this code, simply move the python script into the parent directory, overwriting
the DKM model.

```
mv BatchExpand.py ../../
mv GlobalPool.py ../../
mv MaskCopy.py ../../
mv model.py ../../
mv train_val.py ../../
```

Then follow the README in parent folder. On the test cohort, we achieve a classification
accuracy of 68.3%.
