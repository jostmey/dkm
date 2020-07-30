## Comparison of DKM to a Gated-Recurrent Network (LSTM) Network

This folder contains a script for using a GRU network on the antigen classification problem.
The results can then be compared to DKM. To use this code, simply move the python script into
the parent directory, overwriting the DKM model.

```
mv model.py ../../
```

Then follow the README in parent folder. On the test cohort, we achieve a classification
accuracy of 65.48%.
