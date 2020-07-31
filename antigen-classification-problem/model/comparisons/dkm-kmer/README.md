## Comparison of DKM to DKM using k-mers

This folder contains a script for using DKM with k-mers on the antigen classification problem. Rather than classify each TCR as a sequence of residues, residues are joined together into overlapping subsequences of length k to form a sequence of k-mers, and the sequence of k-mers is classified. The results can then be compared to DKM without k-mers. To use this code, simply move the python script into the parent directory, overwriting the original DKM model.

```
mv KMer.py ../../
mv model.py ../../
```

Then follow the README in parent folder. On the test cohort, we achieve a classification accuracy of 71.56%.
