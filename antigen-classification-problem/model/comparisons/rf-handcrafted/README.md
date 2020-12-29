## Comparison of DKM to a Random Forest Model using Handcrafted Features

This folder contains a script for using a random forest model on the antigen classification problem using handcrafted features. The code for generating the features from each TCR sequence come from https://github.com/bittremieux/TCR-Classifier, which is applied twice, once for the alpha chain and again for the beta chain.

The results can then be compared to DKM. To use this code, simply move the python scripts into the parent directory, overwriting the DKM model.

```
mv dataset.py ../../
mv train_test.py ../../
```

To run the model, use the following command.

```
python3 train_test.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_test Receptor-PMHC-Complex/test --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --output bin/model.p
```

On the test cohort, we achieve a classification accuracy of 67.35%. We considered using the BorutaPy package for for feature selection, but the package does not support the multinomial labels and sample weighting for balancing data required by the antigen classification problem. To determine if BorutaPy would improve the classification accuracy, we applied it to imbalanced data restricting the predictions to two pMHCs, and found that it did not improve performance (see `train_test.py`).

