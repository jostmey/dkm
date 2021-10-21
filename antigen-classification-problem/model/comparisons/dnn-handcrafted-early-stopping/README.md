## Comparison of DKM to a Deep Neural Network using Handcrafted Features

This folder contains a script for using a deep neural network model on the antigen classification problem using handcrafted features. The code for generating the features from each TCR sequence come from https://github.com/bittremieux/TCR-Classifier, which is applied twice, once for the alpha chain and again for the beta chain.

The results can then be compared to DKM. To use this code, simply move the python scripts into the parent directory, overwriting the DKM model.

```
mv model.py ../../
mv train_test.py ../../
```

To run the model, use the following command.

```
python3 train_val_test.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --table_test Receptor-PMHC-Complex/test --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --output bin/model
```

On the test cohort, we achieve a classification accuracy of 67.12%. 
