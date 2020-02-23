## Antigen Classification Problem

To predict T-cell receptor interaction with disease antigens, we need an approach that can classify sequences, which are one kind of non-conforming data. We present *dynamic kernel matching* (DKM) as an approach to modify established statistical classifiers with the computational machinery necessary to handle non-conforming features.

## Running the model

```
mkdir bin
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --output bin/model > train_val.out
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --permute True --output bin/model_permute > train_val_permute.out 
```
