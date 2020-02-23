## Antigen Classification Problem

To illustrate *dynamic kernel matching* (DKM) can be used to classify sequences, we modify a multinomial regression model with DKM and fit it to the antigen classification dataset [(link)](https://github.com/jostmey/dkm/tree/master/antigen-classification-problem/dataset). Because the number of features in each sequence is irregular, with some sequences being longer than others, the number of features does not match the number of weights. Given a sequence, the challenge is to determine the appropriate permutation of features with the weights, allowing us to run those features through the statistical classifier to generate a prediction. The problem appears computationally complex but can be solved in polynomial time using a sequence alignment algorithm like the Needleman-Wunsch algorithm [(link)](https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm).

Our DKM model for classifying sequences is alangous to a shallow convolutional neural network with one filter and global pooling for classifying images. The sequence alignment algorithm identifies the features that the model maximally responds to, just as the global pooling layer identifies the image patch the convolutional neural network maximally responds to.

## Running the model

The scripts below refit the DKM augmented multinomial regression model. The model is written for TensorFlow v1.14 and can efficiently leverage GPU cards. The scripts can take upwards of 15 minutes initializing the model before the gradient optimization based fitting procedure begins.

```
mkdir bin
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --output bin/model > train_val.out
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --permute True --output bin/model_permute > train_val_permute.out 
```

