## Antigen Classification Problem

To illustrate *dynamic kernel matching* (DKM) can be used to classify sequences, we modify a multinomial regression model with DKM and fit it to the antigen classification dataset [(link)](https://github.com/jostmey/dkm/tree/master/antigen-classification-problem/dataset). Because the number of features in each sequence is irregular, with some sequences being longer than others, the number of features does not match the number of weights. Given a sequence, the challenge is to determine the appropriate permutation of features with weights, allowing us to run the features through the statistical classifier to generate a prediction. Given the immense number of possible permutations, the problem appears computationally complex but can be solved in polynomial time using a sequence alignment algorithm, like the Needleman-Wunsch algorithm [(link)](https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm). The code is located in `alignment_score.py` and is implemented in TensorFlow.

The DKM module for handling sequences can be found in `Alignment.py` and is implemented in Keras. The DKM module for classifying sequences is analogous to the pooling operation of a convolutional neural network. The sequence alignment algorithm identifies the permutation of features that will elicit the maximum response, just as the pooling operation identifies the image patch that elicits the maximum response.

## Running the model

The scripts below fits a DKM augmented multinomial regression model to the antigen classification dataset. The scripts can take upwards of 15 minutes initializing the model before the gradient optimization based fitting procedure begins.

```
mkdir bin
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --output bin/model
```

As the script runs, columns of numbers will be printed in the terminal. The five columns of numbers report:
1. The gradient optimization step,
2. the cross-entropy loss over the training cohort,
3. the accuracy over the training cohort (picking the correct of six outcomes)
4. the cross-entropy loss over the validation cohort,
5. and the accuracy over the validation cohort, picking the correct of six outcomes.

After fitting the model, we can evaluate its performance on the test cohort.

```
python3 test.py BLAH BLAH
```



```
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --permute True --output bin/model_permute > train_val_permute.out 
```

