## Antigen Classification Model

To demonstrate *dynamic kernel matching* (DKM) can be used to classify sequences, we modify a multinomial regression model with DKM and fit it to the antigen classification dataset [(link)](https://github.com/jostmey/dkm/tree/master/antigen-classification-problem/dataset). Applying DKM on sequences requires that we implement a sequence alignment to match features to weights. See `alignment_score.py` for our implementation of the [Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm) sequence alignment algorithm in TensorFlow. A Keras wrapper is provided in `Alignment.py` with a pure Keras implementation available in `simplified-codebase/`.

![alt text](../../artwork/antigen-classification-model.png "Antigen classification model")

## Running the Model

The script below fits a DKM augmented multinomial regression model to the antigen classification dataset. The initialization of the model can take 15 minutes or more before gradient optimization begins.

```
mkdir bin
python3 train_val.py --database ../dataset/database.h5 --table_train Receptor-PMHC-Complex/train --table_val Receptor-PMHC-Complex/validate --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --output bin/model
```

As the script runs, columns of numbers are printed on the screen. The five columns of numbers report:
1. The gradient optimization step,
2. the cross-entropy loss over the training cohort,
3. the classification accuracy over the training cohort,
4. the cross-entropy loss over the validation cohort,
5. and the classification accuracy over the validation cohort.

After fitting the model, we can evaluate its performance on the test cohort.

```
python3 test.py --database ../dataset/database.h5 --table Receptor-PMHC-Complex/test --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder --cutoff 0.783349 --input bin/model --output bin/model
```

The script reports the cross-entropy loss and the classification accuracy over the test cohort. We achieved a classification accuracy of 70.5%. Given that we balance over six possible outcomes, the baseline accuracy achievable by chance is 1/6, or equivalent to guessing the outcome of a six-sided die toss.

## Confidence Cutoffs

To provide predictions relevant to clinical decision making, we use a simple approach for capturing only those samples classified with high confidence, separating these samples from indeterminate cases that require additional observation and diagnostic testing. By providing an indeterminate diagnosis on uncertain cases, only the samples that can be diagnosed with a high degree of confidence receive a diagnosis, resulting in a higher classification accuracy. When conducting a blindfolded study, the labels for uncaptured samples must remain blindfolded. The classification accuracy is calculated only from the unblindfolded samples captured by the cutoffs.

To begin, we run every sample through the statistical classifier to generate a prediction. Each prediction represents a probability distribution over outcomes, allowing us to calculate the entropy associated with each prediction. Let `H_j` represent the entropy from the prediction for sample j. We define `H_cutoff` as the cutoff for capturing samples. If `H_j ≤ H_cutoff` the sample is captured because the confidence is high. Otherwise, the sample is not captured by the cutoff because the confidence is low. We start with a value for `H_cutoff` large enough to ensure all the samples are initially captured and decrease `H_cutoff` in increments of 0.01 until we find that the accuracy over captured samples is ≥95% on the validation cohort. We then apply the cutoff to capture samples on the blindfolded test cohort and compute the accuracy.

We provide a script to find `H_cutoff`.

```
python3 cutoff_finder.py --predictions_val bin/model --output cutoff_finder_results.csv
```

Examine `cutoff_finder_results.csv` and find the value for `H_cutoff` associated with at least a 95% classification accuracy on the validation cohort. This is our cutoff. We are ready to capture samples from the test cohort. Suppose `H_cutoff` is 0.783.

```
python3 cutoff_test.py --predictions_test bin/model --cutoff 0.783 --output cutoff_test_results.csv
```

Examine `cutoff_test_results.csv` for the results. **We achieve a classification accuracy of 97% capturing 44% of samples**.
