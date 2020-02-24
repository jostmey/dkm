## Repertoire Classification Problem

To illustrate *dynamic kernel matching* (DKM) can be used to classify sets of sequence, we modify a logistic regression model with DKM and fit it to the repertoire classification dataset [(link)](https://github.com/jostmey/dkm/tree/master/repertoire-classification-problem/dataset). To handle sets of sequences, we apply DKM twice, first to handle the sequences and then again to handle the set. We handle sequences using the code from the antigen classification problem, and we handle the sets by taking the maximum scoring sequence.

![alt text](../../artwork/repertoire-classification-model.png "Repertoire classification model")

## Model fitting

Two special aspects for fitting the model are (i) gradient aggregation and (ii) many local minima.

###### Gradient Aggregation
Each sequenced immune repertoire contains an average of over 100,000 receptors with almost 10,000,000 features per sample. Consequently, only one or a few immune repertoires can be fit into GPU memory at any given time. To fit the model to the training data, we rely on gradient aggregation, which is where the gradients with respect to the cross-entropy loss function are computed one sample at a time in a serial fashion. Afterwards, the gradients are added together and the weights are updated. Gradient aggregation achieves the same computation as if the gradients are computed with respect to the cross-entropy loss over all the samples simultaneously, as is commonly done.


###### Many Local Minima
After running 128 steps of gradient optimization, we observe the fit strongly depends on initial weights. If we refit the model using different values for the initial values for the weights then we get a substantially different result. This indicates the cross-entropy landscape contains many local minima. To attempt to find a global best fit to the training data, we refit the model 128 times, each time starting from different initial values for the weights. We use the weights from the best fit, as measure over the training cohort (the validation cohort have not been used yet).

![alt text](../../artwork/many-fits.png "Best fit to training data")

## Running the model(s)

Running the script `train_val.py` fits 16 copies of the model to the training cohort. We run the script 8 times, resulting in a total of 128 copies of the model. We find the best fit to the training cohort out of all 128 copies. The initialization of the model can take 15 minutes or more before the gradient optimization begins. The script assumes an array of 8 GPUs with the same memory as a P100 16GB GPU.

```
mkdir bin
python3 train_val.py --gpu 0 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_1 > bin/train_val_1.out
python3 train_val.py --gpu 1 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_2 > bin/train_val_2.out
python3 train_val.py --gpu 2 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_3 > bin/train_val_3.out
python3 train_val.py --gpu 3 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_4 > bin/train_val_4.out
python3 train_val.py --gpu 4 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_5 > bin/train_val_5.out
python3 train_val.py --gpu 5 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_6 > bin/train_val_6.out
python3 train_val.py --gpu 6 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_7 > bin/train_val_7.out
python3 train_val.py --gpu 7 --database ../dataset/database.h5 --cohort_train Cohort_I --split_train samples_train --cohort_val Cohort_I --split_val samples_validate --output bin/model_8 > bin/train_val_8.out
```

A seperate script identifies the copy that best fits the training cohort, which also reports how that copy does on the validation cohort.

```
python3 find_bestfit.py
```

The seven columns of numbers report:
1. The gradient optimization step,
2. run with best fitting copy (1 to 8),
3. index of the best fitting copy on sed run (0 to 15),
4. the cross-entropy loss over the training cohort for the best fitting copy,
5. the classification accuracy over training cohort for sed copy,
6. the cross-entropy loss over the validation cohort for sed copy,
7. and the classification accuracy over validation cohort for sed copy,

Once the best fitting copy has been identified, we can evalue the model on the test cohort. Suppose the best fitting run is 1 and the best fitting copy on that run is 14.

```
python3 test.py --gpu 0 --database ../dataset/database.h5 --cohort_test Cohort_II --split_test samples --input bin/model_1 --index 14 --output bin/model_1
```

The script reports the cross-entropy loss and the classification accuracy over the test cohort. We achieved a classification accuracy of 67.6%. Given that we balance over the two possible outcomes, the baseline accuracy achievable by chance is 1/2, or equivalent to tossing a coin. While these results are poor, we planned to use confidence cutoffs from the onset of this study, allowing us to achieve significantly better results with a caveat.

## Confidence Cutoffs

To provide predictions relevant to clinical decision making, we use a simple approach for capturing only those samples classified with high confidence, separating these samples from indeterminate cases that require additional observation and diagnostic testing. By providing an indeterminate diagnosis on uncertain cases, only the patients that can be diagnosed with a high degree of confidence receive a diagnosis, resulting in a higher classification accuracy. The classification accuracy is calculated only from the unblindfolded samples captured by the cutoffs.




