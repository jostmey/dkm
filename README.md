# When Data that Do Not Conform to Rows and Columns: A Case Study of T-cell Receptor Datasets
###### JARED L OSTMEYER, ASSISTANT PROFESSOR, UT SOUTHWESTERN DEPARTMENT OF POPULATION AND DATA SCIENCES

## Introduction

Statistical classifiers are mathematical models that use example data to find patterns in features that predict a label. Most statistical classifiers assume the features are arranged into rows and columns, like a spreadsheet, but many kinds of data do not conform to this structure. Sequences are one example of a different kind of data, which is why this data is best stored in a text document, not a spreadsheet. To uncover patterns in sequences and other nonconforming features, it is important to consider not only the content of the features but the way those features are structured.

There are many ways non-conforming features can be structured. For example, biological data is often structured as a sequence of symbols. The essential property of a sequence is that both the content and order of symbols provide important information. Sets represent another way data can be structured. A set is like a sequence except the order of the symbols does not encode information. Identifying the underlying structure of the features is important because disparate sources of data with the same underlying structure can be handled with the same methods. For example, a social network is a graph of people and the way people are connected just like a molecule is a graph of atoms and the way those atoms are connected, allowing for the same statistical classifier to handle both cases.

To understand different ways data can be structured, we consider two datasets of T-cell receptors, anticipating these datasets to contain signatures for diagnosing disease. The following two T-cell receptor datasets are examples of non-conforming data.

![alt text](artwork/data.png "Layout of data used in this study")

10x Genomics has published a dataset of sequenced T-cell receptors labelled by interaction with disease particles, which are called antigens [(link)](https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/). We refer to this as the antigen classification dataset.
Adaptive Biotechnologies has published a separate dataset of patients' sequenced T-cell receptors, which are called immune repertoires, labelled by those patients' CMV status [(link)](https://clients.adaptivebiotech.com/pub/emerson-2017-natgen).
We refer to this as the repertoire classification dataset.
Training data is used to fit a model, validation data is used for model selection, and test data is for reporting results. All results on test data must be reported.

To uncover patterns in the non-conforming features in datasets such as these, we present *dynamic kernel matching* (DKM) to augment established statistical classifiers with computational machinery for handling non-conforming features. With DKM, the structure representing non-conforming features plays a pivotal role ensuring that appropriate features go into the appropriate inputs. To illustrate that DKM appropriately handles non-conforming features, we fit (i) a multinomial regression model augmented with DKM to a dataset of sequenced T-cell receptors labelled by disease antigen [(link)](https://github.com/jostmey/dkm/tree/master/antigen-classification-problem/model) and (ii) a logistic regression model augmented with DKM to a dataset of sequenced T-cell repertoires labelled by CMV serostatus [(link)](https://github.com/jostmey/dkm/tree/master/repertoire-classification-problem/model). The fit to training data achieved by these simple statistical classifiers demonstrate DKM can be used to uncover meaningful patterns in complex data.

## Requirements

* [Python3](https://www.python.org/)
* [TensorFlow == v1.14](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)
* [h5py](https://www.h5py.org/)

## Recommended Tools

* [HDF5 Database Viewer](https://www.hdfgroup.org/downloads/hdfview/)

## Download

* Download: [zip](https://github.com/jostmey/dkm/zipball/master)
* Git: `git clone https://github.com/jostmey/dkm`
