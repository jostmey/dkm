# Statistical Classifiers for Data that Do Not Conform to Rows and Columns: A Case Study of T-cell Receptor Datasets
###### JARED L OSTMEYER, ASSISTANT PROFESSOR, UT SOUTHWESTERN DEPARTMENT OF POPULATION AND DATA SCIENCES

## Introduction

Most statistical classifiers are designed to find patterns in data where numbers fit into rows and columns, as in a spreadsheet. However, many forms of data cannot be structured in this manner. Sequences, like in a text document, cannot be arranged as numbers in a spreadsheet. As is the case with all sequences, both the content and the order of each symbol in the sequence provides important information. To uncover patterns in sequences and other nonconforming data, it is important to consider not only the content of the data but the way that data is structured.

Computer science provides language for describing different ways non-conforming features can be structured. For example, biological data is often structured as a sequence of symbols. The essential property of a sequence is that both the content and order of symbols provide important information. Sets represent another way data can be structured. A set is like a sequence except the order of the symbols does not encode information. Identifying the underlying structure of the features is important because disparate sources of data with the same underlying structure can be handled with the same methods. For example, a social network is a graph of people and the way people are connected just like a molecule is a graph of atoms and the way those atoms are connected, allowing for the same statistical classifier to handle both cases.

To understand different ways data can be structured, we consider two datasets of T-cell receptors, anticipating these datasets to contain signatures for diagnosing disease. The following two T-cell receptor datasets are examples of non-conforming data.

![alt text](artwork/data.png "Layout of data used in this study")

In the antigen classification dataset, T-cell receptors are labelled by interaction with disease particle. Each T-cell contains a sequenced T-cell receptor, represented as categorial variables and sequences, that are used to predict the antigen, represented as a categorial variable. In the repertoire classification problem, a patient's sequence T-cell receptors are labelled by the patient's CMV status. Each row contains a patient’s age, represented as a numeric variable, and their T-cell receptors, represented as a set of sequences using sub-rows, that predict CMV serostatus, represented as a binary variable. In both datasets, best practices are implemented by splitting each dataset into a training, validation, and test set. Training data is used to fit a model, validation data is used for model selection, and test data is for reporting results (all results on test data must be reported!).

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
