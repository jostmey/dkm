## Antigen Classification Dataset

10XGenomics has published a dataset of T-cell receptors labelled by antigen that we refer to as the antigen classification dataset
[(link)](https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/).
For each T-cell, the V-gene, CDR3 sequence, and J-gene for both the α- and β-chains of the T-cell provide features characterizing
the receptor. A multimer of barcoded pMHCs is sequenced with each T-cell to determine if any of the pMHCs interact with the T-cell
receptor, providing the labels. 

Samples have been built from this dataset and split into a training, validation, and test set. The samples are stored in a [HDF5 file](https://www.hdfgroup.org/downloads/hdfview/)
under
`database.h5`.

## Rebuilding the Dataset

The scripts to reconstruct the samples are provided. Because the last script randomly shuffles the samples, reconstructing
the samples will result in different splits of the data (i.e. the training, validation, and test sets will not
necessarily be the same). Open a Linux terminal and run:

```
cd downloads
sh download.sh
cd ../
python3 insert_10xgenomics.py > insert_10xgenomics.out
```
