## Repertoire Classification Dataset

Adaptive Biotechnologies has published a separate dataset of immune repertoires labelled by CMV exposure that we refer to as the
repertoire classification dataset [(link)](https://clients.adaptivebiotech.com/pub/emerson-2017-natgen). For each patient, their
age and β-chain CDR3 sequences and copy numbers in peripheral blood provide features of their repertoire. Unlike the antigen
classification dataset, T-cell receptors are not labelled by pMHC, but rather the T-cell repertoire is labelled by the patient’s CMV
serostatus. 

Samples have been built from this dataset and split into a training, validation, and test cohort. The samples are stored in a HDF5 file under database.h5. The file, which is over 5GB, is too large to store here. Click [here](https://www.dropbox.com/s/gzp8qy613qeiylx/database.h5?dl=0) to download the database file, which should be placed in this folder.

## Rebuilding the Dataset

The scripts to reconstruct the samples are provided. Because the last script randomly shuffles the samples for the validation and test cohorts, reconstructing the database will result in a different split for the training and validation cohorts.

```
python3 insert_cmv.py
```
