## Analysis

Run script to format the model's weights and bias terms and save the values as numpy arrays and CSV files.
```
cd ../
python3 dump_model.py --database  ../dataset/database.h5 --cohort Cohort_I --split samples_train --input bin/model_8 --index 2 --output analysis/model
cd -
```

Convert the weights into similarity scores for an alignment algorithm.
```
python3 similarity_table.py --input model --output similarity_table.csv
```

Score antigen specific T-cell receptors in the 10x Genomics dataset.
```
python3 score_antigen-specific-receptors.py
```

