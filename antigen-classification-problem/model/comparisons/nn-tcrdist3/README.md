## Comparison of DKM to a Nearest Neighbors Model using TCR-Dist3 as the Distance Metric

This folder contains scripts for running a nearest neighbor model on the antigen classification problem using TCR-Dist as the distance metric. The code for TCR-Dist comes from https://github.com/kmayerb/tcrdist3. Once the scripts have been run, the results can be compared to DKM.

The first step is to install TCR-Dist3. We provide instructions for installing and running TCR-Dist3 from its docker image. Use the following command to install the docker image of TCR-Dist3.

```
docker pull quay.io/kmayerb/tcrdist3:0.1.6
```

Now, move the scripts into the parent directory, overwriting the DKM model.

```
mv dump.py ../../
mv dataset.py ../../
mv train_val_test.py ../../
```

We are ready to run the scripts that generate the required CSV files and predictions.

```
cd ../../
mkdir -p bin
python3 dump.py \
  --database ../dataset/database.h5 --project Receptor-PMHC-Complex \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --output db.csv
docker run -it --rm -v $(pwd):/project quay.io/kmayerb/tcrdist3:0.1.6 sh -c "cd project ; python3 train_val_test.py --input db.csv --output bin/model"
```

On the test cohort, we achieve a classification accuracy of 67.87%. Because we are using a nearest neighbors model, our prediction for each test sample is the label of the nearest neighbor (k=1). When multiple neighbors are tied for the nearest neighbor, we use the label from neighbor with the highest relative frequency to break the tie.

