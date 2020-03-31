## Analysis

Run script to format the model's weights and bias terms and save the values as numpy arrays and CSV files.
```
cd ../
python3 dump_model.py --database ../dataset/database.h5 \
  --table Receptor-PMHC-Complex/train \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --input bin/model \
  --output analysis/model
cd -
```

Convert the weights into similarity scores for an alignment algorithm.
```
python3 similarity_table.py --input model \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --output similarity_table
```

Perform in-silico alanine scan
```
python3 alanine_scan.py --input model \
  --tables similarity_table \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --tra_vgene TRAV27 --tra_cdr3 CAGAGSQGNLIF --tra_jgene TRAJ42 \
  --trb_vgene TRBV19 --trb_cdr3 CASSSRSSYEQYF --trb_jgene TRBJ2-7 \
  --output alanine_scan_1oga
python3 alanine_scan.py --input model \
  --tables similarity_table \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --tra_vgene TRAV27 --tra_cdr3 CAGAIGPSNTGKLIF --tra_jgene TRAJ37 \
  --trb_vgene TRBV19 --trb_cdr3 CASSIRSSYEQYF --trb_jgene TRBJ2-7 \
  --output alanine_scan_5euo
python3 alanine_scan.py --input model \
  --tables similarity_table \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --tra_vgene TRAV35 --tra_cdr3 CAGPGGSSNTGKLIF --tra_jgene TRAJ37 \
  --trb_vgene TRBV27 --trb_cdr3 CASSLIYPGELFF --trb_jgene TRBJ2-2 \
  --output alanine_scan_5e6i
python3 alanine_scan.py --input model \
  --tables similarity_table \
  --tags A0201_GILGFVFTL_Flu-MP_Influenza_binder A0301_KLGGALQAK_IE-1_CMV_binder A0301_RLRAEAQVK_EMNA-3A_EBV_binder A1101_IVTDFSVIK_EBNA-3B_EBV_binder A1101_AVFDRKSDAK_EBNA-3B_EBV_binder B0801_RAKFKQLL_BZLF1_EBV_binder \
  --tra_vgene TRAV24 --tra_cdr3 CAFDTNAGKSTF --tra_jgene TRAJ27 \
  --trb_vgene TRBV19 --trb_cdr3 CASSIFGQREQYF --trb_jgene TRBJ2-7 \
  --output alanine_scan_5isz
```
