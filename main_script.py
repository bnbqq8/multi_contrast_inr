# test all subjects in test set
import json
import os
from pathlib import Path

# python3 main.py --logging --config config/config_brats_mlpv2_custom.yaml
with open("../IXI_preprocess/dataset_split_20251225.json") as f:
    datalist = json.load(f)
patients = [i.split("/")[-1] for i in datalist["test"]]
for patient in patients:
    cmd = f"python3 main.py --logging --config config/config_brats_mlpv2_custom.yaml --subject_id {patient}"
    os.system(cmd)
