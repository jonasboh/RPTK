import pandas as pd
import os
from WORC import download
from pathlib import Path

# download the input files

# Load input CSV file
csv_path = "CRLM_example.csv"
data = pd.read_csv(csv_path)

# Make sure the output directory exists
output_dir = Path("WORC_downloads")
output_dir.mkdir(exist_ok=True)

# Expected columns: 'ID', 'Image', 'Mask' (adjust if needed)
for idx, row in data.iterrows():
    subject_id = row['ID']
    image_url = row['Image']
    mask_url = row['Mask']

    # Download image and mask via WORC API
    print(f"Downloading subject: {subject_id}")
    try:
        download.download_from_url(image_url, output_dir / f"{subject_id}_image.nii.gz")
        download.download_from_url(mask_url, output_dir / f"{subject_id}_mask.nii.gz")
    except Exception as e:
        print(f"Error downloading {subject_id}: {e}")

print("Download complete.")


from os import path
import sys
import os

sys.path.append('/path/to/parent/folder/of/rptk/')

from rptk.rptk import RPTK
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
from rptk.src.feature_extraction.Extractor import Extractor
# -

rptk = RPTK(path2confCSV="data/CRLM_Raw_config_test.csv", 
             n_cpu = 30,  # number of cpu to use
             self_optimize = False,  # enable optimize feature extraction
             input_reformat = True,  # enable reformat for non unique files
             use_previous_output = False,  # use previous output for feature extraction
             normalization = False,
             resampling = False,
             out_path="output/CRLM_test/"
           )

# simple application
# rptk.run()

# Apply each step to see what is goin on (with recommended configuration)
 # get configuration
rptk.get_rptk_config()
    
# create folders
rptk.create_folders()

# check input file format
rptk.check_input_csv_format()

# if ID is a number make it a string
if is_numeric_dtype(rptk.data["ID"]):
    rptk.data["ID"] = rptk.data["ID"].astype(str)

# get Data statistics
rptk.get_data_fingerprint()

# Data preprocessing (image transformations, segmentation perturbations, segmentation evaluation, and Data Fingerprint extraction
rptk.preprocessing(perturbation_method=[])

# extract features
rptk.extract_features()

# filter features
rptk.filter_features(stability_filtering=False)

# select features
rptk.select_features(n_features=10, model=["RandomForestClassifier"])

# predict on selected features
rptk.predict(
            selected_features=rptk.selected_features.copy(),
            shap_analysis=False
            )
