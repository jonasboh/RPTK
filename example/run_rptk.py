import os
import json
import pathlib
import pandas as pd
import numpy as np
from shutil import copyfile

from os import path
import sys

sys.path.append('path/to/rptk/repo/folder/')

from rptk.rptk import RPTK
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
import argparse

def int_or_str(value):
    """Try to convert value to int, otherwise return as string."""
    try:
        return int(value)
    except ValueError:
        return value  

def parse_comma_separated(value):
    """Parse a comma-separated string into a list of strings."""
    value = value.strip("[")
    value = value.strip("]")
    return value.split(',')

def parse_arguments():
    parser = argparse.ArgumentParser(description="User Configuration for RPTK Processing")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--perturbation", type=str, default=None, help="Perturbation methods [connected_component, random_walker]")
    parser.add_argument("--n_features", type=int_or_str, default=10, help="Number of features for selection (number or best)")
    parser.add_argument("--select_models", type=str, default="RandomForestClassifier", help="Models for feature selection")
    parser.add_argument("--resample", type=bool, default=True, help="Resample image/mask to 1x1x1.")
    parser.add_argument("--instability_filter", type=bool, default=True, help="Enable instability filter (True/False)")
    parser.add_argument("--rerun", type=bool, default=False, help="Enable Rerun (True/False), If Ture give output folder of previous run as output_folder.")
    parser.add_argument("--normalization", type=bool, default=False, help="Enable Image (default: z-score) normalization (True/False).")
    return vars(parser.parse_args())

def main():
    user_config = parse_arguments()
    
    print("\n##### Processing RPTK with the following configuration:")
    for key, value in user_config.items():
        print(f"{key}: {value}")
    
    print("\n")
    
    if not user_config["output_folder"].endswith("/"):
        user_config["output_folder"] = user_config["output_folder"] + "/"
        
    # Initialize RPTK with user-defined parameters
    rptk = RPTK(
                            path2confCSV=str(user_config["input_csv"]),
                            n_cpu=user_config["num_cpus"],
                            self_optimize=False, 
                            input_reformat=True, 
                            use_previous_output=user_config["rerun"], 
                            normalization=user_config["normalization"],
                            resampling=user_config["resample"],
                            out_path=user_config["output_folder"]
                        )

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
    rptk.preprocessing()

    # extract features (with PyRadiomics and/or MIRP - use Data fingerprint for optimizing feature extraction)
    rptk.extract_features()

    # filter features
    if not user_config["perturbation"] is None:
        rptk.filter_features(perturbation_method = user_config["perturbation"])
    else:
        rptk.filter_features()

    # select features - Default use Random Forest
    rptk.select_features(n_features=10, model=["RandomForestClassifier"]) 
    # rptk.select_features(n_features= user_config["n_features"], model=[user_config["select_models"]])

    # predict on selected features
    rptk.predict()
    
    print("\nProcessing complete!")
    
if __name__ == "__main__":
    main()
