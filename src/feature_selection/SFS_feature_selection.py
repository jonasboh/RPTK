import os
import pandas as pd
import numpy as np
import seaborn as sns
import csv
import re
import matplotlib.pyplot as plt
import random
import statistics
import multiprocessing
import sys

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from time import time
import logging
import argparse
from tqdm import tqdm

#sys.path.append('src')
#from feature_filtering.Radiomics_Filter import RadiomicsFilter


def nan_in_columns(df):
    """
    Filter for columns only containing NaN values.
    :param df: pd.DataFrame to be filtered
    :return: list of nan features
    """
    nan_columns = df.columns[df.isna().all()].tolist()

    if len(nan_columns) == 0:
        logger.info("Found No Features with NaN.")
    else:
        logger.warning("Found Features with NaN: " + str(len(nan_columns)))

    for column in nan_columns:
        logger.warning("Feature only containing NaN: " + column)

    return nan_columns

def check_for_nan(data, out_folder):

    nan_features_list = nan_in_columns(df=data)
    # nan_features = self.data[nan_features_list]
    data = data.drop(nan_features_list, axis=1)

    if data.isnull().values.any():
        logging.warning("Warning: Detected NaN value for " + str(len(data.index[data.isna().any(axis=1)].values)) +
                        " samples!")
        logging.warning(data.index[data.isna().any(axis=1)].values)
        logging.warning("Warning: Detected NaN value for " + str(len(data.columns[data.isna().any()])) + " Features!")
        logging.warning(data.columns[data.isna().any()])
        logging.warning(data.isnull().sum())

    data = data.fillna(0)
    nan_df = data[data.isna().any(axis=1)]
    nan_df.to_csv(out_folder + '/nan_samples.csv')

    if nan_df.shape[0] > 0:
        logger.warning("Found " + str(nan_df.shape[0]) + " Samples with NaN!")

        #data = data.drop(nan_df.index)

    # data = data.dropna()

    return data, data.index[data.isna().any(axis=1)].values


def get_selected_data(selected_features_path):
    df = pd.DataFrame()  # columns=['Features_selected'])

    with open(selected_features_path) as file_in:
        for line in tqdm(file_in):
            line = line.replace('\n', '')
            line = line.strip('\n')
            if 'Forward_selected:' in line:
                continue
            if 'Backward_selected:' in line:
                continue
            df = pd.concat([df, pd.Series({'Features_selected': line})], axis=0, join='outer', ignore_index=True)
            # df.append({'Features_selected': line}, ignore_index=True)

    list_ = df[0].tolist()

    return list_

def extract_selected_features(data_path, selected_features_path, out_folder, label_name):
    # check for output dir
    out_path_exist = os.path.exists(out_folder)
    if not out_path_exist:
        # Create a new directory because it does not exist
        os.makedirs(out_folder)

    # Get selected data from file
    selected_feature_list = get_selected_data(selected_features_path)
    logging.info("Number of Selected Features: " + str(len(selected_feature_list)))
    for feature in selected_feature_list:
        logging.info("Selected Feature: " + str(feature))

    if label_name != "":
        # Get labels from file
        raw_data = pd.read_csv(data_path, index_col=0)
        label = raw_data[label_name]
        selected_feature_list.append(label)
    else:
        logging.warning("No label file provided. Only features will be extracted.")

    # Sync IDs
    # check if labels are included
    if label_name not in raw_data.columns:
        logging.info("label not in raw_data.columns")

        if label.shape[0] != raw_data.shape[0]:
            logging.warning("Warning: Number of samples in labels and data are not equal.")
            logging.warning("Warning: Number of samples in labels: " + str(labels.shape[0]))
            logging.warning("Warning: Number of samples in data: " + str(data.shape[0]))
            logging.warning("Synchronizing IDs ... ")

            # Set label according to ID in label file
            for index, row in label.iterrows():
                # Synchronize IDs
                raw_data.loc[index, label_name] = row[label_name]
        else:
            # add label to data
            raw_data[label] = label[label_name]

    if "key_0" in raw_data.columns:
        raw_data = raw_data.rename(columns={"key_0": "SubjectID"})
        raw_data.index = raw_data["SubjectID"]

    raw_data.index.rename("SubjectID")

    if "ID" in selected_feature_list:
        logging.info("Warning: ID in selected features!")
        selected_feature_list.remove("ID")

    # Reduce features from selected features
    selected_features = raw_data[selected_feature_list]

    # save
    selected_features.to_csv(out_folder + '/selected_features_' + str(os.path.basename(data_path)[:-len(".txt")])
                             + ".csv")

    # return selected_feature_list, selected_features, raw_data

### Menue ###
parser = argparse.ArgumentParser(description='Sequential Feature Selection for the best 100 Fetaures.', add_help=True)
parser.add_argument('-i', '--input', help='Path to csv file containing features of interest.')
parser.add_argument('-p', '--predictive_value', help='Name of the predictive value as the label.')
parser.add_argument('-l', '--label_file', default="",
                    help='If label not in file with radiomics features then provide path to CSV file with labels.')
parser.add_argument('-c', '--cpu', default=1, type=int, help='Number of CPUs to use.')
parser.add_argument('-o', '--output', help='Path to folder for output.')

args = vars(parser.parse_args())

INPUT = args["input"]
PREDICTIVE_VALUE = args["predictive_value"]
LABEL_FILE = args["label_file"]
CPU = args["cpu"]
# use max number of CPUs
# CORES = int(multiprocessing.cpu_count())
OUT = args["output"]

# create output folder if not exists
if not os.path.exists(OUT):
    os.makedirs(OUT)

# Take the input file name as output
out = os.path.basename(INPUT)

logger = logging.getLogger("Radiomics Feature Selection")

# create file handler which logs info messages
fh = logging.FileHandler(OUT + "/feature_selection_" + str(out[:-len(".csv")]) + ".log", 'w', 'utf-8')
fh.setLevel(logging.INFO)
# creating a formatter
formatter = logging.Formatter('%(name)s - %(levelname)-8s: %(message)s')
# setting handler format
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

# Read feature file
radiomics = pd.read_csv(INPUT, index_col=0, sep=",")
if "ID" not in radiomics.columns:
    for index in radiomics.index:
        IDs = re.findall('[0-9]+', index)
        radiomics.loc[radiomics.index == index, "ID"] = IDs[0]

# radiomics.index = radiomics["ID"]

# check if labels are included
if (len(LABEL_FILE) > 0):

    # Read label file
    labels = pd.read_csv(LABEL_FILE, index_col=0, sep=",")
    if "ID" not in labels.columns:
        for index in labels.index:
            IDs = re.findall('[0-9]+', index)
            labels.loc[labels.index == index, "ID"] = IDs[0]

    # labels.index = labels["ID"]

    # Sync files
    radiomics[PREDICTIVE_VALUE] = labels[PREDICTIVE_VALUE]
else:
    # Check if feature file includes labels
    if PREDICTIVE_VALUE not in radiomics.columns:
        print("ERROR: Label not found! " + PREDICTIVE_VALUE + " not in " + INPUT)
        sys.exit(0)

# Cleaning data frame from non useful columns
if "ID" in radiomics.columns:
    radiomics = radiomics.drop("ID", axis=1)
if "Unnamed: 0" in radiomics.columns:
    radiomics = radiomics.drop("Unnamed: 0", axis=1)
if "Modality" in radiomics.columns:
    radiomics = radiomics.drop("Modality", axis=1)
if "ROI_Label" in radiomics.columns:
    radiomics = radiomics.drop("ROI_Label", axis=1)
if "Rator" in radiomics.columns:
    radiomics = radiomics.drop("Rator", axis=1)
if "Timepoint" in radiomics.columns:
    radiomics = radiomics.drop("Timepoint", axis=1)
if "Image" in radiomics.columns:
    radiomics = radiomics.drop("Image", axis=1)
if "Mask" in radiomics.columns:
    radiomics = radiomics.drop("Mask", axis=1)
if "Mask_Transformation" in radiomics.columns:
    radiomics = radiomics.drop("Mask_Transformation", axis=1)
if "Image_Transformation" in radiomics.columns:
    radiomics = radiomics.drop("Image_Transformation", axis=1)

logging.info("### Features with NaN ###")
nan_features_list = RadiomicsFilter(logger=logging).nan_in_columns(df=radiomics)
radiomics = radiomics.drop(nan_features_list, axis=1)
logging.warning("Found Features with NaN: " + str(len(nan_features_list)))

logging.info("### Samples with NaN ###")
nan_radiomics = RadiomicsFilter(logger=logging).get_samples_with_nan(df=radiomics)
logging.warning("Found Samples with NaN: " + str(nan_radiomics.shape[0]))
radiomics = radiomics.drop(nan_radiomics.index)

if len(radiomics.columns[radiomics.isna().any()].tolist()) > 0:
    logging.warning(
        "Warning: Detected NaN value for " + str(len(radiomics.columns[radiomics.isna().any()].tolist())) + " columns.")
    logging.info(radiomics.columns[radiomics.isna().any()].tolist())

if len(radiomics.index[radiomics.isna().any(axis=1)].values) > 0:
    logging.warning("Warning: Detected NaN value for " + str(
        len(radiomics.index[radiomics.isna().any(axis=1)].values.tolist())) + " samples.")
    logging.info(radiomics.index[radiomics.isna().any(axis=1)].values.tolist())

# check for nan values in rows
radiomics, nan_values = check_for_nan(radiomics, OUT)
radiomics = radiomics.fillna(0)

# radiomics = radiomics.dropna()

logging.info("### Parameter summary ###")
logging.info("INPUT:", INPUT)
logging.info("OUT:", OUT + "/" + str(out[:-4]) + ".txt")
logging.info("PREDICTIVE_VALUE:", PREDICTIVE_VALUE, "\n")


# print ("Run Job with",str(CORES),"CPU's!")
if not os.path.exists(OUT + "/" + str(out[:-4]) + ".txt"):
    rf = RandomForestClassifier(n_estimators=200)  # , min_impurity_decrease=0.01, min_samples_leaf=5)

    # Labels are the values we want to predict
    labels = np.array(radiomics[PREDICTIVE_VALUE])
    radiomics = radiomics.drop(PREDICTIVE_VALUE, axis=1)

    feature_names = np.array(radiomics.columns)

    # Split the data into training and testing sets (70/30)
    X_train, X_test, y_train, y_test = train_test_split(radiomics, labels, test_size=0.3)

    tic_fwd = time()
    logging.info("Starting forward...")
    # print ('sfs_forward = SequentialFeatureSelector(rf,n_features_to_select=10, direction="forward",n_jobs=-1).fit(X_train, y_train)')
    sfs_forward = SequentialFeatureSelector(
        rf, n_features_to_select=10, direction="forward", n_jobs=CPU
    ).fit(X_train, y_train)
    toc_fwd = time()

    tic_bwd = time()
    logging.info("Starting backward...")
    sfs_backward = SequentialFeatureSelector(
        rf, n_features_to_select=10, direction="backward", n_jobs=CPU
    ).fit(X_train, y_train)
    toc_bwd = time()

    forward_selected = feature_names[sfs_forward.get_support()]
    backward_selected = feature_names[sfs_backward.get_support()]

    with open(OUT + '/' + str(out[:-4]) + ".txt", "w+") as f:
        f.write('Forward_selected:\n')
        for i in forward_selected:
            f.write('%s\n' % i)

        f.write('Backward_selected:\n')
        for i in backward_selected:
            f.write('%s\n' % i)

    logging.info(
        "Features selected by forward sequential selection: ",
        feature_names[sfs_forward.get_support()]
    )
    logging.info("Done in", toc_fwd - tic_fwd, "s")
    logging.info(
        "Features selected by backward sequential selection: ",
        feature_names[sfs_backward.get_support()]
    )
    logging.info("Done in", toc_bwd - tic_bwd, "s")
else:
    logging.info("Skipping sequential feature selection! Output file already exists!")

extract_selected_features(data_path=INPUT,
                          selected_features_path=OUT + '/' + str(out[:-4]) + ".txt",
                          out_folder=OUT,
                          label_name=PREDICTIVE_VALUE)


