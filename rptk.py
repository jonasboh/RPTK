from __future__ import print_function

import os
import psutil
import sys
from threading import Semaphore
import shutil
import concurrent.futures

import mimetypes
import magic 

from typing import List, Tuple, Optional, Dict, Iterable

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype, is_number
import yaml
from iteration_utilities import duplicates
# from p_tqdm import p_map
import time
import datetime
import logging
import json
from pathlib import Path
import torch
import random

from tqdm import *
import glob
import re
import multiprocessing

from threading import Thread
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import seaborn as sns
import nrrd
import nibabel as nib
import pathlib

import matplotlib.colors as mcolors


from sklearn.impute import KNNImputer, SimpleImputer

import scipy.cluster.hierarchy as sch
from matplotlib.patches import Patch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging

from rptk.src.feature_filtering.Radiomics_Filter_exe import RadiomicsFilter
from rptk.src.config.InputDataStatistics import DataStatGenerator
from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.config.structure_generator import out_dir_generator
from rptk.src.segmentation_processing.SegProcessor import SegProcessor
from rptk.src.feature_extraction.Extractor import Extractor
from rptk.src.feature_selection.FeatureSelector import FeatureSelectionPipeline
from rptk.src.model_training.ModelTrainer import ModelTrainer
from rptk.src.feature_filtering.Feature_formater import FeatureFormatter

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.mirp.imageClass import *
from rptk.mirp.roiClass import *

here = Path(__file__).resolve().parent
sys.path.append(str(here / 'src'))      # internal src modules
sys.path.append(str(here.parent))       # parent folder, where mirp lives

# import mirp_predict_lib as mirp_predict
# from mirp_predict_lib import *

# from mirp_pipeline.Preprocessor import Preprocessor
# from mirp_pipeline.Experiment_setup import *

# +

# general useful functionsRadiomics Features

def get_unnamed_cols(df: pd.DataFrame):
    """
    Get the unnamed columns of a dataframe
    :param df: dataframe with unnamed columns
    :return: list of columns contain 'Unnamed'
    """
    unamed_cols = [col for col in df.columns if 'Unnamed' in col]

    return unamed_cols


def drop_cols_with_same_values(df: pd.DataFrame):
    """
    Drop columns with duplicated values
    :param df: dataframe with duplicated columns
    :return df: dataframe without duplicated columns
    """
    df = df.T.drop_duplicates().T

    return df


def get_config_column_name(df: pd.DataFrame, list_of_config_candidates=None):
    """
    Get the columns with configuration parameters if columns does not contain 'config'
    :param df: dataframe with configuration parameters in one column
    :param list_of_col_candidates: list of columns that could contain configuration parameters
    :return: list with column name contain configuration parameters
    """

    # if "config" in df.columns:
    if list_of_config_candidates is None:
        list_of_config_candidates = []

    config_cols = [col for col in df.columns if 'config' == col]
    if len(list_of_config_candidates) > 0:
        if len(config_cols) == 0:
            if df.index.name == "ID":
                df["ID"] = df.index
            elif df.index.name == "id_subject":
                df["id_subject"] = df.index
            # get column with configuration parameters
            for col in list_of_config_candidates:
                if "ID" in df.columns:
                    for config, ID in zip(df[col], df["ID"]):
                        if str(ID) in str(config):
                            config_cols.append(col)
                            break
                elif "id_subject" in df.columns:
                    for config, ID in zip(df[col], df["id_subject"]):
                        if str(ID) in str(config):
                            config_cols.append(col)
                            break
                else:
                    print("Error: Configuration of the CSV file is not correct!")
                    raise ValueError("Configuration of the CSV file is not correct!")


        # if more than one column with "config" in name, drop columns with same values and take the first
        elif len(config_cols) > 1:
            config_df = df[config_cols]
            config_df = drop_cols_with_same_values(df=config_df)
            config_cols = [col for col in config_df.columns if 'config' in col]
            if len(config_cols) > 1:
                config_cols = [config_cols[0]]
    else:
        # if there are nor columns contain "config" in name, try to find config column in unknown columns
        if len(config_cols) == 0:
            list_of_config_candidates = get_unnamed_cols(df)
            for col in list_of_config_candidates:
                for config, ID in zip(df[col], df["ID"]):
                    if str(ID) in str(config):
                        config_cols.append(col)
                        break
        else:
            # if more than one column with "config" in name, drop columns with same values and take the first
            if len(config_cols) > 1:
                config_df = df[config_cols]
                config_df = drop_cols_with_same_values(df=config_df)
                config_cols = [col for col in config_df.columns if 'config' in col]
                if len(config_cols) > 1:
                    config_cols = [config_cols[0]]
            else:
                config_cols = [config_cols[0]]

    return config_cols


def make_dir_if_exist(path: str = ""):
    if path == "":
        raise TypeError("No Path given!")
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)


class RPTK:
    """
    The Radiomics Processing Toolkit (RPTK) is a tool for the extraction of radiomics features from medical images.
    This class is the main control class for RPTK to handle all the subprocesses and sub Routine of RPTK.
    :parameter
    """

    def __init__(self,
                 path2confCSV: str,  # path to csv file with configuration parameters about the data
                 extractor = None,  # extraction tool to use: PyRadiomics, MIRP
                 out_path: str = "",  # path to output folder
                 modality: str = None,  # MRI, CT, PET
                 n_cpu: int = 2,  # number of cpu to use
                 delta: bool = False,  # delta radiomics feature calculation
                 self_optimize: bool = False,  # enable optimize feature extraction
                 use_previous_output: bool = False,  # use previous output for feature extraction
                 rptk_config_json: str = None,  # path to rptk config file
                 Max_num_rois: int = None,  # Maximum number of ROIs to accept from data
                 RunID: str = None,  # ID of the run
                 use_gpu: bool = None,  # use GPU for model training
                 Prediction_Label: str = "Prediction_Label",  # predictive label with the value to predict
                 run_neptune: bool = False,  # Use neptune for model training visualization
                 neptune_project: str = None,  # Neptune project to enter
                 neptune_api_token: str = None,  # Neptune api token for using neptune
                 input_reformat: bool = True,  # If your input file names are non-unique (Image or Mask path in input csv)
                 extract_radiomics_fingerprint: bool = True,  # include radiomics features in the data fingerprint
                 fast_mode: bool = True,  # faster feature extraction but more memory consumption
                 pyradiomics_extraction_yaml = None, # path to pyradiomics extraction yaml file
                 mirp_extraction_yaml = None, # path to mirp extraction yaml file
                 stability_filtering: bool = True, # apply stability filtering
                 shap_analysis = True, # apply shap analysis
                 stable_pretraining = None, # if mdoel size should be trained before to get stable performance
                 optimization_iter: int = None, # number of optimization iterations
                 device: str = None, # device to use for model training
                 optimize : bool = True,  # optimize models
                 rand_state: int = None, # random state for reproducibility
                 models :list = None,  # List of models to predict in the end
                 ensemble: bool = None,  # apply ensmbling or not for model training
                 verbose: int = 0, # verbosity level
                 resample_slice_thickness_threshold: int = None, # threshold for resampling
                 resampling: bool = None, # apply resampling
                 normalization: bool = None, # apply normalization
                 resegmentation: bool = True, # apply resegmentation
                 segmentation_perturbation: bool = True, # apply segmentation perturbation
                 timeout: int = 500, # timeout for the process
                 take_label_changes: bool = False, # take label changes into account over time (only when delta = True)
                 calculate_for_every_segmentation: bool = False, # calculate radiomics for every segmentation or mean all features of all segmentations from the same image
                 additional_rois_to_features: bool = False, # if you have multiple ROIs per sample but want to extract radiomics features for each ROI(recommended for ROIs if they have different meaning). Additional ROIs need to have a description in Mask_Transformation column.
                 imbalance_method: str = "SMOTE", # oversampling method if class imbanace has been detected in the data None is no imablance correction should be performed
                 critical_feature_size: int = None, # critical feature size for feature selection
                 min_feature: int = None, # minimum number of features to select
                 max_feature: int = None, # maximum number of features to select
                 feature_selection: bool = True, # apply feature selection
                 feature_selection_model: str = None, # feature selection model to use
                 variance_threshold: float = None, # variance threshold for feature filtering (min varnaice to keep feature)
                 correlation_threshold: float = None, # correlation threshold for feature filtering (min correlation to identify feature clsuter)
                 ICC_threshold: float = None, # ICC threshold for feature filtering (min ICC to keep feature as stable features only spplicable if segmentation perturbation is enabled)
                 perturbation_method: list = None, # list of perturbation methods to apply
                 dice_threshold: float = None, # dice threshold for segmentation perturbation
                 image_transformation: bool = None, # apply image transformation
                 image_transformation_kernels: list = None, # list of image transformation kernels
                 max_num_rois: int = None, # maximum number of ROIs to accept per sample
                 min_roi_threshold: float = None, # min size of ROI to accept
                 data_fingerprint: pd.DataFrame = None, # user specified data fingerprint
                 peritumoral_seg: bool = None, # whether to perform peritumoral segmenttion (segmentation around the voi)
                 check_segmentation_dimension: bool = True, # checks if segmentations are really 3D segmentations
                 ):

        chunksize: int = n_cpu

        self.path2confCSV = path2confCSV
        self.extractor = extractor
        self.out_path = out_path
        self.modality = modality
        self.n_cpu = n_cpu
        self.chunksize = chunksize
        self.delta = delta
        self.self_optimize = self_optimize
        self.use_previous_output = use_previous_output
        self.rptk_config_json = rptk_config_json
        self.Max_num_rois = Max_num_rois
        self.RunID = RunID
        self.use_gpu = use_gpu
        self.Prediction_Label = Prediction_Label
        self.run_neptune = run_neptune
        self.neptune_project = neptune_project
        self.neptune_api_token = neptune_api_token
        self.input_reformat = input_reformat
        self.extract_radiomics_fingerprint = extract_radiomics_fingerprint
        self.fast_mode = fast_mode
        self.pyradiomics_extraction_yaml = pyradiomics_extraction_yaml
        self.mirp_extraction_yaml = mirp_extraction_yaml
        self.stability_filtering = stability_filtering
        self.shap_analysis = shap_analysis
        self.stable_pretraining = stable_pretraining
        self.optimization_iter = optimization_iter
        self.device = device
        self.optimize = optimize
        self.rand_state = rand_state
        self.models = models
        self.ensemble = ensemble
        self.verbose = verbose
        self.resample_slice_thickness_threshold = resample_slice_thickness_threshold
        self.resampling = resampling
        self.normalization = normalization
        self.data = None
        self.clinical_df = pd.DataFrame()
        self.resegmentation = resegmentation
        self.segmentation_perturbation = segmentation_perturbation
        self.timeout = timeout
        self.take_label_changes = take_label_changes
        self.calculate_for_every_segmentation = calculate_for_every_segmentation
        self.additional_rois_to_features = additional_rois_to_features
        self.imbalance_method = imbalance_method
        self.critical_feature_size = critical_feature_size
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.feature_selection = feature_selection
        self.feature_selection_model = feature_selection_model
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.ICC_threshold = ICC_threshold
        self.perturbation_method = perturbation_method
        self.dice_threshold = dice_threshold
        self.image_transformation = image_transformation
        self.image_transformation_kernels = image_transformation_kernels
        self.min_roi_threshold = min_roi_threshold
        self.max_num_rois = max_num_rois
        self.data_fingerprint = data_fingerprint
        self.peritumoral_seg = peritumoral_seg
        self.check_segmentation_dimension = check_segmentation_dimension

        self.train_ids = None
        self.test_ids = None
        self.val_ids = None

        self.extracted_features = {}
        self.filtered_features = {}
        self.selected_features = {}

        print("##########################################################")
        print("### Executing the Radiomics Processing Toolkit (RPTK) ###")
        print(
            "For usage please cite: Bohn, J.R. et al. (2023). RPTK: The Role of Feature Computation on Prediction Performance. \n" +
            "Medical Image Computing and Computer Assisted Intervention – MICCAI 2023 Workshops. \n" +
            "MICCAI 2023. Lecture Notes in Computer Science, vol 14394. Springer, Cham. DOI: 10.1007/978-3-031-47425-5_11")
        print("USAGE WITHOUT ANY WARRANTY!")
        print("##########################################################")

        if not self.delta:
            if self.take_label_changes:
                self.error.error("Label changes can only be taken into account if delta is enabled for calculating delta radiomics!")
                raise ValueError("Label changes can only be taken into account if delta is enabled for calculating delta radiomics!")

        # check for RunID and set it if not given
        if self.RunID is None:
            self.RunID = time.strftime("%Y%m%d-%H%M%S")

        # check for out_path and set it if not given
        if self.out_path == "":
            self.out_path = os.getcwd() + "/RPTK_output/"

        if not self.out_path.endswith("/"):
            self.out_path += "/"

        # create folder in out_path for output
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

        if self.use_previous_output:
            # Use new log files but use the previous generated output
            up_out_dir = os.path.abspath(os.path.join(os.path.dirname(self.out_path), '..'))

            if up_out_dir.endswith("/"):
                up_out_dir = up_out_dir[:-1]

            log_file_name = up_out_dir + "/RPTK_run_" + self.RunID
        else:
            # Use new log files but use the previous generated putput
            log_file_name = self.out_path + "RPTK_run_" + self.RunID

        # if this is enabled the out_path should be a folder from a run with a RunID
        if not self.use_previous_output:
            make_dir_if_exist(path=self.out_path)
            make_dir_if_exist(path=self.out_path + self.RunID)

            self.out_path = self.out_path + self.RunID + "/"
        else:
            self.current_RunID = self.RunID
            # Replace RunID with the given one from the previous run outfile
            splitted_out_path = self.out_path.split("/")
            self.RunID = splitted_out_path[-2]

        self.logger = LogGenerator(
            log_file_name=log_file_name + ".log",
            # log file name
            logger_topic="RPTK"
        ).generate_log()

        self.error = LogGenerator(
            log_file_name=log_file_name + ".err",
            logger_topic="RPTK Failure"
        ).generate_log()

        self.need_to_convert = False
        self.convert_path = self.out_path + "converted/"
        
        print("##########################################################\n")

        if self.use_previous_output:
            self.logger.info("##########################################################")
            self.logger.info("Use output from previous run: " + str(self.out_path))
            self.logger.info("##########################################################")
            print("Use output from previous run: " + str(self.out_path))

        if self.modality is None:
            print("No modality given! Searching in config file ...")
            if os.path.isdir(self.path2confCSV):
                self.error.error("Path to config file is a directory! Please provide a ptah to a csv file!")
                raise ValueError("Path to config file is a directory! Please provide a ptah to a csv file!")

            self.data = pd.read_csv(self.path2confCSV)
            if "Modality" in self.data.columns:
                modality = self.data["Modality"].unique()
                if len(modality) == 1:
                    self.modality = modality[0]
                    print("Modality found in config file: " + self.modality)
                else:
                    self.error.error("More than one modality in config file!")
                    raise ValueError("More than one modality in config file!")
            else:
                self.error.error("No modality found in config file!")
                raise ValueError("No modality found in config file!")
        else:
            self.data = pd.read_csv(self.path2confCSV)
            if "Modality" not in self.data.columns:
                self.data["Modality"] = self.modality
            else:
                modality = self.data["Modality"].unique()
                if len(modality) == 1:
                    self.modality = modality[0]
                    print("Modality found in config file: " + self.modality)
                else:
                    self.error.error("More than one modality in config file!")
                    raise ValueError("More than one modality in config file!")
        
        if self.rptk_config_json is None:
            self.rptk_config_json = str(Path(__file__).parent.resolve()) + "/src/config/rptk_config.json"
            
        with open(self.rptk_config_json, 'r') as f:
            self.config = json.load(f)

        if self.extractor is None:
            self.extractor = self.config["Feature_extraction_config"]["extractor"]

        # General Config
        if self.use_gpu is None:
            self.use_gpu = self.config["General_config"]["use_gpu"]

        if self.device is None:
            self.device = self.config["General_config"]["device"]

        if self.rand_state is None:
            self.rand_state = self.config["General_config"]["rand_state"]

        # Prediction Config
        if self.optimize is None:
            self.optimize = self.config["RPTK_prediction_config"]["optimize"]
            
        if self.optimization_iter is None:
             self.optimization_iter = self.config["RPTK_prediction_config"]["optimization_iter"]
             
        if self.stable_pretraining is None:
            self.stable_pretraining = self.config["RPTK_prediction_config"]["stable_pretraining"]
            
        if self.shap_analysis is None:
            self.shap_analysis = self.config["RPTK_prediction_config"]["shap_analysis"]
        
        if self.models is None:
            self.models = self.config["RPTK_prediction_config"]["model"]
        
        if self.ensemble is None:
            self.ensemble = self.config["RPTK_prediction_config"]["ensemble"]

        if self.resample_slice_thickness_threshold is None:
            self.resample_slice_thickness_threshold = self.config["Preprocessing_config"]["resample_slice_thickness_threshold"]

        if self.critical_feature_size is None:
            if "critical_feature_size" in self.config["Feature_selection_config"]:
                self.critical_feature_size = self.config["Feature_selection_config"]["critical_feature_size"]
            else:
                self.critical_feature_size = 200

        if self.min_feature is None:
            if "min_feature" in self.config["Feature_selection_config"]:
                self.min_feature = self.config["Feature_selection_config"]["min_feature"]
            else:
                self.min_feature = 5

        if self.max_feature is None:
            if "max_feature" in self.config["Feature_selection_config"]:
                self.max_feature = self.config["Feature_selection_config"]["max_feature"]
            else:
                self.max_feature = 20
        
        if self.feature_selection_model is None:
            if "models" in self.config["Feature_selection_config"]:
                self.feature_selection_model = self.config["Feature_selection_config"]["models"]
        
        if self.variance_threshold is None:
            if "variance_threshold" in self.config["Feature_filtering_config"]:
                self.variance_threshold = self.config["Feature_filtering_config"]["variance_threshold"]

        if self.correlation_threshold is None:
            if "correlation_threshold" in self.config["Feature_filtering_config"]:
                self.correlation_threshold = self.config["Feature_filtering_config"]["correlation_threshold"]

        if self.ICC_threshold is None:
            if "ICC_threshold" in self.config["Feature_filtering_config"]:
                self.ICC_threshold = self.config["Feature_filtering_config"]["ICC_threshold"]

        if self.perturbation_method is None:
            if "perturbation_method" in self.config["Preprocessing_config"]:
                self.perturbation_method = self.config["Preprocessing_config"]["perturbation_method"]

        if self.dice_threshold is None:
            if "dice_threshold" in self.config["Preprocessing_config"]:
                self.dice_threshold = self.config["Preprocessing_config"]["dice_threshold"]

        if self.image_transformation is None:
            if "image_transformation" in self.config["Preprocessing_config"]:
                self.image_transformation = self.config["Preprocessing_config"]["image_transformation"]

        if self.image_transformation_kernels is None:
            if "image_transformation_kernels" in self.config["Preprocessing_config"]:
                self.image_transformation_kernels = self.config["Preprocessing_config"]["image_transformation_kernels"]

        if self.max_num_rois is None:
            if "max_num_rois" in self.config["Preprocessing_config"]:
                self.max_num_rois = self.config["Preprocessing_config"]["max_num_rois"]

        if self.min_roi_threshold is None:
            if "min_roi_threshold" in self.config["Preprocessing_config"]:
                self.min_roi_threshold = self.config["Preprocessing_config"]["min_roi_threshold"]
        
        if self.peritumoral_seg is None:
            self.peritumoral_seg = self.config["Preprocessing_config"]["peritumoral_seg"]

        self.logger.info("#### RPTK Run Configuration ####" + "\n" +
                         "\t\t\tRPTK run ID: " + self.RunID + "\n" +
                         "\t\t\tRPTK output path: " + self.out_path + "\n" +
                         "\t\t\tRPTK modality: " + self.modality + "\n" +
                         "\t\t\tRPTK extractor: " + str(self.extractor) + "\n" +
                         "\t\t\tRPTK use device: " + str(self.device) + "\n" +
                         "\t\t\tRPTK number of cpu: " + str(self.n_cpu) + "\n" +
                         "\t\t\tRPTK chunksize: " + str(self.chunksize) + "\n" +
                         "\t\t\tRPTK delta radiomics: " + str(self.delta) + "\n" +
                         "\t\t\tRPTK optimize extraction: " + str(self.self_optimize) + "\n" +
                         "\t\t\tRPTK use previous output: " + str(self.use_previous_output) + "\n" +
                         "\t\t\tRPTK pipeline config file: " + str(self.rptk_config_json) + "\n" +
                         "\t\t\tRPTK data input file: " + str(self.path2confCSV) + "\n")



        print("### RPTK CONFIGURATION ###")
        print("\t\t\tRPTK run ID: " + str(self.RunID))
        print("\t\t\tRPTK output path: " + str(self.out_path))
        print("\t\t\tRPTK modality: " + str(self.modality))
        print("\t\t\tRPTK reformat input: " + str(self.input_reformat))
        print("\t\t\tRPTK fast mode: " + str(self.fast_mode))
        print("\t\t\tRPTK extract radiomics fingerprint: " + str(self.extract_radiomics_fingerprint))
        print("\t\t\tRPTK perform segmentation perturbation: " + str(self.segmentation_perturbation))
        print("\t\t\tRPTK timeout: " + str(self.timeout))
        print("\t\t\tRPTK pipeline config file: " + str(self.rptk_config_json))
        print("\t\t\tRPTK resampling: " + str(self.resampling))
        print("\t\t\tRPTK resample slice thickness threshold: " + str(self.resample_slice_thickness_threshold))
        print("\t\t\tRPTK max number of ROI per Sample: " + str(self.max_num_rois))
        print("\t\t\tRPTK min ROI size: " + str(self.min_roi_threshold))
        print("### RPTK EXTRACTOR CONFIGURATION:")
        print("\t\t\tRPTK extractor: " + str(self.extractor))
        print("\t\t\tRPTK number of cpu: " + str(self.n_cpu))
        print("\t\t\tRPTK resegmentation: " + str(self.resegmentation))
        print("\t\t\tRPTK image transformation " + str(self.image_transformation))
        print("\t\t\tRPTK image transformation kernels: " + str(self.image_transformation_kernels))
        print("\t\t\tRPTK delta radiomics: " + str(self.delta))
        print("\t\t\tRPTK accept label changes over time: " + str(self.take_label_changes))
        print("\t\t\tRPTK calculate for every segmentation of the same image: " + str(self.calculate_for_every_segmentation))
        print("\t\t\tRPTK optimize extraction: " + str(self.self_optimize))
        print("\t\t\tRPTK use previous output: " + str(self.use_previous_output))
        print("### RPTK FILTERING CONFIGURATION:")
        print("\t\t\tRPTK stability filtering: " + str(self.stability_filtering))
        print("\t\t\tRPTK variance threshold: " + str(self.variance_threshold))
        print("\t\t\tRPTK correlation threshold: " + str(self.correlation_threshold))
        print("\t\t\tRPTK perturbation method/s: " + str(self.perturbation_method))
        print("\t\t\tRPTK dice threshold: " + str(self.dice_threshold))
        print("\t\t\tRPTK ICC threshold: " + str(self.ICC_threshold))
        print("### RPTK SELECTION CONFIGURATION:")
        print("\t\t\tRPTK critical feature size: " + str(self.critical_feature_size))
        print("\t\t\tRPTK min feature: " + str(self.min_feature))
        print("\t\t\tRPTK max feature: " + str(self.max_feature))
        print("\t\t\tRPTK feature selection model: " + str(self.feature_selection_model))
        print("### RPTK PREDICTION CONFIGURATION:")
        print("\t\t\tRPTK seed: " + str(self.rand_state))
        print("\t\t\tRPTK prediction label: " + str(self.Prediction_Label))
        print("\t\t\tRPTK stable pretraining: " + str(self.stable_pretraining))
        print("\t\t\tRPTK imbalance method: " + str(self.imbalance_method))
        print("\t\t\tRPTK optimize: " + str(self.optimize))
        print("\t\t\tRPTK optimization iterations: " + str(self.optimization_iter))
        print("\t\t\tRPTK shap analysis: " + str(self.shap_analysis))
        if self.models is not None:
            print("\t\t\tRPTK models: " + str(self.models))
        print("\t\t\tRPTK ensemble: " + str(self.ensemble))
        print("\t\t\tRPTK verbose: " + str(self.verbose))
        print("##########################################################")

        self.logger.info("### Starting RPTK ###")
        print("### Starting RPTK ###")

    def get_data_from_previous_run(self):
        """
        Get data from previous run
        :return:
        """
        preprocessed_data = {}
        extracted_features = {}
        filtered_features = {}
        selected_features = {}

        # check if out_path is a folder from a run with a RunID
        if isinstance(self.out_path[-1], int):
            self.RunID = os.path.basename(os.path.normpath(self.out_path))
        else:
            self.error.error(
                "If use_previous_output is enabled, the out_path should be a folder from a run with a RunID")
            raise ValueError(
                "If use_previous_output is enabled, the out_path should be a folder from a run with a RunID")

        # search for RunID and output files in the output folder
        previous_output = os.listdir(self.out_path)
        if len(previous_output) > 0:
            for extractor in previous_output:
                print("Get data from previous executed extractor: " + extractor)
                self.logger.info("Get data from previous executed extractor: " + extractor)

                print("Get data from feature extraction outfile ...")
                # get extracted features from previous run
                features = pd.read_csv(self.out_path + extractor + "/extracted_features/" + str(
                    extractor) + "_extraction_" + str(self.RunID) + ".csv")
                extracted_features[extractor] = features

                print("Get data from feature filtering outfile ...")
                # get filtered features from previous run
                filtered_feature = pd.read_csv(self.out_path + extractor + "/filtered_features/" + str(
                    extractor) + "_filtered_" + str(self.RunID) + ".csv")
                filtered_features[extractor] = filtered_feature

            # Get config file with preprocessed data
            # get extracted features from previous run
            # get filtered features from previous run
        # get selected features from previous run

    def get_rptk_config(self):
        """
        Read configuration file for RPTK
        """
        print("Configure RPTK ... ")
        with open(self.rptk_config_json, 'r') as f:
            self.config = json.load(f)

        if self.use_previous_output:
            
            previous_config_path = glob.glob(self.out_path + "*" + self.RunID + ".json")

            if len(previous_config_path) > 0:
                print(f"Loading RPTK config from previous run form {previous_config_path[0]} ... ")
                self.logger.info(f"Loading RPTK config from previous run form {previous_config_path[0]} ... ")

                with open(previous_config_path[0], 'r') as f:
                    previous_config = json.load(f)

                if previous_config != self.config:
                    self.error.warning(
                        "The configuration file for RPTK run {} is different from the previous run. ".format(
                            self.RunID))
                    print("WARNING: The configuration file for RPTK run {} is different from the previous run! ".format(
                        self.RunID))

            else:
                self.logger.info("No configuration file found in the output folder.")
                print("WARNING: No configuration file found in the output folder!")
        else:
            print("Saving config for RPTK run in {} ... ".format("RPTK_config_" + self.RunID + ".json"))
            self.logger.info("RPTK run config in {}".format("RPTK_config_" + self.RunID + ".json"))

            # copy rptk config to output folder
            with open(self.out_path + "RPTK_config_" + self.RunID + ".json", "w") as outfile:
                json.dump(self.config, outfile)
                
        # update parameters
        if self.optimize is None:
            self.optimize = self.config["RPTK_prediction_config"]["optimize"]
            
        if self.optimization_iter is None:
             self.optimization_iter = self.config["RPTK_prediction_config"]["optimization_iter"]
             
        if self.stable_pretraining is None:
            self.stable_pretraining = self.config["RPTK_prediction_config"]["stable_pretraining"]
            
        if self.shap_analysis is None:
            self.shap_analysis = self.config["RPTK_prediction_config"]["shap_analysis"]
        
        if self.models is None:
            self.models = self.config["RPTK_prediction_config"]["model"]
        
        if self.ensemble is None:
            self.ensemble = self.config["RPTK_prediction_config"]["ensemble"]

        if self.resample_slice_thickness_threshold is None:
            self.resample_slice_thickness_threshold = self.config["Preprocessing_config"]["resample_slice_thickness_threshold"]


    def create_folders(self):
        """
        Create folders for output and configuration to handle all outputs and sub-outputs from RPTK
        """

        print("Creating folders to store project files ...")
        self.logger.info("Creating folders to store project files ...")

        # if you want to use the output from a previous run, you have to set the out_path to the folder of the previous run
        self.configs_dir, \
        self.plots_dir, \
        self.ICC_dir, \
        self.SHAP_dir, \
        self.extracted_features_dir, \
        self.filtered_features_dir, \
        self.extracted_features_tmp_dir, \
        self.selected_features_dir, \
        self.preprocessed_data_dir, \
        self.perturbed_seg_dir, \
        self.accepted_perturbed_seg_dir, \
        self.processed_seg_dir, \
        self.transformed_images_dir, \
        self.resampled_images_dir, \
        self.resampled_seg_dir = out_dir_generator(logger=self.logger,
                                                   error=self.error,
                                                   RunID=self.RunID,
                                                   out_path=self.out_path,
                                                   extractors=self.extractor,
                                                   config_file=self.config,
                                                   use_previous_output=self.use_previous_output).create_out_dir()

    def nifti_converter(self, input_file_path: str, out_folder: str, label: int = None):
        """
        Convert any file format to nifti
        :param input_file_path: Path to input file
        :param out_folder: path to output folder
        :param label: label to convert to binary mask
        :return: file path of converted file
        """

        input_file_name = os.path.basename(input_file_path)

        if input_file_path.endswith(".nii.gz"):
            converted_file_path = os.path.join(out_folder, input_file_name)
        else:
            file_format = pathlib.Path(input_file_path).suffixes
            converted_file_path = os.path.join(out_folder, input_file_name[:-len(file_format[0])] + '.nii.gz')
        
        # make sure that only one point is there
        converted_file_path = converted_file_path.replace('..', '.')
        
        if not os.path.isfile(converted_file_path):
            # convert file to nifti
            img = sitk.ReadImage(input_file_path)

            if label is not None:
                # if there is a label given remove all other labels and make if binary mask
                mask_array = sitk.GetArrayFromImage(img)

                # Check if the target label exists in the mask
                unique_labels = np.unique(mask_array)
                if label not in unique_labels:
                    self.error.error("Label {} not found in mask {}".format(label, input_file_name))
                    print("Label {} not found in mask {}".format(label, input_file_name))
                    # raise ValueError("Label {} not found in mask {}".format(label, input_file_name))

                # Create binary mask
                binary_mask = np.where(mask_array == label, 1, 0).astype(np.uint8)
                
                # Convert back to SimpleITK image
                binary_mask_sitk = sitk.GetImageFromArray(binary_mask)
                
                # Copy metadata from original image
                binary_mask_sitk.CopyInformation(img)

                self.logger.info("Changed ROI_Label for sample {} from {} to 1".format(input_file_name,label))
                sitk.WriteImage(binary_mask_sitk, converted_file_path)
            else:
                sitk.WriteImage(img, converted_file_path)

        return converted_file_path

    def check_input_format(self, entry: pd.Series):
        """
        Check for input format and convert if necessary
        :param entry: pd.Series
        """

        # get dataframe
        df = self.data.copy().loc[(self.data.copy()["Mask"] == entry["Mask"]) & (self.data.copy()["Image"] == entry["Image"])]
        
        # Assume if no ROI is given, the ROI Label is 1
        if "ROI_Label" not in self.data.columns:
            df["ROI_Label"] = 1

        # detect automatically and convert the files if necessary
        if not entry["Image"].endswith(".nii.gz"):
            self.need_to_convert = True
            if not os.path.exists(self.convert_path):
                os.makedirs(self.convert_path, exist_ok=True)

            nii_image_path = self.nifti_converter(input_file_path=entry["Image"], out_folder=self.convert_path)
            df.loc[(df["Image"] == entry["Image"]) & (df["Mask"] == entry["Mask"]), "Image"] = nii_image_path

        # detect automatically and convert the files if necessary
        if not entry["Mask"].endswith(".nii.gz"):
            self.need_to_convert = True

            if not os.path.exists(self.convert_path):
                os.makedirs(self.convert_path, exist_ok=True)

            if entry["ROI_Label"] != 1:
                nii_mask_path = self.nifti_converter(input_file_path=entry["Mask"], out_folder=self.convert_path, label=int(entry["ROI_Label"]))
                df.loc[(df["Image"] == entry["Image"]) & (df["Mask"] == entry["Mask"]), "ROI_Label"] = 1
                df.loc[(df["Image"] == entry["Image"]) & (df["Mask"] == entry["Mask"]), "Mask"] = nii_mask_path
            else:
                nii_mask_path = self.nifti_converter(input_file_path=entry["Mask"], out_folder=self.convert_path)
                df.loc[(df["Image"] == entry["Image"]) & (df["Mask"] == entry["Mask"]), "Mask"] = nii_mask_path

        elif entry["ROI_Label"] != 1:
            self.need_to_convert = True

            if not os.path.exists(self.convert_path):
                os.makedirs(self.convert_path, exist_ok=True)

            nii_mask_path = self.nifti_converter(input_file_path=entry["Mask"], out_folder=self.convert_path, label=int(entry["ROI_Label"]))
            df.loc[(df["Image"] == entry["Image"]) & (df["Mask"] == entry["Mask"]), "Mask"] = nii_mask_path
            df.loc[(df["Image"] == entry["Image"]) & (df["Mask"] == entry["Mask"]), "ROI_Label"] = 1

        
        return df


    # def nrrd2nii(self, nrrd_file_path: str, out_folder:str):
    #     """
    #     Convert nrrd files to nii.gz files
    #     :param nrrd_file_path: File path to Nrrd file
    #     :param out_folder: Path to folder to write the .nii.gz file
    #     :return nifti_file_path: Path to nfiti file
    #     """
    #
    #     # load nrrd
    #     # Read the .nrrd file
    #     data, header = nrrd.read(nrrd_file_path)
    #
    #     # Create a NIfTI1Image object
    #     nifti_img = nib.Nifti1Image(data, affine=None)
    #
    #     # Update the NIfTI header with necessary information
    #     nifti_img.header.set_data_dtype(data.dtype)
    #     # nifti_img.header.set_zooms(header['space directions'])
    #
    #     # save nifti
    #     img = nib.Nifti1Image(data, np.eye(4))
    #     nifti_file_path = os.path.join(out_folder, os.path.basename(nrrd_file_path)[:-len(".nrrd")] + '.nii.gz')
    #     nib.save(nifti_img, nifti_file_path)
    #
    #     return nifti_file_path
    
    def detect_file_format(self, filename):
        """
        Detect the file format based on the filename extension and file content.

        :param filename: Path to the file.
        :return: Detected MIME type or 'Unknown format' if not detected.
        """
        # Check based on filename extension
        mime_type, _ = mimetypes.guess_type(filename)
        
        # If MIME type is not found or more precision is needed
        if mime_type is None or filename.endswith(('.nii', '.nii.gz')):
            try:
                mime = magic.Magic(mime=True)
                mime_type = mime.from_file(filename)
            except Exception as e:
                mime_type = "Unknown format"
        
        return mime_type if mime_type else "Unknown format"

    def process_file_format(self, row, column_name, reformat_path, progress_bar):
        """
        Helper function to rename and copy files while ensuring uniqueness.
        :param row: pd.Series
        :param column_name: str
        :param reformat_path: str
        :param progress_bar: tqdm
        :return: str
        """
        file_name = os.path.basename(row[column_name])
        file_path = row[column_name]

        # Skip if the ID is already in the filename
        if str(row["ID"]) in file_name:
            progress_bar.update(1)
            return row[column_name]

        if not self.input_reformat:
            self.error.error(f"Non-unique file name detected: {file_path}. Set `input_reformat` to True.")
            # print(f"Warning - Non-unique file name detected: {file_path}. Set `input_reformat` to True.")
            raise ValueError(f"Non-unique file name detected: {file_path}. Set `input_reformat` to True.")

        if not self.reformat:
            self.logger.info("Detected non-unique file name. Generating unique file names ...")
            self.reformat = True

        # Modify filename based on available columns
        rater = ""
        timepoint = ""
        if "Rater" in self.data.columns:
            if pd.notna(row["Rater"]):
                if row["Rater"] != "":
                    rater = f"_{row['Rater']}"
        if "Timepoint" in self.data.columns and pd.notna(row["Timepoint"]):
            if not self.delta:
                self.error.warning("Timepoint detected, but delta variable is False. "
                                "Consider calculating delta radiomics.")
                if not row['ID'].endswith("-" + str(row['Timepoint'])):
                    timepoint = f"-{row['Timepoint']}"

        # Generate new unique filename
        if file_name.endswith(".nii.gz"):
            filetype = ".nii.gz"
        elif file_name.endswith(".nii"):
            filetype = ".nii"
        elif file_name.endswith(".nrrd"):
            filetype = ".nrrd"
        else:
            self.error.error("Not supported file format: ", Path(os.path.basename(file_path)).suffix)
            raise ValueError("Not supported file format: ", Path(os.path.basename(file_path)).suffix)

        unique_file_name = f"{row['ID']}{timepoint}_{file_name[:-len(filetype)]}{rater}.nii.gz"
        new_file_path = os.path.join(reformat_path, unique_file_name)
        new_file_path = self.clean_filename(new_file_path)

        # HERE 
        if not os.path.exists(new_file_path):
            # Copy file to new location
            if file_path.endswith(".nii.gz"):
                shutil.copy(file_path, new_file_path)
            else:
                sitk_img = sitk.ReadImage(file_path)
                sitk.WriteImage(sitk_img, new_file_path)

        progress_bar.update(1)
        return new_file_path

    def process_batch_file_format(self, batch, progress_bar, img_reformat_path, msk_reformat_path):
        """Process a batch of rows in parallel using multiple threads."""
        with ThreadPoolExecutor(max_workers=self.n_cpu) as executor:
            futures = []
            for _, row in batch.iterrows():
                futures.append(executor.submit(self.process_file_format, row, "Image", img_reformat_path, progress_bar))
                futures.append(executor.submit(self.process_file_format, row, "Mask", msk_reformat_path, progress_bar))

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def generate_unique_format(self, max_workers=None, chunk_size=10):
        """
        Reformat files to make them unique using parallel processing.
        Ensures thread safety, prevents deadlocks, and manages batch submission of jobs.

        Parameters
        ----------
        max_workers : int, optional
            Number of parallel threads for file operations (default: 4).
        chunk_size : int, optional
            Number of rows to process in a batch before waiting for completion (default: 10).

        Returns
        -------
        reformat : bool
            True if reformatting was performed, False otherwise.
        """

        if max_workers is None:
            max_workers = self.n_cpu
            chunk_size = self.n_cpu

        self.reformat = False

        # Convert "Timepoint" column to int if present and has no NaN values
        if "Timepoint" in self.data.columns and not self.data["Timepoint"].isna().any():
            self.data["Timepoint"] = self.data["Timepoint"].astype(int)

        # Convert IDs to string format if they are numbers
        self.data["ID"] = self.data["ID"].apply(lambda x: f"ID-{int(x)}" if isinstance(x, (int, float)) else x)

        # Ensure reformat directories exist
        img_reformat_path = os.path.join(self.out_path, "input_reformated/img/")
        msk_reformat_path = os.path.join(self.out_path, "input_reformated/msk/")
        os.makedirs(img_reformat_path, exist_ok=True)
        os.makedirs(msk_reformat_path, exist_ok=True)

        # Total number of files to process (both images and masks)
        total_files = len(self.data) * 2

        # tqdm progress bar with description
        with tqdm(total=total_files, desc="Check input CSV format", unit="file") as progress_bar:
            batch_list = np.array_split(self.data, max(len(self.data) // chunk_size, 1))
            updated_images = []
            updated_masks = []

            for batch in batch_list:
                results = self.process_batch_file_format(batch, progress_bar, img_reformat_path, msk_reformat_path)
                updated_images.extend(results[::2])  # Images are at even indices
                updated_masks.extend(results[1::2])  # Masks are at odd indices

        # Update DataFrame paths efficiently
        self.data.loc[:, "Image"] = updated_images
        self.data.loc[:, "Mask"] = updated_masks

        if len(self.data[self.data['ID'].str.contains('_')]) > 0:
            self.logger.info("Detected illigal character _ in ID. Replacing it with -.")
            self.error.warning("Detected illigal character _ in ID. Replacing it with -.")

            self.data.loc[:,'ID'] = self.data['ID'].str.replace('_', '-')

        return self.reformat
   
    def normalize_if_needed(self, df: pd.DataFrame, clinical_features: list = [], feature_class_index='feature_class',
                        mean_tol=0.01, std_tol=0.01, min_val=0.0, max_val=1.0, range_tol=0.05):
        """
        Identifies and z-score normalizes non-normalized columns in a DataFrame.
        Assumes 'feature_class' is a special row to be preserved.
        """
        feature_class_included = False
        if feature_class_index in df.index.to_list():
            # Extract feature class row
            feature_class_row = df.loc[feature_class_index]
            feature_class_included = True
        
            # Patient/sample data (all except 'feature_class')
            data = df.drop(index=feature_class_index)
        else:
            data = df.copy()

        # Select only numeric columns
        numeric_cols = data.columns.to_list()

        if "ID.1" in numeric_cols:
            numeric_cols.remove("ID.1")

        numeric_data = data[numeric_cols]

        # Calculate column means and stds
        col_means = numeric_data.mean()
        col_stds = numeric_data.std(ddof=0)  # population std (scipy uses ddof=0 by default)

        # Detect non-normalized features by checking if their min/max fall outside [0, 1] ± tol
        non_normalized_cols = []
        for col in clinical_features:
            if col in numeric_data.columns:
                col_min = numeric_data[col].min()
                col_max = numeric_data[col].max()
                if col_min < (min_val - range_tol) or col_max > (max_val + range_tol):
                    non_normalized_cols.append(col)

        if len(non_normalized_cols)>0:
            # Apply z-score normalization only to these columns
            normalized_data = numeric_data.copy()
            # normalized_data[non_normalized_cols] = numeric_data[non_normalized_cols].apply(zscore, axis=0)
            for col in non_normalized_cols:
                normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std(ddof=0)

            if feature_class_included:
                # Merge feature_class row and normalized data
                df_out = pd.concat([feature_class_row.to_frame().T, normalized_data], axis=0)
            else:
                df_out = normalized_data.copy()
        else:
            df_out = df

        return df_out, non_normalized_cols

    def plot_custom_radiomics_heatmap(self, df, prediction_labels,  additional_params=None, normalize=False, normalization_method='zscore', method='average', figsize=(20, 15), y_labelpad=10, title=None, save_path=None, no_feature_names=False, prediction_label_meaning=None):
        """
        Plot a custom heatmap with patients sorted by prediction labels, hierarchical clustering applied
        to the sorted patients, and a feature class bar attached to the heatmap.

        Parameters:
        :param df (pandas.DataFrame): Dataframe containing radiomics features with patients as rows and features as columns.
                            The row with index 'feature_class' contains the feature class information for clustering.
        :param prediction_labels (pandas.Series or list): Prediction labels corresponding to each patient.
        
        :param additional_params (dict, optional): Dictionary of additional parameters to plot below the heatmap, 
                                            where keys are parameter names and values are lists/Series of parameter values.
        :param normalize (bool): Whether to normalize the radiomics features. Default is False.
        :param normalization_method (str): The normalization method to apply ('minmax' or 'zscore'). Default is 'zscore'.
        :param method (str): Linkage method for clustering (default is 'average').
        :param figsize (tuple): Size of the figure. Default is (20, 15).
        :param y_labelpad (int): Padding between y-axis labels and the heatmap. Default is 10.
        :param title (str, optional): Title for the heatmap plot. Default is None.
        :param save_path (str, optional): File path to save the figure. Default is None.
        :param no_feature_names (bool, optinal): If the feature names on the y axis should be displayed or not
        
        Example usage:
        df - your dataframe with radiomics features, including the 'feature_class' row
        prediction_labels - your prediction labels in a list or pandas Series
        additional_params = {'Clusters': cluster data, 'ECOG': ecog data, 'Gender': gender data, ...}
        plot_custom_radiomics_heatmap(df, prediction_labels, additional_params, normalize=False, figsize=(20, 15), y_labelpad=15, title='My Heatmap Plot', save_path="heatmap.png")
        """

        # --- fixed feature-class color map (use consistent labels after your .replace('_',' ') step) ---
        FEATURE_CLASS_ORDER = [
            "GLCM", "GLDZM", "GLRLM", "GLSZM", "NGLDM","NGTDM",
            "IVH", "IS", "IH", "Morphological", "Diagnostic", "Firstorder",
            "LI", "Clinical feature"
        ]

        FEATURE_CLASS_COLORS = {
            "GLCM":             "#ff7f0e",
            "GLDZM":            "#2ca02c",
            "GLRLM":            "#d62728",
            "GLSZM":            "#9467bd",
            "NGLDM":            "#8c564b",
            "NGTDM":            "#1f77b4",
            "IVH":              "#e377c2",
            "IS":               "#7f7f7f",
            "IH":               "#bcbd22",
            "Morphological":    "#17becf",
            "Diagnostic":       "#aec7e8",
            "Firstorder":       "#ffbb78",
            "LI":               "#98df8a",
            "Clinical feature":"#ff9896",
        }

        # optional: fallback if a class name slips through that's not in the dict
        FEATURE_CLASS_FALLBACK = "#9e9e9e"

        def _normalize_clinical_features(df, feature_class_series):
            """Z-score normalize only columns labeled 'Clinical feature'."""
            df_copy = df.copy()
            clinical_cols = feature_class_series[feature_class_series == "Clinical feature"].index

            if len(clinical_cols) == 0:
                return df_copy  # nothing to do

            # z-score normalization per feature (column-wise)
            for col in clinical_cols:
                col_mean = df_copy[col].mean()
                col_std = df_copy[col].std()
                if col_std != 0:
                    df_copy[col] = (df_copy[col] - col_mean) / col_std
                    # clip to [-2, 2] to stay visually comparable
                    df_copy[col] = df_copy[col].clip(-2, 2)
                else:
                    # constant column — set to 0
                    df_copy[col] = 0
            return df_copy

        def _to_rgb_triplet(c):
            """Return (r,g,b) floats in [0,1]. Accepts hex, named color, 3/4-tuples."""
            rgb = mcolors.to_rgba(c)  # handles hex strings, named colors, tuples; returns (r,g,b,a)
            return (rgb[0], rgb[1], rgb[2])

        # --- publication typography defaults (local to this function if you prefer rc_context) ---
        plt.rcParams.update({
            "font.family": "Arial",      
            "font.size": 12,                      # base
            "axes.labelsize": 18,                 # axis titles
            "axes.titlesize": 22,                 # axes' own .set_title if used
            "xtick.labelsize": 12,                # tick labels
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
            "figure.titlesize": 26                # suptitle
        })

        custom_label_palette = [
                        "#40E0D0",  # turquoise
                        "#8A2BE2",  # purple
                    ]

        
        # vector-friendly font embedding
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"]  = 42
        plt.rcParams["svg.fonttype"] = "none"
        
        # Extract feature_class row and remove it from the dataframe
        feature_class_raw = df.loc['feature_class']
        # Format feature classes
        feature_class = feature_class_raw.apply(lambda s: s.replace('_', ' '))# .capitalize())
        df = df.drop(index='feature_class').astype(float)  # Convert to float
        df = _normalize_clinical_features(df, feature_class)
        
        # Normalize the radiomics features if requested
        if normalize:
            if normalization_method == 'minmax':
                normalized_df = (df - df.min()) / (df.max() - df.min())
            elif normalization_method == 'zscore':
                normalized_df = (df - df.mean()) / df.std()
            else:
                raise ValueError("Normalization method should be either 'minmax' or 'zscore'")
        else:
            normalized_df = df  # No normalization applied if normalize=False

        # Sort the patients by prediction labels first (this order will not change)
        sorted_idx = np.argsort(prediction_labels)
        sorted_df = normalized_df.iloc[sorted_idx]
        sorted_labels = prediction_labels.iloc[sorted_idx].values  # Convert to NumPy array

        # Generate the linkage for dendrogram (but don't reorder the patients)
        patient_links = sch.linkage(sorted_df, method=method, metric='euclidean')
        
        # Manually pass the leaf order (sorted_idx) to the dendrogram
        patient_dendrogram = sch.dendrogram(patient_links, no_plot=True)
        leaf_order = np.arange(len(sorted_idx))  # Keep the original order (sorted by prediction labels)

        # Perform hierarchical clustering on feature_class (columns)
        feature_links = sch.linkage(pd.get_dummies(feature_class), method=method)
        feature_dendrogram = sch.dendrogram(feature_links, no_plot=True)
        feature_order = feature_dendrogram['leaves']
        
        # Reorder the features (columns) according to the clustering
        sorted_df = sorted_df.iloc[:, feature_order]
        feature_class = feature_class.iloc[feature_order]

        # Create a colormap for feature class
        unique_classes = feature_class.unique()
        num_classes = len(unique_classes)
        
        # Assign colors to each feature class
        #class_colors = sns.color_palette('Set2', num_classes)
        #class_color_map = dict(zip(unique_classes, class_colors))
        
        # Create a fixed colormap for feature class (subset of the fixed dict)
        unique_classes = feature_class.unique()

        # Use fixed colors; convert to RGB triplets for imshow
        class_color_map = {
            cls: _to_rgb_triplet(FEATURE_CLASS_COLORS.get(cls, FEATURE_CLASS_FALLBACK))
            for cls in unique_classes
        }

        # Create a colormap for prediction labels
        unique_labels = np.unique(prediction_labels)
        num_labels = len(unique_labels)
        # label_colors = sns.color_palette('husl', num_labels)
        label_colors = sns.color_palette(custom_label_palette, num_labels)
        # label_colors = custom_label_palette[:num_labels]
        # label_colors = custom_label_palette[:len(unique_labels)]
  
        label_color_map = dict(zip(unique_labels, label_colors))

        
        # Create the figure and grid for subplots
        num_param_rows = 1  # For prediction labels (always shown)
        if additional_params and len(additional_params) > 0:
            num_param_rows += len(additional_params)

        # Adjust height ratios dynamically: fixed smaller size for prediction labels and other parameters
        height_ratios = [0.2, 0.6] + [0.05] * num_param_rows  # Reduced the dendrogram space
        width_ratios = [0.85, 0.0984, 0.15, 0.15]  # Adjusted width ratios for the legends

        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(nrows=2 + num_param_rows, ncols=4, width_ratios=width_ratios, height_ratios=height_ratios)

        # Plot the dendrogram for patients (directly attached to the top of the heatmap, without border)
        ax_dendro_patients = fig.add_subplot(grid[0, 0])
        sch.dendrogram(patient_links, ax=ax_dendro_patients, no_labels=True, color_threshold=0, 
                    leaf_rotation=90, leaf_font_size=12, labels=leaf_order)
        ax_dendro_patients.set_xticks([])
        ax_dendro_patients.set_yticks([])
        ax_dendro_patients.spines['top'].set_visible(False)
        ax_dendro_patients.spines['right'].set_visible(False)
        ax_dendro_patients.spines['left'].set_visible(False)
        ax_dendro_patients.spines['bottom'].set_visible(False)

        # Create the heatmap using imshow, integrating the hierarchical clustering
        ax_heatmap = fig.add_subplot(grid[1, 0])
        heatmap_data = sorted_df.T  # Transpose to match rows as features
        #im = ax_heatmap.imshow(heatmap_data, 
        #                       aspect='auto', 
        #                       cmap='Spectral_r', 
        #                       vmin=-2, 
        #                       vmax=2)
        
        n_rows, n_cols = heatmap_data.shape

        # Build grid edges
        x = np.arange(n_cols + 1)
        y = np.arange(n_rows + 1)

        im = ax_heatmap.pcolormesh(
                                    x, y, heatmap_data.values,
                                    cmap='Spectral_r',
                                    vmin=-2, vmax=2,
                                    shading='auto'   # <- auto is safer, handles edge cases
                                   )
        
        # Match imshow orientation: feature 0 at the top
        ax_heatmap.invert_yaxis()
        
        ax_heatmap.set_xlabel('Patients', fontsize=22, labelpad=10)
        ax_heatmap.set_ylabel('Features', fontsize=22, labelpad=y_labelpad)
        
        # Determine number of patients (columns = patients)
        n_patients = heatmap_data.shape[1]

        # Decide step size based on sample size
        if n_patients <= 100:
            step = 10
        elif n_patients <= 1000:
            step = 100
        elif n_patients <= 2000:
            step = 200
        else:
            step = n_patients // 10   # fallback: ~10 ticks total

        # Generate ticks
        xticks = np.arange(0, n_patients, step)

        # Apply ticks to axis
        ax_heatmap.set_xticks(xticks)
        ax_heatmap.set_xticklabels(xticks, fontsize=20, rotation=0, ha='center')

        # Control spacing from axis
        ax_heatmap.tick_params(axis='x', labelsize=20, pad=6)

        # ax_heatmap.set_xlabel('Patients', fontsize=12)
        # ax_heatmap.set_ylabel('Features', labelpad=y_labelpad, fontsize=12)

        # Ensure all feature names are shown with increased padding and smaller font size
        if no_feature_names:
            ax_heatmap.set_yticks([])
            ax_heatmap.set_yticklabels([])
        else:
            ax_heatmap.set_yticks(np.arange(n_rows) + 0.5)  # center of each row
            ax_heatmap.set_yticklabels(sorted_df.columns, rotation=0, fontsize=20)
            
            # ax_heatmap.set_yticks(np.arange(len(sorted_df.columns)))
            # ax_heatmap.set_yticklabels(sorted_df.columns, rotation=0, fontsize=20)  # Reduce fontsize to 5 for readability
        
        
        # Control tick label sizes/padding explicitly
        # ax_heatmap.tick_params(axis='x', which='both', labelsize=12, pad=6)
        # ax_heatmap.tick_params(axis='y', which='both', labelsize=12, pad=2)
        
        # ax_heatmap.set_xticks([])

        # Plot the feature class bar aligned with the features (directly attached to the heatmap on the y-axis)
        ax_class = fig.add_subplot(grid[1, 1])
        feature_class_colors = np.array([class_color_map[c] for c in feature_class])
        ax_class.imshow(feature_class_colors.reshape(-1, 1, 3), aspect='auto', interpolation='nearest')
        ax_class.set_xticks([])
        ax_class.set_yticks([])  # Removed y-tick labels
        ax_class.set_title('Feature Classes', fontsize=20, pad=10)  # Label above the feature class bar

        # Plot the prediction labels below the heatmap
        ax_labels = fig.add_subplot(grid[2, 0])
        ax_labels.imshow(np.array([label_color_map[lbl] for lbl in sorted_labels]).reshape(1, -1, 3), aspect='auto', extent=[0, sorted_df.shape[0], 0, 1])
        ax_labels.set_xticks([])
        ax_labels.set_yticks([])

        text_length = len(str(prediction_labels.name))
        dynamic_pad = 20 + int(text_length * 2.5)

        # ax_labels.set_ylabel(prediction_labels.name, rotation=0, labelpad=dynamic_pad, fontsize=18)
        ax_labels.set_ylabel(prediction_labels.name, rotation=0, fontsize=20)
        ax_labels.yaxis.set_label_coords(-0.20, 0.5) # negative x moves it left, 0.5 keeps it vertically centered
        
        param_axes = []
        
        # Plot additional parameters (if any) below the prediction labels
        if additional_params and len(additional_params) > 0:
            param_keys = list(additional_params.keys())
            param_values = [np.array(additional_params[key])[sorted_idx] for key in param_keys]  # Apply same sorting order
            
            # Plot each additional parameter below the prediction labels
            for i, (param, values) in enumerate(zip(param_keys, param_values)):
                ax_param = fig.add_subplot(grid[3 + i, 0])
                ax_param.imshow(values.reshape(1, -1), aspect='auto', cmap='viridis', extent=[0, sorted_df.shape[0], 0, 1])
                ax_param.set_xticks([])
                ax_param.set_yticks([])
                ax_param.set_ylabel(param, rotation=0, labelpad=40, fontsize=20)
                
                param_axes.append(ax_param) 

        # Create the feature class legend in the top right corner
        # Present classes in a stable, fixed order
        present_classes = [cls for cls in FEATURE_CLASS_ORDER if cls in unique_classes]
        feature_class_handles = [
            Patch(
                facecolor=FEATURE_CLASS_COLORS.get(cls, FEATURE_CLASS_FALLBACK),
                label=cls
            )
            for cls in present_classes
        ]

        ax_legend_feature_class = fig.add_subplot(grid[0, 2])
        ax_legend_feature_class.legend(
            handles=feature_class_handles,
            loc='upper left',
            title='Feature Classes',
            fontsize=18,
            title_fontsize=20,
        )

        ax_legend_feature_class.axis('off')  # Hide the axis for the legend


        if prediction_label_meaning is not None:
            # Create legend handles using the provided label meanings
            prediction_label_handles = [
                Patch(
                    facecolor=label_color_map[lbl],
                    label=f'{int(lbl)} = {prediction_label_meaning[int(lbl)]}'
                )
                for lbl in unique_labels
            ]
            
            # Add a new subplot to place the legend
            ax_legend_prediction_label = fig.add_subplot(grid[1, 2])
            ax_legend_prediction_label.legend(
                                                    handles=prediction_label_handles,
                                                    loc='upper center',
                                                    title=prediction_labels.name,
                                                    fontsize=18,
                                                    title_fontsize=20,
                                                )
            ax_legend_prediction_label.axis('off')

        else:
            # Create the prediction label legend below the feature class legend
            prediction_label_handles = [Patch(facecolor=label_color_map[lbl], label=f'Label {lbl}') for lbl in unique_labels]
            ax_legend_prediction_label = fig.add_subplot(grid[1, 2])
            ax_legend_prediction_label.legend(handles=prediction_label_handles, loc='upper center', title=prediction_labels.name, fontsize=18, title_fontsize=20)
            ax_legend_prediction_label.axis('off')  # Hide the axis for the legend

        # Add color bar for heatmap values next to the legends
        # cbar_ax = fig.add_subplot(grid[0, 3])
        cbar_ax = fig.add_axes([0.85, 0.20, 0.02, 0.3])  # [left, bottom, width, height]
        # cbar = fig.colorbar(im, cax=cbar_ax)
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', pad=0.02)
        cbar.set_label('Feature Value', fontsize=20)  # Legend for heatmap values
        cbar.ax.tick_params(labelsize=16, length=4, width=0.8)
        cbar.ax.set_aspect(20)  # Adjust color bar height to match legend

        # now override GridSpec’s geometry
        fig.draw(fig.canvas.get_renderer())  # force layout
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])
        
        
        #cbar_ax = fig.add_subplot(grid[0, 3])
        # cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        #cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', pad=0.02, fraction=0.05)
        #cbar.set_label('Feature Value', fontsize=18)  # Legend for heatmap values
        #cbar.ax.tick_params(labelsize=16, length=4, width=0.8)
        # Adjust the height of the color bar to match the feature class legend
        #cbar.ax.set_aspect(20)  # Adjust color bar height to match legend
        
        #pos = cbar_ax.get_position()   # get [x0, y0, width, height]
        #cbar_ax.set_position([pos.x0 - 0.1, pos.y0, pos.width, pos.height])

        # Set the title if provided
        if title:
            plt.suptitle(title, fontsize=32, y=0.98)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        
        for ax in [ax_heatmap, ax_class, ax_labels]:
            for s in ax.spines.values():
                s.set_linewidth(0.8)

        # --- widen heatmap relative to class bar ---
        fig.canvas.draw()  # update positions

        pos_heatmap = ax_heatmap.get_position()
        pos_class   = ax_class.get_position()

        left_margin = 0.06   # fraction of figure width from left (tune as needed)
        gap_right   = 0.012  # small gap to class bar

        new_x0    = left_margin
        new_width = max(0.01, pos_class.x0 - gap_right - new_x0)

        ax_heatmap.set_position([new_x0, pos_heatmap.y0, new_width, pos_heatmap.height])
        
        # --- align widths of dendrogram + bottom strips to the heatmap ---
        fig.canvas.draw()
        ref = ax_heatmap.get_position()

        def match_width(ax, ref_pos):
            pos = ax.get_position()
            ax.set_position([ref_pos.x0, pos.y0, ref_pos.width, pos.height])

        # Align dendrogram (above the heatmap)
        match_width(ax_dendro_patients, ref)

        # Align prediction-label strip (below the heatmap)
        match_width(ax_labels, ref)

        # Align additional parameter strips (if any)
        for ax in param_axes:
            match_width(ax, ref)
        # --- end alignment ---

        # snug y tick labels closer to the heatmap
        for lab in ax_heatmap.get_yticklabels():
            lab.set_horizontalalignment('right')
        ax_heatmap.tick_params(axis='y', pad=2)

        # (optional) slim the class bar a bit if you need more space
        # pos_c = ax_class.get_position()
        # ax_class.set_position([pos_c.x0, pos_c.y0, pos_c.width * 0.8, pos_c.height])
        # --- end heatmap adjustment ---

        for ax in [ax_heatmap, ax_class, ax_labels]:
            for s in ax.spines.values():
                s.set_linewidth(0.8)

        # Save/show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=600)
            p = Path(save_path)
            plt.savefig(p.with_suffix('.pdf'), bbox_inches='tight')
        else:
            plt.show()
        
        plt.clf()
        plt.close()

    import re


    def rename_radiomics_columns(self,
                            df: pd.DataFrame,
                            feature_class_row: str = "feature_class",
                            clinical_classes: Iterable[str] = ("Clinical_features", "Clinical features", "Clinical", "Clinical feature", "Clinical_feature"),
                            drop_gradient: bool = False,
                            return_mapping: bool = False,
                            extractor:str = "MIRP",
                        ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Rename MIRP-style radiomics feature columns into PyRadiomics-like names,
        while leaving clinical features untouched (gated by a per-column row called `feature_class`).

        Parameters
        ----------
        df : pd.DataFrame
            Input table where columns are features and rows include data; may include a row named `feature_class`
            that labels each column's class. If absent, all columns are treated as radiomics and renamed.
        feature_class_row : str
            Name of the row that stores per-column class labels.
        clinical_classes : str
            Exact value in `feature_class_row` that designates a clinical feature (these columns are not renamed).
        drop_gradient : bool
            If True, drops the token 'GradientMagnitude' from resulting names.
        return_mapping : bool
            If True, returns (renamed_df, mapping_df). Otherwise returns renamed_df only.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        """
        # ---- helpers for class matching -------------------------------------------------
        def _norm(s: str) -> str:
            # normalize for robust comparison: lowercase, remove spaces/underscores
            return re.sub(r"[ _]+", "", str(s).strip())

        # ------------------------------
        # Replacement rules (customize here)
        # ------------------------------
        if extractor == "MIRP":
            MULTI_RULES = [  
                            (("rlnu","norm"), "RLNUNorm"),
                            (("zs","entr"), "ZSEntr"),
                            (("peak","glob"), "GIPeak"),  # Global intensity peak
                            (("PEAK","glob"), "GIPeak"),  # Global intensity peak
                            (("zd","var"), "ZDVar"), # Zone distance variance
                            (("zd","entr"), "ZDEntr"), # Zone distance entropy
                            (("diff","entr"), "DiffEntr"), # Difference entropy
                            (("integ","int"), "IntegInt"), # Integrated intensity
                            (("rl","entr"), "RunLEntr"),
                            (("info","corr1"), "InfoCorr1"),
                            (("gauss","s2.0"), "Gauss"),
                            (("gabor","s","3.0","g","1.0","l","0.9","t","0.0","3D"), "Gabor"),
                            (("gabor","s","3.0","g","1.0","l","0.9","t","0.0"), "Gabor"),
                            (("inv","diff","mom","norm"), "IDMN"),
                            (("inv","diff","norm"), "IDN"),
                            (("info","corr2"), "IMCorr2"),
                            (("sum","entr"), "SUMEnt"),
                            (("laws","l5s5e5","energy","delta","7","invar"), "Laws"),
                            (("laws","e5e5e5","energy","delta","7","invar"), "Laws"),
                            # (("lgce",), "LGCE"),   
                            # (("hdlge",), "HDLGLE"),  # High dependence low grey level emphasis
                            (("zs","var"), "ZoneSizeVariance"),
                            (("int","mean","int","init","roi"), "MeanInitRoi"),
                            (("diff","i25","i75"), "IQR"),
                            (("diff","avg"), "DiffMean"),
                            (("clust","prom"), "ClusterProminence"),
                            (("zd","entr"), "ZeroDistanceEntropy"),
                            (("inv","var"), "InVar"),
                            (("glnu","norm"), "GLNUNorm"),
                            (("vol","dens","aabb"), "VolumeDensityAABB"),
                            (("grad","g"), "GradientMagnitude"),
                            (("joint","max"), "JointMax"),                  # joint_Max_d1_2d_s
                            (("diff","mean"), "DifferenceMean"),                # diff_Mean_d1_2d_Mean
                            (("Diff","mean"), "DifferenceMean"),                # diff_Mean_d1_2d_Mean
                            (("dc","energy"), "JointEnergy"),                   # dc_energy_d1_a0_2d(_margin)
                            (("int","bb","dim","y","init","roi"), "BoundingBoxDimY_InitRoi"),  # int_bb_dim_y_init_roi
                            (("mean","d","15"), "Mean"),
                            (("Mean","d","15"), "Mean"),
                        ]

            SINGLE_RULES: Dict[str, Optional[str]] = {
                "lde":"LDE",
                "sae":"SAE",
                "sahg":"SAHG",
                "lgce":"LGCE",
                "lrlge":"LRLGE",
                "salg":"SALGLE",
                "hdlge":"HDLGLE", # High dependence low grey level emphasis
                "rmad":"RMAD",
                "sze":"SZE",

                # First-order
                "av": "mean",
                "avg": "mean",
                "mean": "mean",
                "max": "Max",
                "min": "Min",
                "range": "Range",
                "kurt": "Kurtosis",
                "mode":"Mode",
                "skew":"Skewness",
                "qcod": "QuantCoefDisp",
                "InverseVariance":"InVar",
                "complexity":"Complexity",
                "contrast":"Contrast",
                # Texture
                "invar": "InVar",
                "hde": "HDE",
                "ldlge": "LDLGE", # Large distance low grey level emphasis
                "szlge": "SZLGE",
                "sdlge": "SDLGE", # Small distance low grey level emphasis
                "lzlge": "LZLGE",
                "lrhge":"LRHGE", # Long run high grey level emphasis
                "ldhge":"LDHGE", # Large distance high grey level emphasis
                "var":"Variance",
                "entropy":"Entropy",
                "sphericity":"Sphericity",
                # Preprocessing
                "sre":"SRE",
                "lre":"LRE", # Long runs emphasis
                "fbs": "",
                "mrg": "",
                "w25.0": "",
                "a0.0": "",
                "rlnu":"RLNU",
                # Geometry/meta
                "d1": "",
                "2d": "2D",
                "3d": "3D",
                "a0": "",
                "wavelet-HHH": "WaveletHHH",
                "wavelet-HHL": "WaveletHHL",
                "wavelet-HLH": "WaveletHLH",
                "wavelet-LLH": "WaveletLLH",
                "wavelet-HLL": "WaveletHLL",
                "wavelet-LHH": "WaveletLHH",
                "wavelet-hhh": "WaveletHHH",
                "wavelet-hhl": "WaveletHHL",
                "wavelet-hlh": "WaveletHLH",
                "wavelet-llh": "WaveletLLH",
                "wavelet-hll": "WaveletHLL",
                "wavelet-lhh": "WaveletLHH",
                "wavelet-LLL": "WaveletLLL",
                "wavelet-LHL": "WaveletLHL",
                "exponential": "",
                "lbp-2d": "LBP2D",
                "gabor": "Gabor",
                "laws": "Laws",
                "l5s5e5": "",
                "e5e5e5":"",
                "energy":"Energy",
                "delta":"Delta",
                "square":"Square",
                "7":"",
                "sde":"SDE", #"SmallDistanceEmphasis",
                "logarithm":"Log",
                "squareroot":"SquareRoot",
                "sphericity":"Sphericity",
                # Drop lone 'v' (voxel/volumetric flag) from names
                "v": "",
                "cov": "Covariance",
                "gradient":"Gradient",
            }

            SINGLE_RULES.update({
                            # Aggregations (lowercase suffixes)
                            "avg": "Mean",
                            "av": "Mean",
                            "mean": "Mean",
                            "s": "Std",                      # trailing _s → _std
                            "median": "Median",
                            

                            # Texture abbreviations you listed
                            "glnu": "GLNU",                     # glnu_2d_(Mean|s)
                            "lgre": "LGLRE",                    # lgre_2d_Mean_margin
                            "szhge": "SZHGLE",            # szhge_2d
                            "zdnu": "ZDNU",                  # zdnu_2d
                            "salg":"SALGLE",
                            "Minimum":"Min",
                            "Maximum":"Max",
                            # Dose-volume style abbreviations
                            "v75": "V75",
                            "v50": "V50",

                            # Keep existing entries from your dict…
                            "invar": "InverseVariance",
                            "hde": "HDE",
                            "szlge": "SZLGE",
                            "lzlge": "LZLGE",
                            "zd_entr": "ZeroDistanceEntropy",
                            "vol_dens_aabb": "VolumeDensityAABB",

                            # Preprocessing / markers (as you had)
                            "grad_g": "GradientMagnitude",
                            "fbs": "",
                            "mrg": "",
                            "w25.0": "",
                            "a0.0": "",
                            "margin": "margin",

                            # Geometry/meta
                            "d1": "",
                            "2d": "2D",
                            "3d": "3D",
                            "a0": "",

                            # Drop lone 'v' flag
                            "v": "",
                        })
        else:
            MULTI_RULES = [  
                            (("gauss","s2.0"), "Gauss"),
                            (("s","3.0","g","1.0","l","0.9","t","0.0","3D"), ""),
                            (("s","3.0","g","1.0","l","0.9","t","0.0"), ""),
                            (("mean","d","15"), "Mean"),
                            (("Mean","d","15"), "Mean"),
                            (("laws","e5e5e5","energy","delta","7","invar"), "Laws"),
                        ]
            SINGLE_RULES: Dict[str, Optional[str]] = {
                # First Order
                "InverseVariance":"InVar",
                "LargeDependenceLowGrayLevelEmphasis": "LDLGE",
                "LowGrayLevelEmphasis":"LGE",
                "GrayLevelNonUniformityNormalized":"GLNUNorm",
                "LongRunHighGrayLevelEmphasis":"LRHGE",
                "LowGrayLevelZoneEmphasis":"LGZE",
                "DependenceNonUniformityNormalized":"DNUNorm",
                "QuantileCoeffDispersion":"QuantCoefDisp",
                "InterquartileRange":"IQR",
                "Idmn":"IDMN",
                "e5e5e5":"",
                "SmallDependenceEmphasis":"SDE",
                "ShortRunEmphasis":"SRE",
                "DependenceVariance":"DVariance",
                "lde":"LDE",
                "wavelet-HHH": "WaveletHHH",
                "wavelet-HHL": "WaveletHHL",
                "wavelet-HLH": "WaveletHLH",
                "wavelet-LLH": "WaveletLLH",
                "wavelet-HLL": "WaveletHLL",
                "wavelet-LHH": "WaveletLHH",
                "wavelet-hhh": "WaveletHHH",
                "wavelet-hhl": "WaveletHHL",
                "wavelet-hlh": "WaveletHLH",
                "wavelet-llh": "WaveletLLH",
                "wavelet-hll": "WaveletHLL",
                "wavelet-lhh": "WaveletLHH",
                "wavelet-LLL": "WaveletLLL",
                "wavelet-LHL": "WaveletLHL",
                "exponential": "Exponential",
                "lbp-2d": "LBP2D",
                "gabor": "Gabor",
                "laws": "Laws",
                "l5s5e5": "",
                "7":"",
                "invar":"InVar",
                "energy":"Energy",
                "delta":"Delta",
                "square":"Square",
                "logarithm":"Log",
                "squareroot":"SquareRoot",
                "contrast":"Contrast",
                "gradient":"Gradient",
            }



        def _tokenize(name: str) -> List[str]:
            return [t for t in re.split(r"_+", name.strip()) if t]

        def _apply_multi(tokens: List[str], i: int) -> Optional[Tuple[int, str]]:
            best = None
            for key_tokens, repl in MULTI_RULES:
                L = len(key_tokens)
                if i + L <= len(tokens) and tuple(tokens[i:i+L]) == key_tokens:
                    if best is None or L > best[0]:
                        best = (L, repl)
            return best

        def _rename_mirp_like(name: str) -> str:
            toks = _tokenize(name)
            out: List[str] = []
            i = 0
            while i < len(toks):
                m = _apply_multi(toks, i)
                if m:
                    L, repl = m
                    if repl:
                        out.append(repl)
                    i += L
                    continue
                t = toks[i]
                repl = SINGLE_RULES.get(t, None)
                if repl is None:
                    out.append(t)
                elif repl != "":
                    out.append(repl)
                i += 1

            s = "_".join(out)
            if drop_gradient:
                s = re.sub(r"(_)?GradientMagnitude(?=(_|$))", "", s)
            s = re.sub(r"_+", "_", s).strip("_")
            s = re.sub(r"(?:_)?margin$", "_margin", s, flags=re.IGNORECASE)
            s = re.sub(r"margin$", "margin", s)
            return s

        df_renamed = df.copy()
        clinical_norm = {_norm(x) for x in clinical_classes}
        has_gate = feature_class_row in df.index
        # has_gate = feature_class_row in df_renamed.index

        old2new: Dict[str, str] = {}

        for col in df_renamed.columns:
            is_clinical = False
            if has_gate:
                cls_val = df_renamed.loc[feature_class_row, col]
                is_clinical = _norm(cls_val) in clinical_norm
            # if clinical → keep EXACT original name (no formatting at all)
            old2new[col] = col if is_clinical else _rename_mirp_like(col)

        df_renamed = df_renamed.rename(columns=old2new)

        mapping_df = (
            pd.DataFrame({"old": list(old2new.keys()), "new": list(old2new.values())})
            .assign(changed=lambda x: x["old"] != x["new"])
            .sort_values(["changed", "old"], ascending=[False, True])
            .reset_index(drop=True)
        )

        return df_renamed, mapping_df


    def generate_feature_overview_map(self, 
                                      features_df, 
                                      extractor, 
                                      out_folder, 
                                      additional_params=None, 
                                      renamed_prediction_label: str=None, 
                                      prediction_label_meaning: tuple=None,
                                      title:str=None):
        """
        Generate an feature overview plot
        :param features_df: radiomics features
        :param extractor: Name of the extractor (PyRadiomics or MIRP)
        :param out_folder: Folder where the plot should be saved
        :param additional_params: pd.DataFrame with additional clinical Parameters of the patients 
        :param renamed_prediction_label (str): if you want to show a sematic prediction label
        :param prediction_label_meaning (tuple): (str for 0 means, string for 1 means) 
        :param title (str): title of the plot if you like a specific title
        """
        
        dropping_features = []

        features = features_df.copy()

        clinical_features = []

        if out_folder.endswith("/"):
            out_folder = out_folder[:-1]
            
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            
        mirp_feature_class= {"morph":"morphological",
                                            "loc":"local_intensity",
                                            "stat":"intensity-based_statistics",
                                            "ivh":"intensity-volume_histogram",
                                            "ih":"intensity_histogram",
                                            "cm":"grey_level_co-occurrence_matrix",
                                            "rlm":"grey_level_run_length_matrix",
                                            "szm":"grey_level_size_zone_matrix",
                                            "dzm":"grey_level_distance_zone_matrix",
                                            "ngt":"neighbourhood_grey_tone_difference_matrix",
                                            "ngl":"neighbouring_grey_level_dependence_matrix",
                                            "diag":"diagnostic",
                                            "img":"img_data"}
        
        py_feature_class = {"shape":"morphological",
                                        "loc":"local_intensity",
                                        "stat":"intensity-based_statistics",
                                        "ivh":"intensity-volume_histogram",
                                        "ih":"intensity_histogram",
                                        "glcm":"grey_level_co-occurrence_matrix",
                                        "glrlm":"grey_level_run_length_matrix",
                                        "glszm":"grey_level_size_zone_matrix",
                                        "gldm":"grey_level_distance_zone_matrix",
                                        "ngtdm":"neighbourhood_grey_tone_difference_matrix",
                                        "diagnostics":"diagnostics",
                                        "firstorder":"firstorder"
                                        }
        
        # config feature space
        if "ROI_Label" in features.columns:
            features = features.drop(["ROI_Label"], axis=1)
        if "ID" in features.columns:
            features = features.drop(["ID"], axis=1)
            
        features.columns = features.columns.str.replace(r'_zscore', '')
        features.columns = features.columns.str.replace(r'original_', '')
        
        if os.path.isfile(self.out_path + extractor + "/filtered_features/" + extractor + "_General_Feature_Profile.csv"):
            # get feature profile
            feature_profile = pd.read_csv(self.out_path + extractor + "/filtered_features/" + extractor + "_General_Feature_Profile.csv", index_col = 0)
            
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("neighbourhood_grey_tone_difference_matrix", "NGTDM")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("grey_level_co-occurrence_matrix", "GLCM")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("grey_level_distance_zone_matrix", "GLDZM")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("grey_level_run_length_matrix", "GLRLM")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("grey_level_size_zone_matrix", "GLSZM")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("neighbouring_grey_level_dependence_matrix", "NGLDM")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("intensity-volume_histogram", "IVH")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("intensity-based_statistics", "IS")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("intensity_histogram", "IH")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("morphological", "Morphological")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("diagnostic", "Diagnostic")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("diagnostics", "Diagnostic")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("firstorder", "Firstorder")
            feature_profile["Feature_Class"] = feature_profile["Feature_Class"].replace("local_intensity", "LI")

            # Format Features for getting radiomics feature profile
            for feat in feature_profile["Name"]:
                new_feat = feat.replace(r'_zscore', '')
                new_feat = new_feat.replace(r'original_', '')
                
                feature_profile.loc[feat == feature_profile["Name"], "Name"] = new_feat

            feature_configs = pd.DataFrame()
            for feature in features.columns:
                included = False
                feature_name = None

                for i,r in feature_profile.iterrows():
                    if r["Name"] in feature:
                        feature_name = r["Name"]
                        feature_class = r["Feature_Class"]
                        feature_config_tmp = pd.DataFrame({feature : [feature_class]})
                        feature_configs = pd.concat([feature_configs, feature_config_tmp], axis=1)
                        if feature_class == "Clinical_feature":
                            clinical_features.append(feature)
                        break

                # get non radiomics features in the plot
                if feature_name is None:
                    print(f"Feature {feature} not present in Radiomics Feature space!")
                    
                    self.error.warning("Feature {feature} not present in Radiomics Feature space!")
                    if feature != "Prediction_Label":
                        try:
                            features[feature].astype(float)
                            feature_config_tmp = pd.DataFrame({feature : ["Clinical feature"]})
                            feature_configs = pd.concat([feature_configs, feature_config_tmp], axis=1)
                            clinical_features.append(feature)
                        except (ValueError, TypeError):
                            # Skip columns that cannot be converted to float
                            self.error.warning(f"{feature} does not contain (only) numbers!")
                            print(f"{feature} does not contain (only) numbers!")
                            dropping_features.append(feature)

            if len(dropping_features) > 0:
                features = features.drop(columns=dropping_features)        
            
            feature_configs["config"] = ["feature_class"]
            feature_configs.set_index("config", inplace=True) 

            features = pd.concat([feature_configs, features])  

            if extractor == "PyRadiomics":
                feature_class_dict = py_feature_class
            elif extractor == "MIRP":
                feature_class_dict = mirp_feature_class
                
            # get non identified feature classes
            nan_features = features.loc["feature_class", features.iloc[0].isnull()].index.tolist()
            
            for nan_feature_class in nan_features:
                found=False
                for feat_class in feature_class_dict:
                    if feat_class in nan_feature_class:
                        found=True
                        features.loc["feature_class", nan_feature_class] = feature_class_dict[feat_class]
                if not found:
                    if nan_feature_class != "Prediction_Label":
                        self.error.warning("Could not find feature class for {} !".format(nan_feature_class))
                        print("Could not find feature class for {} !".format(nan_feature_class))
   
            try:
                features = features.sort_values(by=["Prediction_Label"])
            except:
                features["Prediction_Label"] = features["Prediction_Label"].astype(str)
                features = features.sort_values(by=["Prediction_Label"])

            label = features["Prediction_Label"].drop("feature_class")
            features = features.drop(["Prediction_Label"], axis=1)
            if "ID.1" in features.columns:
                features = features.drop(columns=["ID.1"])

            features.columns = [col.split('_', 1)[1] if ('_' in col) and (col not in clinical_features) else col for col in features.columns]
            # replace peritumoral by surrounding as it is not always a tumor
            features.columns = [col.replace('_peritumoral', '_surrounding') for col in features.columns]
            
            features, non_normalied_features = self.normalize_if_needed(features, clinical_features)
            print(f"Normalized non-normalized features: {non_normalied_features}!")
            
            if not renamed_prediction_label is None:
                label = label.rename(renamed_prediction_label)

            if title is None:
                title = extractor + " Feature Clustering"
            
            # if extractor == "MIRP":
                # config feature names:
                # features.columns = features.columns.str.replace('_w25.0', '')
                # features.columns = features.columns.str.replace('inv_var_d1_3d_v_mrg_fbs', 'invar_d1_3d_mrg_fbs')
                # return features
                # Make feature more human readable
                # features.columns = [clean_mirp_feature_name(c) for c in features.columns]
                # features.columns = [rename_mirp_feature(c) for c in features.columns]
                # Rename all non-clinical features; keep mapping table

            #cons = ConsensusFeatureFormatter(df=features.copy(),
            #                    feature_cols=None, 
            #                    extractor=extractor,
            #                    output_path=out_folder,
            #                    generate_feature_profile_plot=False,
            #                    run_id=self.RunID
            #                )
            #report, features = cons.run(
            #                    title=self.RunID,
            #                    return_dataframe=True,
            #                    rename_mode="rename"           # or "multiindex" or "add_columns"
            #                )


            features, rename_map = self.rename_radiomics_columns(
                features.copy(),
                feature_class_row="feature_class",
                # clinical_class_name="Clinical_features",
                drop_gradient=False,
                extractor=extractor,
                return_mapping=True,
            )
            #features = result[0]
            #rename_map = result[1]
            print("Renamed",features.columns)
            # config feature names:
            features.columns = features.columns.str.replace('_surrounding', '_margin')
            # return features
            self.plot_custom_radiomics_heatmap(features, 
                                            label.astype(float), 
                                            additional_params=additional_params, 
                                            title=title, 
                                            save_path=out_folder + "/" + extractor + "_" + str(self.RunID) + "_Radiomics_Feature_Clustering.png",
                                            no_feature_names=False,
                                            prediction_label_meaning=prediction_label_meaning)
        else:
            self.error.warning(f"Could not find the General Feature Profile for Feature class checkup! File not found: {self.out_path + extractor + '/filtered_features/' + extractor + '_General_Feature_Profile.csv'}")
            print(f"Warning: Could not find the General Feature Profile for Feature class checkup! File not found: {self.out_path + extractor + '/filtered_features/' + extractor + '_General_Feature_Profile.csv'}")
    
    
    def generate_correlation_map(self, df: pd.DataFrame, out_folder: str, name: str):
        """
        Generate Correlation Matrix for Data.
        :param df: Data with parameters to show correlation matrix
        :param out_folder: Path to outfolder to save the graph
        :param name: string as name of file and title of plot
        """

        not_numeric_features = []
        for feature in df.columns:
            try:
                df[feature] = pd.to_numeric(df[feature], errors='raise')
            except:
                not_numeric_features.append(feature)

        if len(not_numeric_features) > 0:
            df = self.feature_encoding(df=df, features=not_numeric_features)

        #    df = df.dropna(subset=[feature])

        corr = df.corr()
        
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        
        ax.set_title(name)
        figure = ax.get_figure()
        figure.savefig(out_folder + name + "_" + 'Correlation_Matrix.png', dpi=400, bbox_inches='tight')
        plt.close(figure)

    def feature_encoding(self, df: pd.DataFrame, features: list = None, print_=False):
        """
        Perform Feature Encoding
        :param df: Data with strings as vales inside
        :param features: List of features
        :return df: Data with only floats
        """
        
        # if no specific feature. go through the entire data
        if features is None:
            
            # Convert features to numerical features
            for feature in tqdm(df.columns, desc="Feature Encoding"):
                convertable = False
                # check if feature is categorical
                for value in df[feature]:
                    try:
                        float(value)
                        convertable = True
                    except:
                        convertable = False
                        break

                if convertable:
                    df[feature].astype(float)
                else:
                    encoding = 0.0
                    # encode values that are categorical or can not be converted
                    print("Encoding feature", feature)
                    for feat in set(df[feature]): 
                        if print_:
                            self.logger.info("Encoding: " + str(feat) + " to " + str(encoding) )
                            print("Encoding:", feat, "to", encoding )
                        # for i, row in clinical_df.iterrows():
                        df.loc[df[feature] == feat, feature] = str(encoding)
                        encoding += 1.0

                    df[feature].astype(float)
        else:
            for feature in tqdm(features, desc="Feature Encoding"):
                convertable = False
                # check if feature is categorical
                for value in df[feature]:
                    try:
                        float(value)
                        convertable = True
                    except:
                        convertable = False
                        break

                if convertable:
                    df[feature].astype(float)
                else:
                    encoding = 0.0
                    # encode values that are categorical or can not be converted
                    for feat in set(df[feature]):
                        if print_:
                            self.logger.info("Encoding: " + str(feat) + " to " + str(encoding))
                            print("Encoding:", feat, "to", encoding)
                        # for i, row in clinical_df.iterrows():
                        df.loc[df[feature] == feat, feature] = str(encoding)
                        encoding += 1.0

                    df[feature].astype(float)
                    
        return df

    def configure_clinical_data(self, clinical_df: pd.DataFrame = None):
        """
        Get the completeness of the data and fill nans of features
        :param clinical_df: (pd.DataFrame) Table containing clinical information
        :return clinical_df: (pd.DataFrame) Table containing imputated and cleaned clinical information
        """

        self.logger.info("Configure clinical Data.")
        self.logger.info("Identified clinical parameter: " + str(clinical_df.columns))

        if clinical_df is None:
            self.error.error("No clinical data provided! Check for clinical data!")
            raise ValueError("No clinical data provided! Check for clinical data!")

        plt.rcParams["figure.figsize"] = (20, 10)
        sns.heatmap(clinical_df.isnull(), cbar=False)
        plt.savefig(self.out_path + "preprocessed_data/plots/clinical_data_completeness.png",
                    bbox_inches='tight',
                    dpi=200)
        plt.close()

        # Config Imputator
        ## Use most frequent Imputator
        simple_imp = SimpleImputer(strategy="most_frequent")
        ## Use KNN Imputator
        knn_imp = KNNImputer(n_neighbors=2)

        # get features with nan
        features_with_na = [features for features in clinical_df.columns if clinical_df[features].isnull().sum() > 0]
        for feature in features_with_na:
            self.logger.info(feature + " " + str(np.round(np.round(clinical_df[feature].isnull().mean(), 4) * 100, 2)) + " % missing values")

            # Simple Imputation for categorical parameter
            if clinical_df[feature].dtype.name == 'category':
                clinical_df[feature] = clinical_df[feature].astype("category")
                clinical_df[feature] = simple_imp.fit_transform(pd.DataFrame(clinical_df[feature]))  # Need to convert to pd.DataFrame

        # Feature encoding if feature is not numeric
        not_numeric_features = []
        for feature in clinical_df.columns:
            try:
                clinical_df[feature] = pd.to_numeric(clinical_df[feature], errors='raise')
            except:
                not_numeric_features.append(feature)

        if len(not_numeric_features) > 0:
            clinical_df = self.feature_encoding(df=clinical_df, features=not_numeric_features)

        # get features with nan
        features_with_na = [features for features in clinical_df.columns if clinical_df[features].isnull().sum() > 0]

        # KNN Imputation for the numerical parameter
        for nan_feature in features_with_na:
            clinical_df[nan_feature] = knn_imp.fit_transform(clinical_df[nan_feature].values.reshape(-1,1))

        plt.rcParams["figure.figsize"] = (20, 10)
        sns.heatmap(clinical_df.isnull(), cbar=False)
        plt.savefig(self.out_path + "preprocessed_data/plots/clinical_data_completeness_after_imputation.png",
                    bbox_inches='tight',
                    dpi=200)
        plt.close()

        return clinical_df
    
    @staticmethod
    def normalize_id_length_end(df, id_col="ID"):
        """
        Normalizes the ID column by padding shorter IDs with trailing zeros to match the longest ID length.

        Parameters:
        - df: pandas DataFrame
        - id_col: Name of the column containing IDs (default: "ID")

        Returns:
        - DataFrame with normalized ID column.
        """
        # Convert IDs to string
        df[id_col] = df[id_col].astype(str)

        # Find the longest ID length
        max_length = df[id_col].apply(len).max()

        # Pad IDs with trailing zeros
        df[id_col] = df[id_col].apply(lambda x: x.zfill(max_length))

        return df

    def validate_and_prefix_id(self, df, id_col="ID", prefix="ID-"):
        """
        Checks if the ID column is numeric and ensures it has a given prefix.

        Parameters:
        - df: pandas DataFrame
        - id_col: Name of the column containing IDs (default: "ID")
        - prefix: Prefix to ensure for each ID (default: "ID-")

        Returns:
        - DataFrame with IDs converted to numeric (if possible) and prefixed.
        - A list of invalid IDs (non-numeric).
        """

        # Ensure the column exists
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in DataFrame.")

        # Convert IDs to string
        df[id_col] = df[id_col].astype(str)

        # Identify numeric IDs (those that contain only digits)
        numeric_ids = df[df[id_col].str.isnumeric()][id_col].tolist()

        # Convert numeric IDs by adding the prefix
        df[id_col] = df[id_col].apply(lambda x: f"{prefix}{x}" if x.isnumeric() else x)

        return df

    def generate_unique_id(self, df, id_col="ID", timepoint_col="Timepoint"):
        """
        Ensures unique values in the specified ID column by:
        - Removing the underscore '_' character
        - Only adding a numerical suffix to truly duplicate IDs (same ID & same Timepoint)
        
        Parameters:
        - df: pandas DataFrame
        - id_col: Name of the column containing IDs (default: "ID")
        - timepoint_col: Name of the column containing Timepoints (default: "Timepoint")
        
        Returns:
        - DataFrame with unique IDs in the specified column.
        """

        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in DataFrame.")

        # Remove underscores from IDs
        df[id_col] = df[id_col].astype(str).str.replace("_", "", regex=False)

        # If Timepoint column exists, consider it in uniqueness check
        if timepoint_col in df.columns:
            # Ensure Timepoint is numeric
            df[timepoint_col] = pd.to_numeric(df[timepoint_col], errors='coerce')

            # Count occurrences of (ID, Timepoint) pairs
            id_timepoint_counts = df.groupby([id_col, timepoint_col]).size()

            # Identify truly duplicate (ID, Timepoint) pairs
            duplicate_id_timepoints = id_timepoint_counts[id_timepoint_counts > 1].index.tolist()
        else:
            # If no Timepoint column, count just based on ID
            id_counts = df.groupby(id_col).size()
            duplicate_id_timepoints = id_counts[id_counts > 1].index.tolist()

        # Dictionary to track occurrences
        seen = {}
        unique_ids = []

        for idx, row in df.iterrows():
            id_value = row[id_col]
            timepoint_value = row[timepoint_col] if timepoint_col in df.columns else None

            if (id_value, timepoint_value) in duplicate_id_timepoints:
                # If this ID-Timepoint pair was seen before, increment its counter
                if (id_value, timepoint_value) in seen:
                    seen[(id_value, timepoint_value)] += 1
                    new_id = f"{id_value}{seen[(id_value, timepoint_value)]}"  # Append a numeric suffix
                else:
                    seen[(id_value, timepoint_value)] = 0
                    new_id = id_value  # Keep original if first occurrence
            else:
                new_id = id_value  # Keep original if it's unique

            unique_ids.append(new_id)

        # Assign back to the DataFrame
        df[id_col] = unique_ids

        return df

    def append_timepoint(self, df, id_col='ID', time_col='Timepoint'):
        """
        Appends the Timepoint value to the ID column with a '-' separator.

        Parameters:
        df (pd.DataFrame): The DataFrame containing ID and Timepoint columns.
        id_col (str): The name of the ID column. Default is 'ID'.
        time_col (str): The name of the Timepoint column. Default is 'Timepoint'.

        Returns:
        pd.DataFrame: A DataFrame with modified ID values.
        """

        df[id_col] = df[id_col].astype(str) + '-' + df[time_col].astype(int).astype(str)

        return df

    def clean_filename(self, full_path):
        """
        Remove patterns like "_resampled" and "_multilabel" from the filename.
        :param full_path: The full path of the file.
        :return: The cleaned filename.
        """

        dir_name = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        cleaned_filename = filename.replace("_resampled", "").replace("_multilabel", "")
        return os.path.join(dir_name, cleaned_filename)

    def extract_id_from_filename(self, filename):
        """
        Extracts the ID from the filename if it follows the pattern 'ID_filename.ext'.

        :param filename: Name of the file.
        :return: Extracted ID or None if no match.
        """
        parts = filename.split("_")
        return parts[0] if parts else None

    def validate_and_correct_pairs(self, df, image_folder, mask_folder):
        """
        Validate and correct image-mask pairing in the DataFrame.

        :param df: DataFrame with 'ID', 'Image', and 'Mask' columns.
        :param image_folder: Directory containing image files.
        :param mask_folder: Directory containing mask files.
        :return: Updated DataFrame with corrected paths.
        """
        self.logger.info("Validating and correcting image-mask pairs...")
        print("Validating and correcting image-mask pairs...")

        # Get all files in the folders for quick lookup

        image_files_paths = glob.glob(image_folder + "*.nii.gz")
        mask_files_paths =  glob.glob(mask_folder + "*.nii.gz")

        image_files = {os.path.basename(f): f for f in image_files_paths}
        mask_files = {os.path.basename(f): f for f in mask_files_paths}

        def validate_image_mask_pair(idx, row):
            try:
                expected_id = str(row["ID"])

                # Extract IDs from filenames
                image_filename = os.path.basename(row["Image"])
                mask_filename = os.path.basename(row["Mask"])

                extracted_image_id = self.extract_id_from_filename(image_filename)
                extracted_mask_id = self.extract_id_from_filename(mask_filename)

                correct_image = extracted_image_id == expected_id and image_folder in row["Image"]
                correct_mask = extracted_mask_id == expected_id and mask_folder in row["Mask"]

                # Find correct image if incorrect
                if not correct_image:
                    matching_images = [
                        file for file in image_files.keys() if self.extract_id_from_filename(file) == expected_id
                    ]
                    if matching_images:
                        row["Image"] = os.path.join(image_folder, matching_images[0])

                # Find correct mask if incorrect
                if not correct_mask:
                    matching_masks = [
                        file for file in mask_files.keys() if self.extract_id_from_filename(file) == expected_id
                    ]
                    if matching_masks:
                        row["Mask"] = os.path.join(mask_folder, matching_masks[0])

                return idx, row
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                return idx, row  # Return unmodified row in case of error

        # Multithreading with ThreadPoolExecutor
        updated_rows = []
        with ThreadPoolExecutor(self.n_cpu) as executor:
            futures = {executor.submit(validate_image_mask_pair, idx, row): idx for idx, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df), desc="Checking image-mask pairs", unit="sample"):
                try:
                    updated_rows.append(future.result())
                except Exception as e:
                    self.logger.error(f"Thread failed with error: {e}")

        # Update DataFrame
        for idx, updated_row in updated_rows:
            df.loc[idx] = updated_row

        return df

    import numpy as np

    def process_delta_clinical_data(self, df:pd.DataFrame, output_csv_path:str):
        """
        Processes clinical timepoint data for each patient and computes delta values where appropriate.

        This function:
        - Parses patient ID and timepoint from the 'ID' column.
        - Groups the data by patient.
        - For each patient:
            - Calculates cumulative deltas for numeric features:
            (T1 - T0), then (T2 - (T1 - T0)), and so on.
            - Concatenates string feature paths from first and last timepoints if they differ.
            - Retains the original value if a feature does not vary.
            - Reports how many timepoints are found for each patient.
        - Generates one row per patient.
        - Labels each row with a patient-specific identifier in the format 'PatientID-delta-startTP-endTP'.
        - Outputs the final processed DataFrame to a CSV file.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing timepoint-level clinical data.

        output_csv_path : str
            Path to save the processed CSV file where each row corresponds to a single patient.

        Returns:
        -------
        writes file to output_csv_path
        """

        self.logger.info("Processing clinical data for delta calculations ...")

        # Extract PatientID and Timepoint from ID
        df[['PatientID', 'Timepoint']] = df['ID'].str.rsplit('-', n=1, expand=True)
        df['Timepoint'] = df['Timepoint'].astype(int)

        # Drop original ID column if needed
        df = df.drop(columns=['ID'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Detect feature types
        feature_cols = [col for col in df.columns if col not in ['PatientID', 'Timepoint']]
        string_features = df[feature_cols].select_dtypes(include='object').columns.tolist()
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        result_rows = []

        # Group by Patient
        for patient_id, group in df.groupby('PatientID'):
            group = group.sort_values('Timepoint')
            timepoints = group['Timepoint'].values
            first_tp = timepoints[0]
            last_tp = timepoints[-1]
            patient_label = f"{patient_id}-delta-{first_tp}-{last_tp}"

            result_row = {}

            for feat in string_features:
                vals = group[feat].values
                if len(np.unique(vals)) == 1:
                    result_row[feat] = vals[0]
                else:
                    result_row[feat] = f"{vals[0]} - {vals[-1]}"

            # Handle numeric features with progressive delta logic
            for feat in numeric_features:
                vals = group[feat].values
                if len(vals) == 1 or np.all(vals == vals[0]):
                    result_row[feat] = vals[0]
                else:
                    delta = vals[1] - vals[0]
                    for i in range(2, len(vals)):
                        delta = vals[i] - delta
                    result_row[feat] = delta

            # Set label column
            result_row["ID"] = patient_label  # acts as index
            result_rows.append(result_row)

        # Convert to DataFrame
        final_df = pd.DataFrame(result_rows)

        # Drop Timepoint column if present
        if "Timepoint" in final_df.columns:
            final_df = final_df.drop(['Timepoint'], axis=1)

        # Save to CSV
        final_df.to_csv(output_csv_path)

        return final_df

    def check_input_csv_format(self):
        """
        Check the format of the input CSV. Image and Msk files should have a unique name!
        Copy reformated files if input_reformat set to true
        """

        needed_columns = ["ID", "Image", "Mask", self.Prediction_Label,
                          "Modality", "ROI_Label", "Image_Transformation",
                          "Mask_Transformation", "Timepoint", "Rater"]

        if self.data is not None:

            reformat = False
            self.data.drop(self.data.columns[self.data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
            
            # Make ID unique
            self.data = RPTK.normalize_id_length_end(df=self.data.copy(), id_col="ID")

            # if you want to calculate for every segmentation if false all samples with different segmentation but the same Image are having the same ID and will be caluclated as mean for features
            if self.calculate_for_every_segmentation:
                # check if ID is unique
                self.data = self.generate_unique_id(df=self.data.copy(), id_col="ID")
            
            # make ID to string and add prefix
            self.data = self.validate_and_prefix_id(df=self.data.copy(), id_col="ID", prefix="ID-")

            if "Timepoint" not in self.data.columns:
                if self.delta:
                    print("Timepoint is not in the data! Please add Timepoint to the data!")
                    self.error.error("Timepoint is not in the data! Please add Timepoint to the data!")
                    raise ValueError("Timepoint is not in the data! Please add Timepoint to the data!")
            else:
                if self.data["Timepoint"].isnull().sum() > 0:
                    if self.data["Timepoint"].isnull().sum() == len(self.data):
                        self.error.warning("No Timepoints in the data but parameter is given! Remove Timepoint column!")
                        self.data.drop(['Timepoint'], axis = 1, inplace = True) 
                    else:
                        self.error.error("Timepoint contains missing values! Please check the data!")
                        raise ValueError("Timepoint contains missing values! Please check the data!")
                else:
                    self.data = self.append_timepoint(self.data.copy(), id_col='ID', time_col='Timepoint')

            # check if prediction label are strings
            prediction_type = self.data[self.Prediction_Label].dtype
            if prediction_type == 'object':     
                self.logger.info(self.Prediction_Label + " contains string! Perform encoding ...")
                print(self.Prediction_Label,"contains string! Perform encoding ...")
                self.data = self.feature_encoding(df=self.data.copy(), features=[self.Prediction_Label], print_=True)

            # collect clinical parameter
            for col in tqdm(self.data.columns, desc="Check for clinical parameter", unit="parameter"):
                if col not in needed_columns:
                    if ("Unnamed" not in col) or ("unnamed" not in col):
                        self.clinical_df[col] = self.data[col]
                        self.logger.info("Found clinical feature: " + col)
                        self.data.drop(columns=col, inplace=True)

            # check for nan values are in Image or Mask
            if self.data["Image"].isnull().sum() > 0:
                self.error.error("Image contains missing values! Please check the data!")
                raise ValueError("Image contains missing values! Please check the data!")

            if self.data["Mask"].isnull().sum() > 0:
                self.error.error("Mask contains missing values! Please check the data!")
                raise ValueError("Mask contains missing values! Please check the data!")

            # cleaning filename with used str for img and mask
            #self.data["Image"] = self.data["Image"].apply(self.clean_filename)
            #self.data["Mask"] = self.data["Mask"].apply(self.clean_filename)

            if is_numeric_dtype(self.data["ID"]):
                self.data["ID"] = self.data["ID"].astype(str)

            if len(self.clinical_df) > 0:
                self.clinical_df["ID"] = self.data["ID"]
                self.clinical_df["Image"] = self.data["Image"]
                self.clinical_df["Mask"] = self.data["Mask"]

                # Configure clinical features
                self.clinical_df = self.configure_clinical_data(clinical_df=self.clinical_df)

                # check for correlation in clinical data
                self.generate_correlation_map(df=self.clinical_df,
                                              out_folder=self.out_path + "preprocessed_data/plots/",
                                              name="Clinical_Data")

                # include original values from configuration
                self.clinical_df["ID"] = self.data["ID"].copy()
                self.clinical_df["Image"] = self.data["Image"].copy()
                self.clinical_df["Mask"] = self.data["Mask"].copy()

            reformat_path = self.out_path + "input_reformated/reformated_input.csv"

            if not os.path.exists(reformat_path):
                reformat = self.generate_unique_format()
            else:
                self.path2confCSV = reformat_path
                if os.path.getsize(reformat_path) == 0:
                    reformat = self.generate_unique_format()
                else:
                    self.data = pd.read_csv(reformat_path, index_col=0)

                    # update clinical data
                    self.clinical_df["ID"] = self.data["ID"].copy()
                    self.clinical_df["Image"] = self.data["Image"].copy()
                    self.clinical_df["Mask"] = self.data["Mask"].copy()

            self.logger.info("Check image input format.")

            # check converted files 
            self.converted_files = glob.glob(self.out_path + "converted/*.nii.gz")

            if "ROI_Label" not in self.data.columns:
                self.logger.info("No ROI_Label found in data. Assuming ROI_Label in Segmentation is always 1.")
                self.data["ROI_Label"] = 1

            # convert files with any other format than nifti if they are in the csv
            result = pd.DataFrame()
            out = []
            with tqdm(total=len(self.data), desc='Check image input format') as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                    # futures = {executor.submit(func, row): row for row in chunk}
                    # for future in concurrent.futures.as_completed(futures):
                    for results in executor.map(self.check_input_format, [row for i, row in self.data.copy().iterrows()], chunksize=self.n_cpu):
                        pbar.update(1)
                        out.append(results)
                        
                        # result = pd.concat([result, results.to_frame().T], ignore_index=True)
            result = pd.concat(out)

            # possibly multiple times the same image and mask included
            result = result.drop_duplicates(subset=['Image', 'Mask'])

            # drop wrong samples
            result[~result["Mask"].str.contains(r"input_reformated/img/", na=False, regex=True)]
            result[~result["Image"].str.contains(r"input_reformated/msk/", na=False, regex=True)]

            # check if all files in the out folder are in the result
            self.data = self.data.drop_duplicates(subset=['Image', 'Mask'])

            # drop wrong samples
            self.data[~self.data["Mask"].str.contains(r"input_reformated/img/", na=False, regex=True)]
            self.data[~self.data["Image"].str.contains(r"input_reformated/msk/", na=False, regex=True)]
            
            # I hate myself TODO:
            filenames = glob.glob(self.out_path + "input_reformated/msk/*.nii.gz")

            # Extract unique IDs from filenames (before '_')
            file_ids = []
            for file in filenames:
                file_ids.append((os.path.basename(file).split("_")[0], file))
            
            new_rows = []
            for id in file_ids:
                if id[1] not in self.data["Mask"].to_list():
                    if id[0] not in self.data["ID"].to_list():
                        print("ID not found in data: ", id[0])
                    else:
                        # Create new rows for missing IDs
                        new_row = self.data[self.data["ID"]==id[0]].iloc[0].copy()
                        new_row["Mask"] = id[1]
                        new_rows.append(new_row)

            self.data = pd.concat([self.data, pd.DataFrame(new_rows)], ignore_index=True)
            result = pd.concat([result, pd.DataFrame(new_rows)], ignore_index=True)

            self.data = self.data.drop_duplicates(subset=['Image', 'Mask'])
            result = result.drop_duplicates(subset=['Image', 'Mask'])
            result = result.sort_values(by=['ID'], ascending=True)

            # if the shape is identical: replace self.data
            if result.shape == self.data.shape:
                self.data = result
            else:
                for i, row in result.iterrows():
                    if row["Image"] not in self.data["Image"].to_list():
                        print("Image not found in data: ", row["Image"])
                    if row["Mask"] not in self.data["Mask"].to_list():
                        print("Mask not found in data: ", row["Mask"])
                
                for i, row in self.data.iterrows():
                    if row["Image"] not in result["Image"].to_list():
                        print("Image not found in result: ", row["Image"])
                    if row["Mask"] not in result["Mask"].to_list():
                        print("Mask not found in result: ", row["Mask"])

                self.error.warning("File reformatting error! Not able to convert files to nifti!")
                raise ValueError("File reformatting error! Not able to convert files to nifti!")
                
            # if files are converted and reloaded
            if not self.need_to_convert:
                for img in self.data["Image"].to_list():
                    if "/converted/" in img:
                        self.need_to_convert = True
                        break

                for msk in self.data["Mask"].to_list():
                    if "/converted/" in msk:
                        self.need_to_convert = True
                        break
                

            if self.need_to_convert:
                convert_dir_path = self.out_path + "converted/"

                if not os.path.exists(convert_dir_path):
                    Path(convert_dir_path).mkdir(parents=True, exist_ok=True)

                self.path2confCSV = convert_dir_path + "converted_data.csv"
                # check if converted files are also in the file
                converted_files = glob.glob(convert_dir_path + "*.nii.gz")

                for msk in tqdm(self.data.copy()["Mask"].to_list(), desc="Include converted mask files"):
                    if "/converted/" not in msk:
                        if not os.path.basename(msk).endswith(".nii.gz"):
                            format = os.path.basename(msk).split('.')[-1]
                        else:
                            # if only label got reformated
                            format = ".nii.gz"
                        for conv_file in converted_files:
                            if os.path.basename(conv_file)[:-len(".nii.gz")] == os.path.basename(msk)[:-len(str(format))]:
                                self.data.loc[self.data["Mask"] == msk, "Mask"] = conv_file
                                if self.data.loc[self.data["Mask"] == msk, "ROI_Label"].values[0] != 1:
                                    self.data.loc[self.data["Mask"] == msk, "ROI_Label"] = 1
                                break
                
                for img in tqdm(self.data.copy()["Image"].to_list(), desc="Include converted image files"):
                    if "/converted/" not in img:
                        format = os.path.basename(img).split('.')[-1]
                        for conv_file in converted_files:
                            if os.path.basename(conv_file)[:-len("nii.gz")] == os.path.basename(img)[:-len(str(format))]:
                                self.data.loc[self.data["Image"] == img, "Image"] = conv_file
                                break

                # remove non preprocessed
                self.data = self.validate_sample_masks(df=self.data)

                self.data.to_csv(self.path2confCSV)


            if reformat:

                self.logger.info("Use reformated csv for further processing!")
                print("Use reformated csv for further processing!")
                self.path2confCSV = reformat_path
                self.logger.info("Write reformated csv to: " + str(reformat_path))

                # check correct images and masks are assigned correctly to each other
                self.data = self.validate_and_correct_pairs(df=self.data.copy(), image_folder= self.out_path + "input_reformated/img/", mask_folder= self.out_path + "input_reformated/msk/")

                 # check if all files in the out folder are in the result
                self.data = self.data.drop_duplicates(subset=['Image', 'Mask'])

                # drop if image files got into mask column and vice versa
                self.data[~self.data["Mask"].str.contains(r"input_reformated/img/", na=False, regex=True)]
                self.data[~self.data["Image"].str.contains(r"input_reformated/msk/", na=False, regex=True)]
                
                # If multiple segmentations are included
                filenames = glob.glob(self.out_path + "input_reformated/msk/*.nii.gz")

                # Extract IDs from filenames (before '_')
                file_ids = []
                for file in filenames:
                    file_ids.append((os.path.basename(file).split("_")[0], file))
                
                # check if all reformated masks are in the data
                new_rows = []
                for id_ in file_ids:
                    # if the Mask is not in the data
                    if id_[1] not in self.data["Mask"].to_list():
                        if id_[0] not in self.data["ID"].to_list():
                            print("ID not found in data: ", id_[0])
                        else:
                            # Create new rows for missing IDs
                            new_row = self.data[self.data["ID"]==id_[0]].iloc[0].copy()
                            new_row["Mask"] = id_[1]
                            new_rows.append(new_row)

                self.data = pd.concat([self.data, pd.DataFrame(new_rows)], ignore_index=True)
                self.data = self.data.drop_duplicates(subset=['Image', 'Mask'])

                # write new input csv file and replace variable
                self.data.to_csv(reformat_path)

                # update clinical data
                self.clinical_df["ID"] = self.data["ID"].copy()
                self.clinical_df["Image"] = self.data["Image"].copy()
                self.clinical_df["Mask"] = self.data["Mask"].copy()
        
            if len(self.clinical_df) > 0:

                self.clinical_df = self.validate_sample_masks(df=self.clinical_df)

                if self.clinical_df.index.name == "ID":
                    self.clinical_df["ID"] = self.clinical_df.index
                
                # in case there are multiple segmentations from different raters but only one should be accepted
                self.clinical_df = self.clinical_df.drop_duplicates(subset=['Image', 'Mask'], keep='first')
            
            self.clinical_df.to_csv(self.out_path + "preprocessed_data/clinical_data.csv")

            if self.delta:
                self.clinical_df = self.process_delta_clinical_data(df=self.clinical_df.copy(), 
                                                                    output_csv_path=self.out_path + "preprocessed_data/clinical_data.csv")

        else:
            print("No format check. Could not find input data.  Check the data parameter.")
            self.error.warning("No format check. Could not find input data. Check the data parameter.")


    def validate_sample_masks(self, df):
        """
        Check for preprocessed peritumoral segmentsations and if there are duplicates in Image/Mask and if Image and Mask are from non preprocessed folders.
        """

        if df.index.name == "ID": 
            df = df.reset_index(drop=False)

        cleaned_df = df.copy()
        initial_samples = len(cleaned_df)
        
        def process_sample_group(sample_group):
            
            sample = str(set(sample_group["ID"].to_list()))
            try:
                # Find Peritumoral masks and validate their filenames
                peritumoral_masks = sample_group.copy()[sample_group['Mask_Transformation'] == 'Peritumoral']
                invalid_peritumoral_masks = peritumoral_masks[~peritumoral_masks['Mask'].apply(
                    lambda filename: bool(re.search(r'roiT_[0-9]*_roiN_[0-9]*(_[0-9]*[_]*peritumoral)', str(os.path.basename(filename))))
                )]['Mask']
            except Exception as ex:
                self.error.error(f"It seems like peritumoral segmentation for sample {sample} is not correctly formated or missing! Check the preprocessing files for peritumoral segmentations. " + str(ex))
                raise ValueError(f"It seems like peritumoral segmentation for sample {sample} is not correctly formated or missing! Check the preprocessing files for peritumoral segmentations. " + str(ex))

            if len(invalid_peritumoral_masks) > 0:
                print(f"Found {len(invalid_peritumoral_masks)} invalid Peritumoral masks for sample {sample}: ", invalid_peritumoral_masks.to_list())
                self.error.warning(f"Found {len(invalid_peritumoral_masks)} invalid Peritumoral masks for sample {sample}: {invalid_peritumoral_masks.to_list()}")
                # Drop invalid Peritumoral masks from the sample group
                cleaned_sample_group = sample_group.loc[~sample_group['Mask'].isin(invalid_peritumoral_masks)]
            else:
                self.logger.info(f"All Peritumoral masks are valid for sample {sample}.")
                cleaned_sample_group = sample_group.copy()

        
            # If no valid Peritumoral masks, return an empty DataFrame
            if len(cleaned_sample_group) == 0:
                return pd.DataFrame()
            
            # Return all masks for this sample group
            return cleaned_sample_group
        
        # remove duplicated columns
        if any(df.columns.duplicated()):
            self.logger.info("Found duplicated columns: " + str(df.columns.duplicated()))
            self.logger.info("Remove duplicated columns.")
            print("Found duplicated columns: " + str(df.columns.duplicated()))
            df = df.loc[:,~df.columns.duplicated()]

        if "Mask_Transformation" in df.columns:
            if "Peritumoral" in df["Mask_Transformation"].values.tolist():
                if cleaned_df.index.name in cleaned_df.columns:
                    # drop index name column
                    cleaned_df = cleaned_df.drop(columns=cleaned_df.index.name)
                
                # Apply processing to each sample group
                cleaned_df = cleaned_df.groupby('ID', group_keys=False).apply(process_sample_group).reset_index(drop=False)
            else:
                cleaned_df = df.copy() # No Peritumoral mask transformation found in data
                self.error.warning("Could not find any peritumoral mask transformation in data! Check the data if this is correct.")
        else:
            print("Could not find any Mask_Transformation in data.")

        
        # Additional filtering steps
        if "Mask" in cleaned_df.columns:
            cleaned_df_preprocessed = cleaned_df.copy()[
                ~cleaned_df.copy()['Mask'].str.contains('input_reformated|converted', na=False)
            ]
        else:
            self.error.warning("Could not find any Mask coloumn in data. continue further ... ")
            cleaned_df_preprocessed = pd.DataFrame()

        # check if preprocessing has been performed already
        if len(cleaned_df_preprocessed) >0:
            if "Mask_Transformation" in df.columns:
                if len(cleaned_df_preprocessed[cleaned_df_preprocessed["Mask_Transformation"].isnull()]) > 0:
                    print("Scan for non preprocessed data ...")

                    self.logger.info("Scan for non preprocessed data ...")
                    cleaned_df = cleaned_df_preprocessed

            else:
                self.logger.info("Scan for non preprocessed data ...")
                # cleaned_df = cleaned_df_preprocessed
        else:
            self.logger.info("No preprocessing has been performed yet.")

        if "Image" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=['Image', 'Mask'])

        # Warnings
        if len(cleaned_df) == 0:
            self.error.warning("No samples remain after validation. All samples were filtered out.")
        
        if initial_samples != len(cleaned_df):
            self.error.warning(f"Samples reduced from {initial_samples} to {len(cleaned_df)} after validation.")
        
        # check for 3D masks
        if self.check_segmentation_dimension:
            masks = cleaned_df["Mask"].tolist()
            if masks:
                with ThreadPoolExecutor(max_workers=self.n_cpu) as ex:
                    is3d_flags = list(
                        tqdm(
                            ex.map(self.is_segmentation_3d, masks),
                            total=len(masks),
                            desc="Check 3D Masks",
                            unit="Mask",
                        )
                    )

                not_3d_segs = [m for m, ok in zip(masks, is3d_flags) if not ok]

                if not_3d_segs:
                    print(f"Found {len(not_3d_segs)} Segmentations which are not three dimensional: {not_3d_segs}")
                    print(f"Removing {len(not_3d_segs)} Samples from data ...")
                    self.logger.warning(
                        f"Found {len(not_3d_segs)} Segmentations which are not three dimensional. "
                        f"If you want to keep them disable check_segmentation_dimension parameter."
                    )
                    self.error.warning(
                        f"Found {len(not_3d_segs)} Segmentations which are not three dimensional and need to get removed: {not_3d_segs}"
                    )
                    cleaned_df = cleaned_df[~cleaned_df["Mask"].isin(not_3d_segs)]

        return cleaned_df

    def get_data_fingerprint(self, return_=False):
        """
        Get input data statistics and plots for an overview of the input data
        Extract the following parameters from the dataset:
        ROI size,
        Number of ROI,
        Spacing distribution,
        Number of timepoints,
        intensity distribution
        :param return_: (bool) if result should get returned or stays in class
        :return: data_fingerprint (pd.DataFrame)
        """

        print("Get Fingerprint from data : " + self.path2confCSV)

        data_stat = DataStatGenerator(input_path=self.path2confCSV,
                                      out_path=self.out_path + "preprocessed_data",
                                      RunID=self.RunID,
                                      logger=self.logger,
                                      error=self.error,
                                      num_cpus=self.n_cpu,
                                      extract_radiomics_fingerprint=self.extract_radiomics_fingerprint,
                                      bin_width=self.config["Feature_extraction_config"]["bin_width"],
                                      prediction_label=self.Prediction_Label)

        success, data_fingerprint = data_stat.run()

        # ROI size

        # Min of ROI size

        # Max of ROI size

        # Number of ROI

        # Spacing distribution

        # Number of time points

        # Intensity distribution (peritumoral)

        if not success:
            self.error.warning("Error in DataStatGenerator!")
            self.data_fingerprint = None
        else:
            self.logger.info("Data statistics successfully generated!")

            # check for correlation in Meta data
            self.generate_correlation_map(df=data_fingerprint,
                                          out_folder=self.out_path + "preprocessed_data/plots/",
                                          name="Data_Fingerprint")

            self.data_fingerprint = data_fingerprint

        if return_:
            return self.data_fingerprint

    def get_preprocessor(self, perturbation_method : list = None):
        """
        Create SegProcessor object for configuration of preprocessing.
        :return: SegProcessor object
        """

        if (not self.data_fingerprint is None) and (self.self_optimize):
            self_optimization = True
        else:
            if self.self_optimize:
                self.error.warning(
                    "No optimization of RPTK processing possible as extraction of data fingerprint failed!")
            self_optimization = False

        if self.resampling is None:
            self.resampling = self.config["Preprocessing_config"]["resampling"]

        if self.normalization is None:
            self.normalization = self.config["Preprocessing_config"]["normalization"]

        if self.segmentation_perturbation is None:
            self.segmentation_perturbation = self.config["Preprocessing_config"]["segmentation_perturbation"]

        if self.Max_num_rois is None:
            self.Max_num_rois = self.config["Preprocessing_config"]["max_num_rois"]
        
        if self.image_transformation is None:
            self.image_transformation = self.config["Preprocessing_config"]["image_transformation"]

        if self.dice_threshold is None:
            self.dice_threshold = self.config["Preprocessing_config"]["dice_threshold"]

        if perturbation_method is None:
            if self.perturbation_method is None:
                perturbation_method = self.config["Preprocessing_config"]["perturbation_method"]
            else:
                perturbation_method = self.perturbation_method

        SegPro = SegProcessor(rptk_config_json=self.rptk_config_json,
                              path2csv=self.path2confCSV,
                              modality=self.modality,
                              out_path=self.out_path + "preprocessed_data",
                              kernels=self.config["Preprocessing_config"]["transformation_kernels"],
                              RunID=self.RunID,
                              isotropic_scale=self.config["Preprocessing_config"]["isotropic_scale"],
                              resampling=self.resampling,
                              roi_threshold=self.config["Preprocessing_config"]["min_roi_threshold"],
                              max_num_rois=self.Max_num_rois,
                              n_cpu=self.n_cpu,
                              normalization=self.normalization,
                              normalization_method=self.config["Preprocessing_config"]["normalization_method"],
                              segmentation_perturbation=self.segmentation_perturbation,
                              perturbation_method=perturbation_method,
                              image_transformation=self.image_transformation,
                              roi_pert_repetition=self.config["Preprocessing_config"]["roi_pert_repetition"],
                              perturbation_roi_adapt_type=self.config["Preprocessing_config"][
                                  "perturbation_roi_adapt_type"],
                              perturbation_roi_adapt_size=self.config["Preprocessing_config"][
                                  "perturbation_roi_adapt_size"],
                              dice_threshold=self.dice_threshold,
                              peritumoral_seg=self.peritumoral_seg,
                              peri_dist=self.config["Preprocessing_config"]["peri_dist"],
                              expand_seg_dist=self.config["Preprocessing_config"]["expand_seg_dist"],
                              perturbation_factor=self.config["Preprocessing_config"]["perturbation_factor"],
                              seg_closing_radius=self.config["Preprocessing_config"]["seg_closing_radius"],
                              logger=self.logger,
                              error=self.error,
                              consider_multiple_labels=self.config["Preprocessing_config"]["consider_multiple_labels"],
                              #log_file_path=self.out_path + "preprocessed_data/RPTK_preprocessing_" + str(
                              #    self.RunID),
                              use_previous_output=self.use_previous_output,
                              self_optimize=self_optimization,
                              mean_slice_thickness=round(self.data_fingerprint['Slice_thickness'].mean()),
                              resample_slice_thickness_threshold=int(self.config["Preprocessing_config"]["resample_slice_thickness_threshold"]),
                              timeout=self.timeout,
                              seed=self.rand_state,
                              )

        return SegPro

    def preprocessing(self, return_=False, perturbation_method: list = None):
        """
        Do preprocessing of the Data means to transform images and perturb segmentations to gather information from
        configuration (config.json) and put it into a (config) CSV file for feature extraction
        :param return_ : (bool) if the result should get returned
        :param perturbation_method: (list) list of perturbation methods possibilities: ["supervoxel","connected_component", "random_walker"]
        """

        # self.logger.info("More information about the process can be seen in the log file {}".format(
        #    self.logs_dir + "/RPTK_preprocessing_" + str(self.RunID) + ".log"))

        self.Preprocessor = self.get_preprocessor(perturbation_method = perturbation_method)

        preprocessed_data = self.Preprocessor.process()  # Best give a configuration dataFrame out with all the data included
        

        if return_:
            return preprocessed_data
        else:
            del preprocessed_data

    def get_Extractor(self, extractor: str = None, out_path: str = None):
        """
        Extraction opbjection
        :param extractor: Extraction pipeline PyRadiomics or MIRP
        return: object
        """
        if extractor == "PyRadiomics":
            extraction_yaml = self.pyradiomics_extraction_yaml
        elif extractor == "MIRP":
            extraction_yaml = self.mirp_extraction_yaml
        else:
            self.error.error("Extractor {} not known!".format(str(extractor)))
            extraction_yaml = None

        if out_path is None:
            out_path = self.out_path
        else:    
            if not out_path.endswith("/"):
                out_path = out_path + "/"

            os.makedirs(out_path + extractor + "/extracted_features/tmp/", exist_ok=True)
            
        extractor = Extractor(extractor=extractor,  # Either PyRadiomics or MIRP
                              out_path=out_path + extractor,  # Path to output directory
                              n_cpu=self.n_cpu,  # Number of CPUs to use
                              chunksize=self.chunksize,  # Chunksize for multiprocessing
                              path2confCSV=self.out_path + "preprocessed_data/" + self.RunID + "_preprocessing_out.csv",
                              # Path to CSV file with Image and Segmentation paths
                              modality=self.modality,  # CT or MRI
                              # Correlation threshold for feature filtering
                              delta=self.delta,  # Delta Radiomics calculation
                              RunID=self.RunID,
                              extracted_features_tmp_dir=out_path + extractor + "/extracted_features/tmp/",
                              # logs_dir=self.out_path + extractor + "/logs/",
                              extraction_yaml=extraction_yaml,
                              extracted_features_dir=out_path + extractor + "/extracted_features/",
                              self_optimize=self.self_optimize,
                              logger=self.logger,
                              error=self.error,
                              use_previous_output=self.use_previous_output,
                              rptk_config_json=self.rptk_config_json,
                              fast_mode=self.fast_mode,
                              resegmentation=self.resegmentation,
                              take_label_changes=self.take_label_changes,
                              )
        return extractor

    def extract_features(self, return_=False, out_path=None):
        """
        Extract features from images and segmentations
        :param return_ : (bool) if the result should get returned or stay in class
        :return: list of done feature extractions
        """

        # create folder for feature extraction
        if not os.path.exists(self.extracted_features_dir):
            os.makedirs(self.extracted_features_dir)

        self.logger.info("### Start RPTK Feature extraction ###\n")
        print("### Start RPTK Feature extraction ###")

        # extract all features
        for extraction_pipeline in self.extractor:
            self.logger.info("### Extracting features with {} ###".format(extraction_pipeline))
            print("### Extracting features with {} ...".format(extraction_pipeline))

            #self.logger.info(
            #    "For logging on this process please see {} ###".format(str(self.out_path + extractor + "/logs/")))

            if not out_path is None:
                out_path_path = pathlib.PurePath(out_path)
                if extraction_pipeline not in out_path_path.parent.name:
                    out_path = out_path + extraction_pipeline + "/"

            extractor = self.get_Extractor(extractor=extraction_pipeline, out_path=out_path)
            features = extractor.exe()

            if self.self_optimize:
                # Need get encoding of configuration
                if extraction_pipeline == "MIRP":
                    if "MIRP_config" in features.columns:
                        features = self.feature_encoding(df=features, features=["MIRP_config"])
                        features.to_csv(extractor.outfile)


            self.extracted_features[extraction_pipeline] = features

            print("### Extraction with {} Finished!".format(extraction_pipeline))
            self.logger.info("### Extraction with {} Finished! ###".format(extraction_pipeline))
            self.logger.info("Extracted {} Radiomics Features for {} Entries!".format(str(len(features.columns)),
                                                                                      str(len(features))))
            print("Extracted {} Radiomics Features for {} Samples!!".format(str(len(features.columns)),
                                                                            str(len(features))))
            self.logger.info("Memory usage after %s Feature extraction: %0.3f MB" % (extraction_pipeline,
                                                                                     psutil.Process().memory_info().rss / 1e6))

            # if not os.path.exists(self.out_path + extractor + "/extracted_features/" + extractor + "_extraction_" + self.RunID + ".csv"):
            #    features.to_csv(self.out_path + extractor + "/extracted_features/" + extractor + "_extraction_" + self.RunID + ".csv", index=False)


        print("### RPTK Feature extraction finished! ###")
        self.logger.info("### RPTK Feature extraction finished! ###")

        if return_:
            return self.extracted_features

    def filter_features(self, extracted_features=None, return_=False, stability_filtering=None, format_check=True, peritumoral=True, out_path=None):
        """
        Filter features based on variance and correlation
        :param features: List of pd.DataFrames or path to outfiles including extracted features
        :param return_: (bool) if result should get returned or stays in class
        :return: filtered_features: List of pd.DataFrames including filtered features
        """

        if extracted_features is None:
            extracted_features = self.extracted_features
        
        if stability_filtering is None:
            stability_filtering = self.stability_filtering
        
        self.filtered_features = {}

        self.logger.info("### Start RPTK Feature Filtering ###\n")
        print("### Start RPTK Feature Filtering ###")

        if out_path is None:
            out_path = self.out_path

        for extractor in extracted_features:
            features = extracted_features[extractor]
            self.logger.info("### Starting {} Radiomics Feature Filtering".format(extractor))
            print("### Starting {} Radiomics Feature Filtering ... ".format(extractor))

            if len(features) == 0:
                self.logger.error("No features were provided. Please check your feature extraction!")
                raise ValueError("No features were provided. Please check your feature extraction!")

            # self.features = self.out_path + "/features/" + self.extractor + "_features.csv"
            self.logger.info("Starting Radiomics Feature Filtering for {} features".format(extractor))

            log_file_path = out_path + extractor + "/logs/feature_filtering.log"
            self.logger.info("For loggings on this process please see {} ###".format(log_file_path))

            out_path_path = pathlib.PurePath(out_path)

            if extractor not in out_path_path.parent.name:
                os.makedirs(out_path + extractor + "/logs/", exist_ok=True)
                os.makedirs(out_path + extractor + "/filtered_features/", exist_ok=True)
                extractor_out_path = out_path + extractor + "/"
            else:
                os.makedirs(out_path + "logs/", exist_ok=True)
                os.makedirs(out_path + "filtered_features/", exist_ok=True)
                extractor_out_path = out_path + extractor + "/"

            # features.to_csv(os.path.join(out_path + extractor, "filtered_features/before_filtering.csv"))

            # check data for preprocessed data
            features = self.validate_sample_masks(df=features.copy())

            # features.to_csv(os.path.join(out_path + extractor, "filtered_features/before_filtering.csv"))

            self.radiomics_filter = RadiomicsFilter(data=features,
                                                    out_path=extractor_out_path,
                                                    # TODO: Set alternative if features is a path or a dataframe
                                                    path_to_img_seg_csv=self.path2confCSV,
                                                    variance_threshold=
                                                    self.config["Feature_filtering_config"][
                                                        "variance_threshold"],
                                                    correlation_threshold=
                                                    self.config["Feature_filtering_config"][
                                                        "correlation_threshold"],
                                                    extractor=extractor,
                                                    ICC_threshold=self.config["Feature_filtering_config"][
                                                        "ICC_threshold"],
                                                    RunID=self.RunID,
                                                    n_cpu=self.n_cpu,
                                                    stability_filtering=stability_filtering,
                                                    error=self.error,
                                                    format_check=format_check,
                                                    peritumoral=peritumoral,
                                                    additional_rois_to_features=self.additional_rois_to_features,
                                                    delta=self.delta,)

            self.filtered_features[extractor] = self.radiomics_filter.run()

            if "ID" not in self.filtered_features[extractor].columns:
                if "ID" != self.filtered_features[extractor].index.name:
                    if os.path.isfile(out_path + extractor + "/filtered_features/filtered_features.csv"):
                        self.filtered_features[extractor] = pd.read_csv(out_path + extractor + "/filtered_features/filtered_features.csv", index_col = 0)
                    else:
                        print("Warning: ID is missing in filtered features and output file {} is missing!".format(out_path + extractor + "/filtered_features/filtered_features.csv"))
                        self.error.warning("ID is missing in filtered features and output file {} is missing!".format(out_path + extractor + "/filtered_features/filtered_features.csv"))

            self.logger.info("Filtered feature space contains {} features and {} entries!".format(
                len(self.filtered_features[extractor].columns),
                len(self.filtered_features[extractor])))
            print("Filtered feature space contains {} features and {} entries!".format(
                len(self.filtered_features[extractor].columns),
                len(self.filtered_features[extractor])))

            self.logger.info("Memory usage after %s Feature filtering: %0.3f MB" % (
            extractor, psutil.Process().memory_info().rss / 1e6))

            self.logger.info("### Filtering of {} Radiomics Features Finished!\n".format(extractor))
            print("### Filtering of {} Radiomics Features Finished!\n".format(extractor))

        print("### RPTK Feature filtering finished! ###")
        self.logger.info("### RPTK Feature filtering finished! ###")

        if ("ID" not in self.filtered_features[extractor].columns) & ("ID" != self.filtered_features[extractor].index.name):
            self.error.error("ID is missing in filtered features! Please check the output before selection.")
            print("ID is missing in filtered features! Please check the output before selection.")
        else:
            self.logger.info("ID is present in filtered features!")

        if return_:
            return self.filtered_features

    def process_filename(self, filepath):
        """
        Process the filename to get the config
        :param filepath: (str) path to the file
        :return: (str) config
        """
        return os.path.basename(filepath).replace(".nii.gz", "")

    def drop_non_float_convertible_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features which are not numbers
        """

        convertible_cols = []
        for col in df.columns:
            if col == "ID":
                convertible_cols.append(col)
                continue
            try:
                df[col].astype(float)
                convertible_cols.append(col)
            except (ValueError, TypeError):
                # Skip columns that cannot be converted to float
                self.error.warning(f"Could not convert {col} to float! Dropping ...")
                print(f"Could not convert {col} to float! Dropping ...")
                continue

        return df[convertible_cols]

    def select_features(self, filtered_features=None, return_=False, n_features=None, model=None, backward_sfs=True, forward_sfs=True, critical_feature_size=None, min_feature=None, max_feature=None, rand_state=None, out_folder=None):
        """
        Select features based on feature importance
        :param filtered_features: dict with extractor as key of pd.DataFrames including filtered features
        :param return_: (bool) if result should get returned or stays in class
        :param n_features: (int) number of features to select
        :return: selected_features: dict with extractor as key  of pd.DataFrames including selected features
        """
        
        os.environ['OMP_NUM_THREADS'] = str(self.n_cpu)
        
        if critical_feature_size is None:
            if "critical_feature_size" in self.config["Feature_selection_config"]:
                critical_feature_size = self.config["Feature_selection_config"]["critical_feature_size"]
            else:
                critical_feature_size = 200

        if min_feature is None:
            if "min_feature" in self.config["Feature_selection_config"]:
                min_feature = self.config["Feature_selection_config"]["min_feature"]
            else:
                min_feature = 5

        if n_features is None:
            n_features = self.config["Feature_selection_config"]["n_features"]

        if max_feature is None:
            if "max_feature" in self.config["Feature_selection_config"]:
                max_feature = self.config["Feature_selection_config"]["max_feature"]
            else:
                max_feature = 20

        if len(self.clinical_df) > 0:
            if self.clinical_df.index.name == "ID":
                self.clinical_df["ID"] = self.clinical_df.index
            
            # in case there are multiple segmentations from different raters but only one should be accepted
            self.clinical_df = self.clinical_df.drop_duplicates(subset=['ID', 'Image'], keep='first')
        
        os.makedirs(self.out_path + "preprocessed_data/plots/", exist_ok=True)
        
        self.clinical_df.to_csv(self.out_path + "preprocessed_data/clinical_data.csv")

        if self.delta:
            self.clinical_df = self.process_delta_clinical_data(df=self.clinical_df.copy(), 
                                                                output_csv_path=self.out_path + "preprocessed_data/clinical_data.csv")

        if filtered_features is None:
            filtered_features = self.filtered_features

        if rand_state is None:
            rand_state = self.rand_state

        if self.feature_selection:
            self.logger.info("### Start RPTK Feature Selection ###\n")
            print("### Start RPTK Feature Selection for {} ###".format([key for key in filtered_features]))

            for extractor in filtered_features:
                features = filtered_features[extractor]
                
                self.logger.info("### Starting Radiomics Feature Selection for {} Features".format(extractor))
                print("### Starting {} Radiomics Feature Selection ... ".format(extractor))
                print("Filtered Features Input:", features.shape)
                
                if len(features) == 0:
                    self.logger.error("No features were provided. Please check your feature extraction!")
                    raise ValueError("No features were provided. Please check your feature extraction!")

                self.logger.info("Starting Radiomics Feature Selection for {} features".format(extractor))

                log_file_path = self.out_path + extractor + "/logs/feature_selection.log"
                self.logger.info("For loggings on this process please see {} ###".format(log_file_path))

                if len(self.clinical_df) > 0:
                    
                    # Format features
                    if "config" not in features.columns:
                            
                        if features.index.name == "config":
                            features["config"] = features.index

                        elif "unique_config" in features.columns:
                                features.rename(columns={'unique_config': 'config'}, inplace=True, errors='raise')
                                features.index = features["config"]

                        elif ("Mask" in features.columns) and ("Image" in features.columns):
                            # Extract the base filename and remove the ".nii.gz" ending
                            

                            features['config'] = features['Image'].apply(self.process_filename) + "_" + features['Mask'].apply(self.process_filename)

                        else:
                            print("No config nor Image/Mask found in inut data. Please add these features.")
                            self.error.error("No config nor Image/Mask found in inut data. Please add these features.")
                            raise ValueError("No config nor Image/Mask found in inut data. Please add these features.")
                    
                    if self.clinical_df.index.name == "ID":
                        self.clinical_df["ID"] = self.clinical_df.index
                    
                    if self.delta:
                        # sync the ID with filtered data
                        self.clinical_df['ID'] = self.clinical_df['ID'].str.replace(r'-delta-[^-]+', '', regex=True)

                    # missing ID in the features -> getting ID from clinical data
                    for id_ in self.clinical_df["ID"]:
                        for conf in features["config"]:
                            if str(conf).startswith(str(id_)):
                                features.loc[features["config"] == conf, "ID"] = self.clinical_df.loc[self.clinical_df["ID"] == id_, "ID"].values[0]

                    self.clinical_df.index = self.clinical_df['ID']
                    self.clinical_df.drop(['ID'], axis=1, inplace=True)

                    features.index = features['ID']
                    features.drop(['ID'], axis=1, inplace=True)

                    # Check if there are clinical features not present in radiomics features
                    integrate_clinical_features = False
                    for clinical_feature in self.clinical_df.columns:
                        if clinical_feature not in features.columns.tolist():
                            integrate_clinical_features = True
                            break
                        
                    # clincal fetaure integration        
                    if integrate_clinical_features:
                        
                        combined_data = pd.DataFrame()

                        if features.index.duplicated().any:
                            print("Detected multiple ROIs per Sample.")

                            # Got clinical data per sample not per ROI
                            if len(self.clinical_df) < len(features):
                                features = features.reset_index()
                                self.clinical_df = self.clinical_df.reset_index()
                                
                                for id in features.index.values.tolist():
                                    # Include clincal data based on the ID (if ID occurs multiple times in fetures for every ROI)
                                    tmp = pd.concat([features[features.index==id],self.clinical_df[self.clinical_df["ID"]==features.loc[features.index==id,"ID"].values[0]]], keys=["ID"], axis=0, ignore_index=True)
                                    combined_data = pd.concat([combined_data, tmp])
                            else:
                                combined_data = pd.concat([self.clinical_df, features], axis=1)
                            
                            if "ID" in  combined_data.columns:
                                combined_data = combined_data.set_index("ID")
                        else:
                            # Add clinical parameter to radiomics parameter
                            combined_data = pd.concat([self.clinical_df, features], axis=1)
                            
                    else:
                        self.logger.info("No clinical features need to get integrated.")
                        print("No clinical features need to get integrated.")
                        combined_data = features
                    
                    
                    # combined_data = pd.merge(features, self.clinical_df, left_on=True)

                    # drop image and mask as they are not the right ones for the sample anyhow (if cropping, if resampling etc)
                    if "Image" in combined_data.columns:
                        combined_data.drop(['Image'], axis=1, inplace=True)
                    if "Mask" in combined_data.columns:
                        combined_data.drop(['Mask'], axis=1, inplace=True)

                    combined_data.set_index('config')
                    features = combined_data

                if model is None:
                    model = self.config["Feature_selection_config"]["models"]

                if out_folder is None:
                    out_folder = self.out_path + extractor + "/selected_features"
                    os.makedirs(self.out_path + extractor + "/selected_features/", exist_ok=True)
                else:
                    out_folder_path = pathlib.PurePath(out_folder)
                    if "selected_features" not in out_folder_path.parent.name:
                        out_folder = out_folder + extractor + "/selected_features"
                        os.makedirs(out_folder + extractor + "/selected_features/", exist_ok=True)

                # drop non float parameters
                features = self.drop_non_float_convertible_columns(features)

                features, non_normalied_features = self.normalize_if_needed(features.copy(), self.clinical_df.columns.to_list())
                if len(non_normalied_features) >0:
                    print(f"Normalized non-normalized clinical selected features: {non_normalied_features}!")
                    self.logger.info(f"Normalized non-normalized clinical selected features: {non_normalied_features}!")

                if isinstance(rand_state, list):
                    for seed in rand_state:
                        self.feature_selector = FeatureSelectionPipeline(
                                                                    data=features,
                                                                    n_cpus=self.n_cpu,
                                                                    RunID=self.RunID,
                                                                    logger=self.logger,
                                                                    error=self.error,
                                                                    out_folder=out_folder,
                                                                    use_gpu=self.use_gpu,
                                                                    model=model,
                                                                    verbose=self.verbose,
                                                                    sfs_lib=self.config["Feature_selection_config"][
                                                                                "sfs_lib"],
                                                                    save_model=self.config["Feature_selection_config"][
                                                                                "save_model"],
                                                                    Prediction_Label=self.Prediction_Label,
                                                                    self_optimize=self.self_optimize,
                                                                    extractor=extractor,
                                                                    rand_state=seed,
                                                                    modality=self.modality,
                                                                    n_features=n_features,
                                                                    backward_sfs=backward_sfs,
                                                                    forward_sfs=forward_sfs,
                                                                    critical_feature_size=critical_feature_size,
                                                                    min_feature=min_feature,
                                                                    max_feature=max_feature,
                                                                    imbalance_method=self.imbalance_method)

                        if extractor in self.selected_features:
                            self.selected_features[extractor] = pd.concat([self.selected_features[extractor], self.feature_selector.select_features()], axis=1)
                        else:
                            self.selected_features[extractor] = self.feature_selector.select_features()


                else:
                    self.feature_selector = FeatureSelectionPipeline(
                                                                data=features,
                                                                n_cpus=self.n_cpu,
                                                                RunID=self.RunID,
                                                                logger=self.logger,
                                                                error=self.error,
                                                                out_folder=self.out_path + extractor + "/selected_features",
                                                                use_gpu=self.use_gpu,
                                                                model=model,
                                                                verbose=self.verbose,
                                                                sfs_lib=self.config["Feature_selection_config"][
                                                                            "sfs_lib"],
                                                                save_model=self.config["Feature_selection_config"][
                                                                            "save_model"],
                                                                Prediction_Label=self.Prediction_Label,
                                                                self_optimize=self.self_optimize,
                                                                extractor=extractor,
                                                                rand_state=rand_state,
                                                                modality=self.modality,
                                                                n_features=n_features,
                                                                backward_sfs=backward_sfs,
                                                                forward_sfs=forward_sfs,
                                                                critical_feature_size=critical_feature_size,
                                                                min_feature=min_feature,
                                                                max_feature=max_feature,
                                                                imbalance_method=self.imbalance_method)

                    self.selected_features[extractor] = self.feature_selector.select_features()


                self.logger.info("Memory usage  after %s Feature selection: %0.3f MB" % (
                    extractor, psutil.Process().memory_info().rss / 1e6))

                # check for correlation in clinical data
                self.generate_correlation_map(df=self.selected_features[extractor],
                                            out_folder=self.out_path + extractor + "/selected_features/",
                                            name="Selected_Features")
                
                self.generate_feature_overview_map(features_df=self.selected_features[extractor].copy(), 
                                                extractor=extractor, 
                                                out_folder=self.out_path + extractor + "/selected_features",
                                                title=f"Selected {extractor} Features Clustering")
                
        else:
            self.logger.info("Feature selection disabled! Set 'feature_selection' to True in the configuration file to enable it.")
            print("Feature selection disabled! Set 'feature_selection' to True in the configuration file to enable it.")
            self.selected_features = filtered_features

        if return_:
            return self.selected_features

    def get_most_common_kernels(self, kernels, best_n):
        """
        Get n most repeated elements in a list
        :param kernels: list of elements
        :param best_n: most n repeted elements
        :return list of n most common elements
        """
        
        most_common_kernels = []
        
        for count in Counter(kernels).most_common(best_n):
            most_common_kernels.append(count[0])

        return most_common_kernels

    def extract_selected_kernels(self, extractor, selected_features, feature_profile):
        """
        Get Fetaures without kernel name and kernels in feature space
        :param extractor: name of extractor (PyRadiomics or MIRP)
        :param selected_features: pd.DataFrame with selected features and ID as index
        :param feature profile: generated feature profile from RPTK feature profil generator
        """
        kernels_in_selected_features = []
        selected_features_wo_kernel = []
        selected_peritumoral_features = []

        for feat in self.selected_features[extractor].columns:
            for feature in feature_profile["Name"]:
                if feature == feat:
                    if "peritumoral" in feat:
                        if feat not in selected_peritumoral_features:
                            selected_peritumoral_features.append(feat)
                    else:
                        kernel=feature_profile.loc[feature_profile["Name"]==feature, "Image_Kernel"].values[0]
                        if kernel != "Original_Image":
                            kernels_in_selected_features.append(kernel)
                        feat = feat.replace(kernel + "_", '')
                        feat = feat.replace("_zscore", '')

                        if feat not in selected_features_wo_kernel:
                            selected_features_wo_kernel.append(feat)
                        break
                        
        return kernels_in_selected_features, selected_features_wo_kernel, selected_peritumoral_features

    def summarize_selected_features_from_filtering(self, 
                                                   filtered_features, 
                                                   selected_peritumoral_features, 
                                                   original_selected_features, 
                                                   selected_features_wo_kernel, 
                                                   most_common_kernels):
        """
        Concat selected features on sample level
        """
        transform_kernel = {
                                        "LoGsigma":"log-sigma",
                                        "Wavelet":"wavelet-",
                                        "LBP2D":"lbp-2D",
                                        "gaussian":"gauss",
                                        "laws":"laws_",
                                        # "Exponential":"exponential",
                                        "Gradient":"gradient",
                                        "Squareroot":"squareroot",
                                        "Square":"square",
                                            }
        
        available_selected_features = {}  # seleczed features from kernel OI (most common kernel)
        # original_selected_features = []
        
        for filtered_feature in filtered_features:
            
            # Peritmuoral features are only from orginal images
            if filtered_feature in selected_peritumoral_features:
                if filtered_feature not in original_selected_features:
                    original_selected_features.append(filtered_feature)
            else:
                for raw_feature in selected_features_wo_kernel:
                    if raw_feature == filtered_feature:
                        if raw_feature not in original_selected_features:
                            original_selected_features.append(filtered_feature)
                    elif raw_feature in filtered_feature:
                        for kernel in most_common_kernels:
                            
                            if kernel in filtered_feature:
                                if kernel in available_selected_features:
                                    available_selected_features[kernel].append(filtered_feature)
                                else:
                                    available_selected_features[kernel] = [filtered_feature]
                            else:
                                for translated_kernel in transform_kernel:
                                    if translated_kernel in kernel:
                                        kernel = kernel.replace(translated_kernel, transform_kernel[translated_kernel])
                                        if transform_kernel[translated_kernel] in filtered_feature:
                                            if kernel in available_selected_features:
                                                available_selected_features[kernel].append(filtered_feature)
                                            else:
                                                available_selected_features[kernel] = [filtered_feature]
                                                
                                        break
                                            
                                
        if len(available_selected_features) == 0:
            print("Could not find selected features with kernels" + str(most_common_kernels) + " check configuration between filtered featureas and selected features")
            self.error.error("Could not find selected features with kernels" + str(most_common_kernels) + " check configuration between filtered featureas and selected features")
            raise ValueError("Could not find selected features with kernels" + str(most_common_kernels) + " check configuration between filtered featureas and selected features")
        
        return available_selected_features, original_selected_features

    def get_features_without_kernel(self, available_selected_features, filtered_features):
        """
        Get Features with kernel name from filtered feature space and remove kernel name but put into pd.DataFrame
        """
        
        included_features = []
        transformed_features = pd.DataFrame()
        for kernel in available_selected_features:
            
            for feature in available_selected_features[kernel]:
                if feature in filtered_features.columns:
                    if feature not in included_features:
                        included_features.append(feature)
                else:
                    print("Feature {} not in filtered feature space!".format(feature))
                    self.error.warning("Feature {} not in filtered feature space!".format(feature))

            transformed_features_tmp = filtered_features[included_features]

            # remove kernel from feature names
            for feat in transformed_features_tmp.columns:
                raw_feat = feat.replace(kernel + "_", '')
                transformed_features_tmp = transformed_features_tmp.rename(columns = {feat:raw_feat})

            transformed_features_tmp = transformed_features_tmp.loc[:,~transformed_features_tmp.columns.duplicated()].copy()
            transformed_features = pd.concat([transformed_features,transformed_features_tmp], axis=0)

        transformed_features = transformed_features.dropna(axis='columns')
        
        return transformed_features


    def generate_augmented_feature_space(self, extractor=None, selected_features=None, filtered_features=None, n_best_kernels=None, mask_transformations=None):
        """
        Generate feature space with concatinated feaures wicht modt representative image transofrmation kernel
        :param extractor: Name of extractor (PyRadiomics or MIRP)
        :param selected_features: Dict with Name of extractor: selected feature space with ID as indey (pd.DataFrame)
        :param filtered_features: pd.DataFrame of normalized features containing all and more than only the selected featires for concatinating the selected feature space over different transformations
        :param n_best_kernels (int): number of most represented kernels in selected features 
        :param mask_transformations: data with mask transformations 
        """

        if isinstance(filtered_features, dict):
            real_filtered_features = filtered_features[extractor]
            filtered_features = real_filtered_features
            
        if filtered_features.index.name != "ID":
            if "ID" in filtered_features.columns:
                filtered_features = filtered_features.set_index("ID")
            else:
                print("Warning: Could not find ID in filtered data!")
                self.error.warning("Error could not find ID in filtered data!")
                
        if selected_features[extractor].index.name != "ID":
            if "ID" in selected_features[extractor].columns:
                selected_features[extractor] = selected_features[extractor].set_index("ID")
                selected_features[extractor].index.name = "ID"
            else:
                print("Warning: Could not find ID in selected data!")
                self.error.warning("Error could not find ID in selected data!")
                
        if not mask_transformations is None:
            
            if isinstance(mask_transformations, dict):
                mask_transformations = mask_transformations[extractor]

            if mask_transformations.index.name != "ID":
                if "ID" in mask_transformations.columns:
                    mask_transformations = mask_transformations.set_index("ID")
                    selected_features[extractor].index.name = "ID"
                else:
                    print("Warning: Could not find ID in mask transformaton data!")
                    self.error.warning("Error could not find ID in mask transformaton data!")
        
        selected_feature_profile = FeatureFormatter(features=selected_features[extractor].columns,
                                                    extractor=extractor,
                                                    logger=self.logger,
                                                    error=self.error,
                                                    output_path=self.out_path + extractor +"/selected_features/").exe(title="Selected_Feature_Profile")
        
        selected_feature_profile.to_csv(self.out_path + extractor +"/selected_features/selected_feature_profile.csv")
        
        selected_features[extractor].index.name = "ID"

        # 1. get all features with image transformations -> count transformation representations in selected features only take most represented transformation
        kernels_in_selected_features = []
        selected_features_wo_kernel = []
        selected_peritumoral_features = []

        kernels_in_selected_features, selected_features_wo_kernel, selected_peritumoral_features = self.extract_selected_kernels(extractor, 
                                                                                                                                                                                                selected_features, 
                                                                                                                                                                                                selected_feature_profile)

        most_common_kernels = self.get_most_common_kernels(kernels_in_selected_features, n_best_kernels)
        
        print("Found most present image transformations in selected features: " + str(most_common_kernels))
        self.logger.info("Found most present image transformations in selected features: " + str(most_common_kernels))
        
        self.filtered_common_features = filtered_features
        
        available_selected_features = {}
        original_selected_features = []

        available_selected_features, original_selected_features = self.summarize_selected_features_from_filtering(filtered_features, 
                                                                                                            selected_peritumoral_features, 
                                                                                                            original_selected_features, 
                                                                                                            selected_features_wo_kernel,
                                                                                                            most_common_kernels)
        

        # get selected features from filtered feature space for each selected kernel
        transformed_features = self.get_features_without_kernel(available_selected_features, filtered_features)
        
        if not mask_transformations is None:
            mask_transformed_features = self.get_features_without_kernel(available_selected_features, mask_transformations)
            transformed_features = pd.concat([transformed_features, mask_transformed_features])
        
        feature_consensus = []
        for feature in transformed_features.columns:
            if feature in filtered_features.columns:
                feature_consensus.append(feature)

        transformed_features["Transformations"] = 1

        original_features = filtered_features[feature_consensus]
        original_features = original_features.loc[:,~original_features.columns.duplicated()].copy()
        original_features["Transformations"] = 0

        final_feature_space = pd.concat([original_features,transformed_features], axis=0)
        final_feature_space = final_feature_space.dropna(axis='columns')
        final_feature_space["Prediction_Label"] = np.nan
        
        for id_ in list(set(final_feature_space.index.tolist())):
            prediction = np.nan
            
            if is_number(filtered_features.loc[id_,"Prediction_Label"]):
                prediction = filtered_features.loc[id_, "Prediction_Label"]
            else:
                prediction = filtered_features.loc[id_, "Prediction_Label"].values[0]
    
            final_feature_space.loc[id_,"Prediction_Label"] = prediction
        
        if final_feature_space['Prediction_Label'].isnull().values.any():
            print("Prediction_Label seems to have nan values. Check index configuration!")
            self.error.error("Prediction_Label seems to have nan values. Check index configuration!")
            raise ValueError("Prediction_Label seems to have nan values. Check index configuration!")
        
        final_feature_space.index.name = "ID"

        return final_feature_space

    def predict(self, 
                selected_features=None, 
                train_ids = None,
                test_ids = None,
                val_ids = None,
                shap_analysis=None, 
                neptune_api_token=None, 
                neptune_project=None, 
                stable_pretraining=None, 
                neptune_run_name=None,
                optimization_iter=None,
                extended_parameter_set=None,
                super_extended_parameter_set=None,
                extractor=None,
                seed=None,
                out_folder=None,
                reuse_split=True,  # Training by reusing train test split from feature selection
                model=None,
                optimize=None,
                ensemble=None,
                repeat_prediction=False,
                device=None,
                mask_transformations=None,
                include_augmentations=False,
                n_best_kernels=1,
                use_cross_validation=None,
                cross_val_splits=None,
                predict_only=False,
                model_path=None,
                autoradiomics_splits_path=None,
                ):
        """
        Prediction of the label on the selected feature space with different models integrated into RPTK.
        """

        os.environ['OMP_NUM_THREADS'] = str(self.n_cpu)

        self.predict_only = predict_only
        self.model_path = model_path

        set_outfolder_None = False
        if out_folder is None:
            set_outfolder_None = True
        else:
            repeated_out_folder = out_folder

        if not train_ids is None:
            self.train_ids = train_ids 

        if not test_ids is None:
            self.test_ids = test_ids

        if not val_ids is None:
            self.val_ids = val_ids

        if seed is None:
            seed = [self.rand_state]

        # if ypu provide multiple seeds you want to perform multiple times predictions with different seeds
        elif isinstance(seed, list):
            repeat_prediction = True
        
        if repeat_prediction:
            reuse_split = False
            for i in range(repeat_prediction):
                random.seed(i)
                seed.append(random.randint(1, 1000))
        else:
            repeat_prediction = False
        
        if use_cross_validation is None:
            use_cross_validation = self.config["RPTK_prediction_config"]["use_cross_validation"]

        if cross_val_splits is None:
            if "cross_val_splits" in self.config["RPTK_prediction_config"]:
                cross_val_splits = self.config["RPTK_prediction_config"]["cross_val_splits"]
            else:
                cross_val_splits = 5

        if shap_analysis is None:
            shap_analysis = self.shap_analysis
        
        if model is None:
            model = self.models
        
        if ensemble is None:
            ensemble = self.ensemble
        
        if extended_parameter_set is None:
            extended_parameter_set = True
        
        if super_extended_parameter_set is None:
            super_extended_parameter_set = False

        if neptune_run_name is None:
            neptune_run_name = ""
        
        if stable_pretraining is None:
            stable_pretraining =  self.config["RPTK_prediction_config"]["stable_pretraining"]
            
        if optimize is None:
            optimize = self.optimize
        
        if (neptune_api_token is None) and (neptune_project is None):
            if self.run_neptune:
                run_neptune = True
                neptune_api_token = self.neptune_api_token
                neptune_project = self.neptune_project
            else:
                run_neptune = False
        else:
            run_neptune = True
        
        if optimization_iter is None:
            optimization_iter = self.optimization_iter
        
        if selected_features is None:
            selected_features = self.selected_features.copy()
         
        for extractor in selected_features:

            if include_augmentations:
                
                if len(glob.glob(self.out_path + extractor + "/filtered_features/tmp/*_normalized_data_wo_mask_transformation.csv")) > 0:
                    # Get filtered features from 
                    filtered_features_path = glob.glob(self.out_path + extractor + "/filtered_features/tmp/*_normalized_data_wo_mask_transformation.csv")[0]
                else:
                    self.error.error("Could not find filtered features to condense the feature space. Please check if a file with ending _normalized_data_wo_mask_transformation.csv is in {} !".format(str(self.out_path + extractor + "/filtered_features/tmp/")))
                    raise ValueError("Could not find filtered features to condense the feature space. Please check if a file with ending _normalized_data_wo_mask_transformation.csv is in {} !".format(str(self.out_path + extractor + "/filtered_features/tmp/")))
                #filtered_features_path = self.out_path + extractor + "/filtered_features/tmp/" + str(self.RunID) + "_normalized_data_wo_mask_transformation.csv"
                filtered_features = pd.read_csv(filtered_features_path, index_col=0)

                print("Reading non-transformed and normalized data from", filtered_features_path)

                if not mask_transformations is None:
                    if not isinstance(mask_transformations, pd.DataFrame):
                        if not isinstance(mask_transformations, str):
                            mask_filtered_features_path = glob.glob(self.out_path + extractor + "/filtered_features/tmp/*_normalized_data_mask_transformation.csv")[0]
                            # mask_filtered_features_path = self.out_path + extractor + "/filtered_features/tmp/" + str(self.RunID) + "_normalized_data_mask_transformation.csv"
                            mask_transformations = pd.read_csv(mask_filtered_features_path, index_col=0)
                            print("Reading transformed and normalized data from", mask_filtered_features_path)
                        else:
                            mask_transformations_data = pd.read_csv(mask_transformations, index_col=0)
                            print("Reading transformed and normalized data from", mask_transformations)
                            mask_transformations = mask_transformations_data
                            

                selected_features_extractor = self.generate_augmented_feature_space(extractor=extractor, 
                                                                                    selected_features=selected_features, 
                                                                                    filtered_features=filtered_features, 
                                                                                    n_best_kernels=n_best_kernels, 
                                                                                    mask_transformations=mask_transformations)
            else:
                selected_features_extractor = selected_features[extractor]

            self.logger.info("### Starting RPTK Prediction for {} Features".format(extractor))
            print("### Starting {} RPTK Prediction ... ".format(extractor))

            if len(selected_features_extractor) == 0:
                self.error.error("No features were provided. Please check your feature extraction!")
                raise ValueError("No features were provided. Please check your feature extraction!")
            else:
                print("Perform on Selected Feature Space",selected_features_extractor.shape)

            self.logger.info("Starting RPTK Prediction for selected {} features".format(extractor))
            
            log_file_path = self.out_path + extractor + "/logs/prediction.log"
            self.logger.info("For loggings on this process please see {} ###".format(log_file_path))
            
            if device is None:
                # check if gpu is available and allowed
                if torch.cuda.is_available() and self.use_gpu:
                    if not self.device is None:
                        if "cuda" not in self.device:
                            device = "cuda"
                        else:
                            device = self.device 
                    else:
                        device = "cuda"
                    print("Processing on GPU & CPU ...")
                    self.logger.info("Processing on GPU & CPU ...")
                else:
                    print("No GPU found, processing on CPU ...")
                    self.logger.info("No GPU found, processing on CPU ...")
                    device = "cpu"
            else:
                print("Processing on {} ...".format(str(device)))

            # drop columns if they are None
            if None in selected_features_extractor.columns:
                selected_features_extractor = selected_features_extractor.drop([None], axis=1)

            # remove columns with config_ or Raw_
            selected_features_extractor = selected_features_extractor.loc[:, ~selected_features_extractor.columns.str.startswith('config_')]
            selected_features_extractor = selected_features_extractor.loc[:, ~selected_features_extractor.columns.str.startswith('Raw_')]
            
            if out_folder is None:
                out_folder = self.out_path + extractor + "/prediction"
            elif set_outfolder_None:
                out_folder = self.out_path + extractor + "/prediction"
            else:
                if not out_folder.endswith("/"):
                    out_folder = out_folder + "/"

                if not repeated_out_folder.endswith("/"):
                    repeated_out_folder = repeated_out_folder + "/"

                # in order to not have stacked predition folder
                out_folder = repeated_out_folder + extractor + "/prediction"

            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            if reuse_split:
                print("Starting Training with data split from feature selection.") 
                self.logger.info("Starting Training with new data split from feature selection.") 
                
                # get same train IDs as from selection
                train_id_path = self.out_path + extractor + "/selected_features/Train_idx_" + str(self.RunID) + "_" + str(self.rand_state) + ".csv"
                test_id_path = self.out_path + extractor + "/selected_features/Test_idx_" + str(self.RunID) + "_" + str(self.rand_state) + ".csv"
                val_id_path = self.out_path + extractor + "/selected_features/Val_idx_" + str(self.RunID) + "_" + str(self.rand_state) + ".csv"

                if os.path.isfile(train_id_path):
                    if "ID" in pd.read_csv(train_id_path):
                        print("Loading Training data ...")
                        self.train_ids = pd.read_csv(train_id_path)["ID"].to_list()
                    else:
                        self.train_ids = None
                        self.error.warning("No Training data found.")

                if os.path.isfile(test_id_path):
                    if "ID" in pd.read_csv(test_id_path):
                        print("Loading Testing data ...")
                        self.test_ids = pd.read_csv(test_id_path)["ID"].to_list()
                    else:
                        self.test_ids = None
                        self.error.warning("No Testing data found.")

                if os.path.isfile(val_id_path):
                    if "ID" in pd.read_csv(val_id_path):
                        print("Loading Validaion data ...")
                        self.val_ids = pd.read_csv(val_id_path)["ID"].to_list()
                    else:
                        self.val_ids = None
                        self.error.warning("No Validation data found.")
            else:
                if not seed is None:
                    print("Starting Training with new data split with seed {}.".format(seed)) 
                    self.logger.info("Starting Training with new data split with seed {}.".format(seed)) 
                else:
                    seed = self.rand_state
                    print("Starting Training with new data split using seed {}.".format(seed)) 
                    self.logger.info("Starting Training with new data split using seed {}.".format(seed)) 
            
            if repeat_prediction:
                for rand in seed:

                    out_folder = self.out_path + extractor + "/prediction/seed_" + str(rand)

                    # check if all folder are there
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)

                    if not os.path.exists(out_folder + "/models"):
                         os.makedirs(out_folder + "/models")

                    if not os.path.exists(out_folder + "/plots"):
                         os.makedirs(out_folder + "/plots")

                    self.model_trainer = ModelTrainer(data=selected_features_extractor.copy(),
                                             Prediction_Label=self.Prediction_Label,
                                             selected_features_path=self.out_path + extractor + "/selected_features/",
                                             out_folder=out_folder,
                                             ensemble=ensemble,
                                             model_save_dir=out_folder + "/models",
                                             plot_save_dir=out_folder + "/plots",
                                             model=model,  # model to be used or list of models to be used
                                             use_cross_validation=use_cross_validation,  # use cross validation
                                             cross_val_splits=cross_val_splits,  # number of splits
                                             train_idx=self.train_ids,  # ID of training samples is specific samples are needed
                                             test_idx=self.test_ids,  # ID of test samples is specific samples are needed
                                             val_idx=self.val_ids,  # ID of validate samples is specific samples are needed
                                             logger=self.logger,  # logger for log
                                             error=self.error,  # logger for error
                                             RunID=self.RunID,  # RunID for selection of the Run
                                             n_cpus=self.n_cpu,  # number of cpus to be used
                                             log_file_path=log_file_path,
                                             device=device,
                                             shap_analysis=shap_analysis,
                                             run_neptune=run_neptune,
                                             neptune_api_token=neptune_api_token,
                                             neptune_project=neptune_project,
                                             stable_pretraining=stable_pretraining,
                                             neptune_run_name=neptune_run_name,
                                             optimization_iter=optimization_iter,
                                             rand_state=rand,
                                             optimize=optimize,
                                             extended_parameter_set=extended_parameter_set,
                                             super_extended_parameter_set=super_extended_parameter_set,
                                             predict_only = self.predict_only,
                                             model_path = self.model_path,
                                             imbalance_method = self.imbalance_method,
                                             autoradiomics_splits_path=autoradiomics_splits_path,
                                             )
            else:
                # check if all folder are there
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)

                if not os.path.exists(out_folder + "/models"):
                     os.makedirs(out_folder + "/models")

                if not os.path.exists(out_folder + "/plots"):
                     os.makedirs(out_folder + "/plots")

                self.model_trainer = ModelTrainer(data=selected_features_extractor.copy(),
                                                Prediction_Label=self.Prediction_Label,
                                                selected_features_path=self.out_path + extractor + "/selected_features/",
                                                out_folder=out_folder,
                                                ensemble=ensemble,
                                                model_save_dir=out_folder + "/models",
                                                plot_save_dir=out_folder + "/plots",
                                                model=model,  # model to be used or list of models to be used
                                                use_cross_validation=use_cross_validation,  # use cross validation
                                                cross_val_splits=cross_val_splits,  # number of splits
                                                train_idx=self.train_ids,  # ID of training samples is specific samples are needed
                                                test_idx=self.test_ids,  # ID of test samples is specific samples are needed
                                                val_idx=self.val_ids,  # ID of validate samples is specific samples are needed
                                                logger=self.logger,  # logger for log
                                                error=self.error,  # logger for error
                                                RunID=self.RunID,  # RunID for selection of the Run
                                                n_cpus=self.n_cpu,  # number of cpus to be used
                                                log_file_path=log_file_path,
                                                device=device,
                                                shap_analysis=shap_analysis,
                                                run_neptune=run_neptune,
                                                neptune_api_token=neptune_api_token,
                                                neptune_project=neptune_project,
                                                stable_pretraining=stable_pretraining,
                                                neptune_run_name=neptune_run_name,
                                                optimization_iter=optimization_iter,
                                                rand_state=seed[0],
                                                optimize=optimize,
                                                extended_parameter_set=extended_parameter_set,
                                                predict_only = self.predict_only,
                                                model_path = self.model_path,
                                                imbalance_method = self.imbalance_method,
                                                autoradiomics_splits_path=autoradiomics_splits_path,
                                                )

            self.model_trainer.train()

        print(3*"#","RPTK Finished Prediction for {}".format(str(selected_features.keys())),3*"#")
        self.logger.info("RPTK Finished Prediction for {}".format(str(selected_features.keys())))

    def is_segmentation_3d(self, nifti_path: str, label: int = 1) -> bool:
        """
        Return True iff the segmentation has label==1 on >1 axial (z) slices.
        This filters out single-slice segmentations even if they cover many x/y pixels.

        Parameters
        ----------
        nifti_path : str
            Path to a .nii or .nii.gz file.
        label : int, default 1
            Foreground label to check.

        Returns
        -------
        bool
            True if label occupies at least two distinct z-slices; else False.
        """
        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"File not found: {nifti_path}")

        img = nib.load(nifti_path)
        data = img.get_fdata(dtype=np.float32)

        # Collapse 4D if needed (presence in any frame)
        if data.ndim == 4:
            data = data.max(axis=-1)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D/4D NIfTI; got shape {data.shape}")

        # Robust to NaNs
        np.nan_to_num(data, copy=False, nan=0.0)

        # Foreground == exact label (use np.isclose if your pipeline writes non-integers)
        fg = (data == float(label))

        # Count occupied slices along each axis
        occupied_counts = []
        for axis in (0, 1, 2):
            reduce_axes = tuple(a for a in (0, 1, 2) if a != axis)
            occupied = np.any(fg, axis=reduce_axes).sum()
            occupied_counts.append(int(occupied))

        # Fully 3D only if every axis has >1 occupied slice
        return all(c > 1 for c in occupied_counts)

    def run(self):
        """
        Run the whole RPTK pipeline
        """

        # get configuration
        self.get_rptk_config()
            
        # create folders
        self.create_folders()

        # check input file format
        self.check_input_csv_format()

        # if ID is a number make it a string
        if is_numeric_dtype(self.data["ID"]):
            self.data["ID"] = self.data["ID"].astype(str)

        # get data fingerprint TODO: integrate data fingerprint based optimization into feature extraction
        # get Data statistics
        self.get_data_fingerprint()

        # Data preprocessing (image transformations, segmentation perturbations, segmentation evaluation, and Data Fingerprint extraction
        self.preprocessing()

        # extract features (with PyRadiomics and/or MIRP - use Data fingerprint for optimizing feature extraction)
        self.extract_features()

        # filter features
        self.filter_features()

        # select features
        self.select_features()

        # predict on selected features
        self.predict() 
