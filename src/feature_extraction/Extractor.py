from __future__ import print_function

import os
import sys
from threading import Semaphore
import psutil

import SimpleITK as sitk
import numpy as np
import pandas as pd
import yaml
import json
from iteration_utilities import duplicates
from p_tqdm import p_map
import time
import datetime
import logging
import subprocess as subp
import gc

from pathlib import Path
import matplotlib.pyplot as plt

from multiprocessing.pool import Pool
# from loaders import BarLoader, SpinningLoader, TextLoader
import itertools

# sys.path.append('src')

# from src.feature_filtering.Radiomics_Filter_exe import RadiomicsFilter
from rptk.src.segmentation_processing.SegProcessor import SegProcessor
from rptk.src.feature_extraction.Outfile_checker import Outfile_validator, MissingExperimentDetector
from rptk import rptk
from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.feature_filtering.ibsi_feature_formater import IBSIFeatureFormater
# from feature_extraction.Outfile_checker import

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.mirp.imageClass import *
from rptk.mirp.roiClass import *

import radiomics
from radiomics import *

import memory_profiler as mp
from memory_profiler import profile

# import mirp_predict_lib as mirp_predict
# from mirp_predict_lib import *

# from mirp_pipeline.Preprocessor import Preprocessor
# from mirp_pipeline.Experiment_setup import *

from tqdm import *
import glob
import re
import multiprocessing
from functools import partial

from threading import Thread
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import concurrent.futures
import traceback

class Extractor:
    """
    Extract Radiomics features with PyRadiomics or MIRP Pipeline.
    Programmed by Jonas Bohn (2021). No Warranty for usage.
    :param out_path: Path to output directory.
    :param n_cpu: Number of CPUs to use.
    :param path2confCSV: Path to CSV file with Image and Segmentation paths.
    :param modality: Modality of images (CT or MRI).
    :param extraction_yaml: Path to yaml file for custom extraction (default in config folder).
    :param variance_threshold: Variance threshold for feature filtering.
    :param correlation_threshold: Correlation threshold for feature filtering.
    :param delta: Delta Radiomics calculation (yes/no).
    :param time_format: Time format for delta calculation if not single value.
    :param cohort_ID: Cohort ID for delta calculation.
    :param extractor: Either PyRadiomics or MIRP.
    :param chunksize: Chunksize for multiprocessing.
    :param verbose: Verbose output level (0=None, 1=Some, 2=Everything).
    :param RunID: Run ID for logging.
    :param extracted_features_tmp_dir: Temporary directory.
    :param logs_dir: Log directory.
    :param extracted_features_dir: Feature directory.
    :param self_optimize: Self-optimization of PyRadiomics or MIRP.
    :param logger: Logger for logging
    :param error: Error logger for error reporting
    :param use_previous_output: Use previous output if experiments have been applied already.
    :param rptk_config_json: Path to RPTK config.
    :param fast_mode: Speed up feature extraction but higher mem consumption.
    :param resegmentation: Resegmentation of segmetnations.
    :param take_label_changes: Take label changes into account over a timeseries.
    """

    def __init__(self,
                 extractor: str,  # Either PyRadiomics or MIRP
                 out_path: str,  # Path to output directory
                 n_cpu: int = 1,  # Number of CPUs to use
                 chunksize: int = 10,  # Chunksize for multiprocessing
                 path2confCSV: str = "",  # Path to CSV file with Image and Segmentation paths
                 modality: str = "",  # CT or MRI
                 extraction_yaml: str = None,  # Path to yaml file for custom extraction (default in config folder)
                 # variance_threshold: float = 0.1,  # Variance threshold for feature filtering
                 # correlation_threshold: float = 0.9,  # Correlation threshold for feature filtering
                 delta: bool = False,  # Delta Radiomics calculation
                 time_format: str = "%Y%m%d",  # Time format for delta calculation if not single value
                 cohort_ID: str = "",  # Cohort ID for delta calculation
                 verbose: int = 1,  # Verbose output level (0=None, 1=Some, 2=Everything)
                 RunID: str = time.strftime("%Y%m%d-%H%M%S"),  # Run ID for logging
                 extracted_features_tmp_dir: str = None,  # Temporary directory
                 logs_dir: str = None,  # Log directory
                 extracted_features_dir: str = None,  # Feature directory
                 self_optimize: bool = False,  # Self-optimization of PyRadiomics or MIRP
                 logger=None,  # Logger
                 error=None,
                 use_previous_output: bool = False,  # Use previous output
                 rptk_config_json: str = None,  # Path to RPTK config
                 fast_mode: bool = False,  # Speed up feature extraction but higher mem consumption
                 resegmentation: bool = True,
                 take_label_changes: bool = False,
                 ):

        self.extractor = extractor
        self.out_path = out_path
        self.n_cpu = n_cpu
        self.chunksize = chunksize
        self.path2confCSV = path2confCSV
        self.modality = modality
        self.extraction_yaml = extraction_yaml
        # self.variance_threshold = variance_threshold
        # self.correlation_threshold = correlation_threshold
        self.time_format = time_format
        self.delta = delta
        self.cohort_ID = cohort_ID
        self.verbose = verbose
        self.RunID = RunID
        self.extracted_features_tmp_dir = extracted_features_tmp_dir
        self.logs_dir = logs_dir
        self.extracted_features_dir = extracted_features_dir
        self.self_optimize = self_optimize
        self.logger = logger
        self.error = error
        self.use_previous_output = use_previous_output
        self.rptk_config_json = rptk_config_json
        self.fast_mode = fast_mode
        self.resegmentation = resegmentation
        self.take_label_changes = take_label_changes

        # self.RunID = time.strftime("%Y%m%d-%H%M%S")

        self.done_extraction_out_files = []

        if not self.out_path.endswith("/"):
            self.out_path = self.out_path + "/"

        # check out directory
        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(self.out_path)

        # Config Logger #
        # if self.logger is None:
        if self.logs_dir is None:
            self.logger = LogGenerator(
                log_file_name=self.out_path + "extracted_features/RPTK_feature_extraction_" + self.extractor + "_" + self.RunID + ".log",
                logger_topic="RPTK " + self.extractor + " Feature Extraction"
            ).generate_log()

            self.error = LogGenerator(
                log_file_name=self.out_path + "extracted_features/RPTK_feature_extraction_" + self.extractor + "_" + self.RunID + ".err",
                logger_topic="RPTK Feature Extraction Error"
            ).generate_log()
        else:
            self.logger = LogGenerator(
                log_file_name=self.logs_dir + "RPTK_feature_extraction_" + self.extractor + "_" + self.RunID + ".log",
                logger_topic="RPTK " + self.extractor + " Feature Extraction"
            ).generate_log()

            self.error = LogGenerator(
                log_file_name=self.logs_dir + "RPTK_feature_extraction_" + self.extractor + "_" + self.RunID + ".err",
                logger_topic="RPTK " + self.extractor + " Feature Extraction Error"
            ).generate_log()

        # Check dirs for output
        if self.extracted_features_tmp_dir is None:
            self.extracted_features_tmp_dir = os.path.join(self.out_path, "extracted_features/tmp/")

        # if self.logs_dir is None:
        #    self.logs_dir = os.path.join(self.out_path, "logs/")

        if self.extracted_features_dir is None:
            self.extracted_features_dir = os.path.join(self.out_path, "extracted_features/")

        if self.extracted_features_dir + "failed_extractions.csv" in glob.glob(self.extracted_features_dir + "*.csv"):
            self.failed_extractions = pd.read_csv(self.extracted_features_dir + "failed_extractions.csv", index_col=0)
        else:
            self.failed_extractions = pd.DataFrame()

        self.subject_dir = os.path.join(self.extracted_features_tmp_dir, "subjects/")

        self.outfile = self.extracted_features_dir + str(self.extractor) + "_extraction_" + str(self.RunID) + ".csv"
        # self.tmp_outfile = None
        #
        # if self.extractor == "PyRadiomics":
        #     # check if oufile exists already
        #     if os.path.isfile(self.outfile):
        #         # put output in subjects folder
        #         for i in range(pd.read_csv(self.path2confCSV).shape[0]):
        #             self.tmp_outfile = self.subject_dir + str(self.extractor) + "_extraction_" + str(self.RunID) + "_" + str(i) + ".csv"
        #             if not os.path.isfile(self.tmp_outfile):
        #                 break

        # check tmp directory
        if not os.path.exists(os.path.dirname(self.extracted_features_tmp_dir)):
            os.makedirs(self.extracted_features_tmp_dir)

        # check subject directory
        if not os.path.exists(os.path.dirname(self.subject_dir)):
            os.makedirs(self.subject_dir)

        # check log directory
        # if not os.path.exists(os.path.dirname(self.logs_dir)):
        #    os.makedirs(self.logs_dir)

        # check feature directory
        if not os.path.exists(os.path.dirname(self.extracted_features_dir)):
            os.makedirs(self.extracted_features_dir)

        # check if extraction yaml exists
        self.df = pd.DataFrame()

        if self.n_cpu == 0 or self.n_cpu == multiprocessing.cpu_count():
            self.n_cpu = multiprocessing.cpu_count() - 1

        if self.extraction_yaml is None:
            if self.modality == "CT":
                if self.extractor == "PyRadiomics":
                    self.extraction_yaml = os.path.dirname(os.path.dirname(__file__)) + "/config/PyRadiomics/CT.yaml"
                elif self.extractor == "MIRP":
                    self.extraction_yaml = os.path.dirname(os.path.dirname(__file__)) + "/config/MIRP/CT.yaml"
                else:
                    raise ValueError("Extractor not supported. Choose PyRadiomics or MIRP.")

            elif self.modality == "MRI" or self.modality == "MR":
                if self.extractor == "PyRadiomics":
                    self.extraction_yaml = os.path.dirname(os.path.dirname(__file__)) + "/config/PyRadiomics/MRI.yaml"
                elif self.extractor == "MIRP":
                    self.extraction_yaml = os.path.dirname(os.path.dirname(__file__)) + "/config/MIRP/MRI.yaml"
                else:
                    raise ValueError("Extractor not supported. Choose PyRadiomics or MIRP.")
            else:
                raise ValueError("Modality not supported. Please choose CT or MRI.")
        else:
            if not os.path.exists(self.extraction_yaml):
                raise ValueError("Yaml file not found.")
            else:
                self.logger.info("Loading extraction config from config file.")

        if self.rptk_config_json is None:
            self.rptk_config_json = os.path.dirname(os.path.dirname(__file__)) + "/config/rptk_config.json"

        self.extractor_setting = {}

        # Read YAML file
        # with open(os.path.abspath(self.extraction_yaml), 'r') as stream:
        #     self.extractor_setting = yaml.safe_load(stream)
        # Read json file
        with open(self.rptk_config_json, 'r') as f:
            self.rptk_config = json.load(f)

        self.path2fingerprint = os.path.abspath(os.path.join(os.path.dirname(self.out_path),
                                                             '..',
                                                             'preprocessed_data',
                                                             'Data_stats')) + "/Input_Data_stats.csv"
        # get data fingerprint
        self.data_fingerprint = pd.read_csv(self.path2fingerprint)
        # put ID to the fingerprint
        self.data_fingerprint['ID'] = self.data_fingerprint.Image.apply(lambda x: str(os.path.basename(x).split('_')[0]))

        # -

    def get_bin_borders(self):
        """
        Optimize bin number with data from the fingerprint
        :parameter path2fingerprint: path to csv file with data fingerprint
        """

        if "ID" not in self.data_fingerprint.columns:
            # put ID to the fingerprint
            self.data_fingerprint['ID'] = self.data_fingerprint.Image.apply(lambda x: str(os.path.basename(x).split('_')[0]))

        # get num bins upper (95.0%) and lower quantile (5.0%)
        min_bin_number, max_bin_number = np.quantile(self.data_fingerprint['Number_of_bins'].values, [0.05, 0.95])

        if min_bin_number < self.rptk_config["Feature_extraction_config"]["min_num_bin"]:
            self.min_bin_number = int(self.rptk_config["Feature_extraction_config"]["min_num_bin"])
        else:
            self.min_bin_number = int(min_bin_number)

        if max_bin_number > self.rptk_config["Feature_extraction_config"]["max_num_bin"]:
            self.max_bin_number = int(self.rptk_config["Feature_extraction_config"]["max_num_bin"])
        else:
            self.max_bin_number = int(max_bin_number)

    def get_info_from_csv(self, path2csv: str):
        """
        Get all information for extraction from csv file
        :parameter str: path2csv: path to csv file with image and segmentation paths
        """
        #loader = SpinningLoader(text="Reading CSV file")
        
        self.logger.info("Reading CSV file.")
        # print("Reading CSV file.")

        df = pd.read_csv(path2csv, sep=',')

        # if df has another separator try to read it with ;
        if len(df.columns) < 3:
            df = pd.read_csv(path2csv, sep=";")

        # drop columns with NAs
        nans = df.columns[df.isna().all()].tolist()
        if len(nans) > 0:
            self.error.warning("Dropping the following columns containing NANs: " + str(nans))
            # drop columns which only contain NANs
            df = df.drop(columns=nans)
            # df = df.dropna(axis=1, how='all')

        # drop rows with NANs
        nans = df.index[df.isna().all(1)].tolist()
        if len(nans) > 0:
            self.error.warning("The following rows contain NANs: " + str(nans))
            self.logger.info("Dropping rows with NANs.")

            # first drop columns which only contain NANs
            df = df.dropna(axis=0, how='all')

        df = SegProcessor(logger=self.logger,
                          error=self.error,
                          out_path=self.out_path + "extracted_features/",
                          RunID=self.RunID).check_format(df=df)
        
        #loader.start()
        
        only_nan_in_columns = df.loc[:, df.isnull().all()].columns.tolist()

        if len(only_nan_in_columns) > 0:
            self.error.warning("The following columns contain NANs after Format check: " + str(only_nan_in_columns))
            df.drop(columns=only_nan_in_columns, inplace=True)

        # self.logger.info("Size of Data: " + str(df.shape))

        self.logger.info("Removing duplicates from Configuration File.")
        # generate Index with Image name and Mask name for unique config string
        image = df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
        mask = df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

        id = []
        for x, y in zip(image, mask):
            id.append(x + "_" + y)

        ID = pd.Series(index=df.index, data=id)
        df.set_index(ID, inplace=True)

        # df.index.rename("img_data_config", inplace=True)
        df.drop_duplicates(inplace=True)

        # self.logger.info("Removing Mask and Image Transformations from Configuration File.")
        # drop all columns with mask transformation and image transformation --> not needed
        # df_mask_image_transform = df.loc[~df["Mask_Transformation"].isnull()]
        # df_mask_image_transform = df_mask_image_transform.loc[
        #    ~df_mask_image_transform["Image_Transformation"].isnull()]

        # df.drop(index=df_mask_image_transform.index.values.tolist(), inplace=True)
        if "Image_Transformation" in df.columns:
            if ("N4BF_normalized" in df["Image_Transformation"].values) or (
                    "Z-score_normalized" in df["Image_Transformation"].values):
                df = self.add_normalized_image_transformation_prefix(df=df)

        self.logger.info("Size of Data: " + str(df.shape))

        # self.logger.info("Removing Unnamed Columns:", )
        #loader.stop()
        
        return df

    def drop_columns_by_keyword(self, df, keyword="diagnostics_Mask-corrected"):
        """
        Drops all columns containing a specific keyword in their name.

        Parameters:
        - df: pandas DataFrame
        - keyword: String keyword to search for in column names (default: "diagnostics_Mask-corrected")

        Returns:
        - New DataFrame without the matching columns.
        - List of dropped column names.
        """
        # Identify columns to drop
        cols_to_drop = [col for col in df.columns if keyword in col]

        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns containing {keyword}: {cols_to_drop}")#
            self.error.warning(f"Dropping {len(cols_to_drop)} columns containing {keyword}: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        return df

    def get_extractor_config(self):
        """
        Get configuration from yaml file.
        """
        self.logger.info("Reading " + self.extractor + " configuration from " + str(self.extraction_yaml))
        with open(self.extraction_yaml, 'r') as file:
            config = yaml.safe_load(file)

        return config

    def generate_mirp_settings(self, experiment: pd.Series):
        """
        Generates the settings for the mirp feature extraction reading from yaml config file.
        :return: settings: dict with all settings for the mirp feature extraction
        """
        self.experiment = experiment
        self.logger.info("Generating settings for MIRP Pipeline.")

        if len(self.extractor_setting) == 0:
            self.extractor_setting = self.get_extractor_config()

        change_extraction_setting = False

        # Default
        base_discretisation_method = self.extractor_setting["feature_extraction_parameters"]["base_discretisation_method"]
        base_discretisation_n_bins = self.extractor_setting["feature_extraction_parameters"]["base_discretisation_bin_number"]
        ivh_discretisation_method = self.extractor_setting["feature_extraction_parameters"]["ivh_discretisation_method"]
        ivh_discretisation_n_bins = self.extractor_setting["feature_extraction_parameters"]["ivh_discretisation_bin_number"]

        # Get optimal Feature computation settings
        if self.self_optimize:
            
            # check if ID is a number and check if it is the same as the ID from the Image/Mask
            if str(experiment["ID"]).isnumeric():
                img_ID =  str(os.path.basename(experiment["Image"]).split('_')[0])
                if img_ID != experiment["ID"]:
                    experiment["ID"] = str(img_ID)
            
            # enable normalization on imaging
            

            # optimize pixel dicretization based on number of bins in ROI
            number_of_bins = int(self.data_fingerprint.loc[
                                     self.data_fingerprint["ID"] == str(experiment["ID"]), "Number_of_bins"].to_list()[
                                     0])
            if not hasattr(self, 'min_bin_number'):
                self.get_bin_borders()

            # get the number of bins for the sample
            if number_of_bins < self.min_bin_number:
                self.logger.info(
                    "Need to optimize settings for " + str(os.path.basename(experiment["Image"])) + " to " +
                    str(self.min_bin_number) + " as " + str(number_of_bins) + " is very small!")

                base_discretisation_method = "fixed_bin_number"
                base_discretisation_n_bins = int(self.min_bin_number)
                ivh_discretisation_method = "fixed_bin_number"
                ivh_discretisation_n_bins = int(self.min_bin_number)

                change_extraction_setting = True

            elif number_of_bins > self.max_bin_number:
                self.logger.info(
                    "Need to optimize settings for " + str(os.path.basename(experiment["Image"])) + " to " +
                    str(self.max_bin_number) + " as " + str(number_of_bins) + " is very large!")

                base_discretisation_method = "fixed_bin_number"
                base_discretisation_n_bins = int(self.max_bin_number)
                ivh_discretisation_method = "fixed_bin_number"
                ivh_discretisation_n_bins = int(self.max_bin_number)

                change_extraction_setting = True

        if change_extraction_setting:

            # Generate feature extraction settings
            feature_computation_parameters = FeatureExtractionSettingsClass(
                by_slice=self.extractor_setting["feature_extraction_parameters"]["by_slice"],
                no_approximation=self.extractor_setting["feature_extraction_parameters"]["no_approximation"],
                ibsi_compliant=self.extractor_setting["feature_extraction_parameters"]["ibsi_compliant"],
                base_feature_families=self.extractor_setting["feature_extraction_parameters"]["base_feature_families"],
                base_discretisation_method=base_discretisation_method,
                base_discretisation_bin_width=base_discretisation_n_bins,
                base_discretisation_n_bins=base_discretisation_n_bins,
                ivh_discretisation_method=ivh_discretisation_method,
                ivh_discretisation_bin_width=float(self.extractor_setting["feature_extraction_parameters"]["ivh_discretisation_bin_number"]),
                ivh_discretisation_n_bins=ivh_discretisation_n_bins,
                glcm_distance=self.extractor_setting["feature_extraction_parameters"]["glcm_distance"],
                glcm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["glcm_spatial_method"],
                glrlm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["glrlm_spatial_method"],
                glszm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["glszm_spatial_method"],
                gldzm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["gldzm_spatial_method"],
                ngtdm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["ngtdm_spatial_method"],
                ngldm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["ngldm_spatial_method"],
                ngldm_distance=self.extractor_setting["feature_extraction_parameters"]["ngldm_distance"],
                ngldm_difference_levels=self.extractor_setting["feature_extraction_parameters"]["ngldm_difference_level"],
            )

        else:
            # Generate feature extraction settings
            feature_computation_parameters = FeatureExtractionSettingsClass(
                by_slice=self.extractor_setting["feature_extraction_parameters"]["by_slice"],
                no_approximation=self.extractor_setting["feature_extraction_parameters"]["no_approximation"],
                ibsi_compliant=self.extractor_setting["feature_extraction_parameters"]["ibsi_compliant"],
                base_feature_families=self.extractor_setting["feature_extraction_parameters"]["base_feature_families"],
                base_discretisation_method=self.extractor_setting["feature_extraction_parameters"]["base_discretisation_method"],
                base_discretisation_bin_width=self.extractor_setting["feature_extraction_parameters"]["base_discretisation_bin_number"],
                ivh_discretisation_method=self.extractor_setting["feature_extraction_parameters"]["ivh_discretisation_method"],
                ivh_discretisation_bin_width=float(self.extractor_setting["feature_extraction_parameters"]["ivh_discretisation_bin_number"]),
                glcm_distance=self.extractor_setting["feature_extraction_parameters"]["glcm_distance"],
                glcm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["glcm_spatial_method"],
                glrlm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["glrlm_spatial_method"],
                glszm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["glszm_spatial_method"],
                gldzm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["gldzm_spatial_method"],
                ngtdm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["ngtdm_spatial_method"],
                ngldm_spatial_method=self.extractor_setting["feature_extraction_parameters"]["ngldm_spatial_method"],
                ngldm_distance=self.extractor_setting["feature_extraction_parameters"]["ngldm_distance"],
                ngldm_difference_levels=self.extractor_setting["feature_extraction_parameters"]["ngldm_difference_level"],
            )

        # self.rptk_config
        # self.data_fingerprint
        # Generate settings
        general_settings = GeneralSettingsClass(
            by_slice=self.extractor_setting["general_parameters"]["by_slice"]
        )

        # Generate interpolate settings
        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=self.extractor_setting["general_parameters"]["by_slice"],
            interpolate=self.extractor_setting["image_interpolation_parameters"]["interpolate"],
            anti_aliasing=self.extractor_setting["image_interpolation_parameters"]["anti_aliasing"]
        )

        # Generate perturbation settings
        perturbation_settings = ImagePerturbationSettingsClass()

        # Generate image transformation settings
        image_transformation_settings = ImageTransformationSettingsClass(
            by_slice=False,
            response_map_feature_settings=feature_computation_parameters)

        if self.resegmentation:
            # Generate resegmentation settings
            if self.extractor_setting["resegmentation_parameters"]["resegmentation_method"] == "threshold":
                resegmentation_settings = ResegmentationSettingsClass(
                    resegmentation_method=self.extractor_setting["resegmentation_parameters"]["resegmentation_method"],
                    resegmentation_intensity_range=self.extractor_setting["resegmentation_parameters"]["resegmentation_intensity_range"],
                    resegmentation_sigma=self.extractor_setting["resegmentation_parameters"]["resegmentation_sigma"]
                )
            elif self.extractor_setting["resegmentation_parameters"]["resegmentation_method"] == "range":
                resegmentation_settings = ResegmentationSettingsClass(
                    resegmentation_method=self.extractor_setting["resegmentation_parameters"]["resegmentation_method"],
                    resegmentation_sigma=self.extractor_setting["resegmentation_parameters"]["resegmentation_sigma"]
                )
            else:
                if self.extractor_setting["resegmentation_parameters"]["resegmentation_method"] == "none":
                    # self.logger.info("Do not perform Resegmentation for MIRP extraction!")
                    resegmentation_settings = ResegmentationSettingsClass()
                else:
                    self.error.error("Can not perform Resegmentation for MIRP extraction! Unknown resegmentation method in MIRP Settings! Please select threshold or range.")
                    raise ValueError("Can not perform Resegmentation for MIRP extraction! Unknown resegmentation method in MIRP Settings! Please select threshold or range.")
        else:
            # self.logger.info("Do not perform Resegmentation for MIRP extraction!")
            resegmentation_settings = ResegmentationSettingsClass()

        # Generate roi interpolation settings
        roi_interpolate_settings = RoiInterpolationSettingsClass(
            roi_spline_order=self.extractor_setting["roi_interpolate_parameters"]["roi_spline_order"],
            roi_interpolation_mask_inclusion_threshold=self.extractor_setting["roi_interpolate_parameters"]["roi_interpolation_mask_inclusion_threshold"],
        )

        image_processing_setting = ImagePostProcessingClass(
            intensity_normalisation=self.extractor_setting["image_postprocessing"]["intensity_normalisation"]
            )

        # Summarize settings
        settings = SettingsClass(
            general_settings=general_settings,
            post_process_settings=image_processing_setting,
            img_interpolate_settings=image_interpolation_settings,
            roi_interpolate_settings=roi_interpolate_settings,
            roi_resegment_settings=resegmentation_settings,
            perturbation_settings=perturbation_settings,
            img_transform_settings=image_transformation_settings,
            feature_extr_settings=feature_computation_parameters
        )

        return settings

    def generate_mirp_experiments(self): # TODO Adapt to new mirp version
        """
        Generates a list of experiments for the MIRP pipeline.
        """
        #loader = SpinningLoader(text="Config MIRP extraction")
        #loader.start()
        
        experiments = []

        self.logger.info("Generating experiments for MIRP Pipeline.")

        for index, row_ in self.df.iterrows():

            # get setting
            setting = self.generate_mirp_settings(experiment=row_)

            # configure subject as it is too long data name if we have cropped scans
            subject = os.path.basename(row_["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row_["Mask"])[
                                                                                :-len(".nii.gz")]
            if "_cropped_resample" in subject:
                subject = os.path.basename(row_["Image"])[:-len(".nii.gz")]

            experiment = ExperimentClass(
                modality=self.modality,
                subject=subject,
                cohort=self.cohort_ID,
                write_path=self.subject_dir,
                image_folder=os.path.dirname(row_["Image"]),
                roi_folder=os.path.dirname(row_["Mask"]),
                roi_reg_img_folder=None,
                image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                registration_image_file_name_pattern=None,
                roi_names=[os.path.basename(row_["Mask"])[:-len(".nii.gz")]],
                data_str=[],
                # [os.path.basename(row_["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row_["Mask"])[
                #:-len(".nii.gz")]],
                provide_diagnostics=self.extractor_setting["experiment_parameters"]["provide_diagnostics"],
                settings=setting,
                compute_features=self.extractor_setting["experiment_parameters"]["compute_features"],
                extract_images=self.extractor_setting["experiment_parameters"]["extract_images"],
                plot_images=self.extractor_setting["experiment_parameters"]["plot_images"],
                keep_images_in_memory=False # self.extractor_setting["experiment_parameters"]["keep_images_in_memory"]
            )

            experiments.append(experiment)
        del experiment
        
        #loader.stop()
        
        return experiments

    def execute_mirp_experiment(self, experiment):
        """
        Executes a single experiment.
        """

        self.logger.info("Processing Features from " + str(experiment.subject))
        # print("Processing Features from " + str(experiment.subject))
        try:
            experiment.process()
        except Exception as exp:
            self.error.error("Error processing experiment: " + str(experiment.subject) + ": " + str(exp))
            self.error.error(traceback.format_exc())

        # del experiment
        return experiment

    def make_image_mask_index(self, df: pd.DataFrame):
        """
        Creates a pd.Dataframe index from the image and mask filenames.
        :param df:
        :return: df with index of image and mask filenamesReading csv files
        """

        if "Image" not in df.columns or "Mask" not in df.columns:
            self.error.error("Image and/or mask column not found in dataframe. Check the input data.")
            raise ValueError("Image and/or mask column not found in dataframe. Check the input data.")

        df['Image'] = df['Image'].astype(str)
        df['Mask'] = df['Mask'].astype(str)

        try:
            image = df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
            mask = df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
        except Exception as e:
            self.error.error(e)
            raise e

        id = []
        for x, y in zip(image, mask):
            id.append(x + "_" + y)

        ID = pd.Series(index=df.index, data=id)
        df.set_index(ID, inplace=True)

        return df

    def check_config_index_sync(self, idx, df):
        """
        Check if index should be based on cropped samples or not
        :param df: pd.DataFrame with configuration of the samples (Image, Mask, Image_Transformation ...)
        :param idx: str with index from extracted sample
        """

        result = None

        if ("Raw_Image" in df.columns) and ("Raw_Mask" in df.columns):
            if "id_subject" not in df.columns:
                df.index = df["id_subject"]

            if idx not in df["id_subject"].to_list():
                for img, msk in zip(df["Raw_Image"].to_list(), df["Raw_Mask"].to_list()):
                    img_base = os.path.basename(img)[:-len(".nii.gz")]
                    msk_base = os.path.basename(msk)[:-len(".nii.gz")]

                    if idx.startswith(img_base) and idx.endswith(msk_base):
                        if (img_base + "_" + msk_base) == idx:
                            df.loc[(df["Raw_Image"] == img) & (
                                    df["Raw_Mask"] == msk), "id_subject"] = img_base + "_" + msk_base

                            result = df.loc[(df["Raw_Image"] == img) & (df["Raw_Mask"] == msk), "id_subject"]

                            self.error.warning(
                                "Need to adapt configuration index to add configuration to feature extraction of sample {}!".format(
                                    str(idx)))
                            break

        return result

    def add_config_cols_to_df(self, experiments: pd.DataFrame, df=None):
        """
        Add config columns like Image, Mask etc to the feature space
        :param experiments: DataFrame with features where the cinfig columns needed to get added to
        :param df: DataFrame with config
        :return:
        """

        if df is None:
            df = self.df.copy()

        if self.extractor == "MIRP":
            # check for index synchronization
            if "id_subject" in experiments.columns:
                experiments.index = experiments["id_subject"]
                if "id_subject" not in df.columns:
                    df["id_subject"] = np.nan
                for i, r in df.iterrows():
                    # for cropped extraction only the name of the image is the ID
                    if "cropped" in os.path.basename(str(r["Image"])):
                        df.loc[i, "id_subject"] = os.path.basename(str(r["Image"]))[
                                                  :-(len(".nii.gz"))]
                    else:
                        df.loc[i, "id_subject"] = os.path.basename(str(r["Image"]))[
                                                  :-(len(".nii.gz"))] + "_" + os.path.basename(str(r["Mask"]))[
                                                                              :-(len(".nii.gz"))]
                # df["id_subject"] = os.path.basename(df["Image"])[:-(len(".nii.gz"))] + "_" + os.path.basename(df["Mask"])[:-(len(".nii.gz"))]

            elif experiments.index.name == "id_subject":
                for i, r in df.iterrows():
                    # for cropped extraction only the name of the image is the ID

                    df.loc[i, "id_subject"] = os.path.basename(str(r["Raw_Image"]))[
                                              :-(len(".nii.gz"))] + "_" + os.path.basename(str(r["Raw_Mask"]))[
                                                                          :-(len(".nii.gz"))]
                # df["id_subject"] = os.path.basename(df["Image"])[:-(len(".nii.gz"))] + "_" + os.path.basename(df["Mask"])[:-(len(".nii.gz"))]


            else:
                self.error.error("Could not find id_subject index in MIRP output file!")

        # check index configuration
        partial_check_index_sync = partial(self.check_config_index_sync, df=df)
        result = pd.DataFrame()

        with tqdm(total=len(experiments.index), desc="Check index configuration", unit="sample") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                for results in executor.map(partial_check_index_sync, experiments.index, chunksize=self.n_cpu):
                    if results is not None:
                        if len(result) > 0:
                            result = pd.concat([result, results])
                        else:
                            result = results
                    pbar.update(1)

        # extend df with different configuration
        df = pd.concat([result, df])

        if "id_subject" in df.columns:
            df = df.set_index('id_subject')

        if df.index.name != experiments.index.name:
            self.logger.info("Synchronizing index for concatenation!")
            print("Synchronizing MIRP extraction index for concatenation!")

            experiments.index.name = df.index.name
            
        #df.to_csv(self.out_path + "/extracted_features/tmp/index_config.csv")
        #experiments.to_csv(self.out_path + "/extracted_features/tmp/experiments.csv")
        
        # get all related columns for the experiments that are done
        experiments = pd.concat([df, experiments], axis=1)

        # dropping cols with only nan
        only_nan_in_columns = experiments.loc[:, experiments.isnull().all()].columns.tolist()

        if len(only_nan_in_columns) > 0:
            self.logger.info("Dropping columns with only nan values: " + str(only_nan_in_columns))
            experiments.drop(only_nan_in_columns, axis=1, inplace=True)

        return experiments

    def check_for_missing_experiments(self):
        """
        Checks if the outfile already exists and contains all experiments which should be processed. (PyRadiomics)
        :return: missing_samples: list of missing samples
        :return done_exps: list of performed out file names
        :return done_samples: list of done experiments ids including Image and Mask file name
        """

        done_experiments = pd.DataFrame()
        done_samples = []
        missing_samples = []
        done_exps = []  # list of file paths to done csv out files

        # file_start = str(self.extractor) + "_extraction_"
        file_end = ".csv"

        # 1. Find all outfiles from previous extraction
        if len(glob.glob(self.extracted_features_dir + "/*.csv")) > 0:
            done_exps = glob.glob(self.extracted_features_dir + "/*.csv")
            print("Found done experiments in " + str(self.extracted_features_dir))

        # if there is no final file check for temporary files
        if len(glob.glob(self.subject_dir + "/*.csv")) > 0:
            self.logger.info("Found done experiments in " + str(self.extracted_features_dir))
            done_exps += glob.glob(self.subject_dir + "/*.csv")

        # 2. Adding all out files in one dataframe (from subject folder as well as from outfile
        if len(done_exps) > 0:
            # check if all files do have any content
            for exp_ in done_exps:

                # if file is empty
                if os.path.getsize(exp_) == 0:
                    done_exps.remove(exp_)

            print("Found {} previous feature extraction!".format(len(done_exps)))
            self.logger.info("Found {} previous feature extraction!".format(len(done_exps)))

            done_experiments = self.concat_extraction(csv_files=done_exps)

        if done_experiments.empty:
            self.error.warning("No outfiles found.")

            # return empty list as there is no outfile for reading done experiments from
            return missing_samples, done_exps, done_samples
        else:
            # self.logger.info("Found " + str(len(done_experiments)) + " done experiments.")
            # print("Found " + str(len(done_experiments)) + " done experiments.")

            # check if the done experiments are indexed by the image and mask filenames
            if "id_subject" in done_experiments.columns:
                done_experiments.index = done_experiments["id_subject"]
            elif "Image" in done_experiments.columns:
                done_experiments = self.make_image_mask_index(done_experiments)
            else:
                self.error.warning("Wrong configuration of feature extraction!")
        
        if len(done_experiments) > 0:
            # sync done experiments with config file
            if "Image" not in done_experiments.columns:
                self.logger.info("Image not in done experiments!")

                done_experiments = self.add_config_cols_to_df(experiments=done_experiments)
                done_experiments = self.make_image_mask_index(done_experiments)

            for i, row in done_experiments.iterrows():
                id_ = os.path.basename(row["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row["Mask"])[
                                                                            :-len(".nii.gz")]
                done_samples.append(id_)

        if self.extractor == "MIRP":
            missing_samples = self.get_missing_experiments(done_samples, self.experiments)
        elif self.extractor == "PyRadiomics":
            missing_samples = self.get_missing_experiments(done_samples, self.df)

        else:
            self.logger.info("No outfile found.")

        return missing_samples, done_exps, done_samples

    def get_IDs_from_mirp_outfile_names(self, done_samples):

        done_exp = glob.glob(self.subject_dir + "/*.csv")
        self.logger.info("Found " + str(len(done_exp)) + " done experiments in " + str(self.subject_dir) + ".")
        # self.logger.info("Need to process  " + str(len(self.experiments) - len(done_exp)) + " experiments.")

        for exp in tqdm(done_expt, desc="Include done experiments from subjects folder", unit="sample"):
            if self.extractor == "MIRP":
                done_exp = os.path.basename(exp)[:-len("_YEAR-MO-DA_features.csv")]
            else:
                done_exp = os.path.basename(exp)[:-len(".csv")]
            if done_exp not in done_samples:
                # self.logger.info("Found new done experiment in subjects folder: " + str(done_exp))
                done_samples.append(done_exp)

        return done_samples

    def get_missing_experiments_from_subject_folder(self, done_samples):
        """
        Checks if the subject folder already exists and contains all experiments which should be processed.
        The Subject folder only contains MIRP results of each sample so far.
        :param done_samples: list of file paths of done experiments
        :return missing_experiments: list of missing experiments
        :return done_samples: list: of done experiments file names including ID without ending
        """

        self.logger.info("Checking for done experiments in " + str(self.subject_dir))

        missing_experiments = []

        if os.path.exists(self.subject_dir):
            # check if subject dir is empty
            if len(os.listdir(self.subject_dir)) > 0:
                done_samples = self.get_IDs_from_mirp_outfile_names(done_samples=done_samples)

                if self.extractor == "MIRP":
                    missing_experiments = self.get_missing_experiments(done_samples, self.experiments)
                self.logger.info("Need to process " + str(len(missing_experiments)) + " experiments.")

            else:
                self.logger.info("Subject directory (" + self.subject_dir + ") is empty.")

                if self.extractor == "MIRP":
                    if len(done_samples) > 0:
                        missing_experiments = self.get_missing_experiments(done_samples, self.experiments)
                    else:
                        self.logger.info("No experiments have been processed yet.")
                        missing_experiments = self.experiments
                self.logger.info("Need to process " + str(len(missing_experiments)) + " experiments.")

        else:
            self.logger.info("Subject directory does not exist.")
            if self.extractor == "MIRP":
                if len(done_samples) > 0:
                    missing_experiments = self.get_missing_experiments(done_samples, self.experiments)
                else:
                    self.logger.info("No experiments have been processed yet.")
                    missing_experiments = self.experiments
            self.logger.info("Need to process " + str(len(missing_experiments)) + " experiments.")

        return missing_experiments, done_samples

    def config_missing_experiments(self, missing_exp, all_experiments: list):
        """
        Returns a list of experiments that have already been processed.
        :param missing_exp: experiment that have already been processed.
        :param all_experiments: list (MIRP) or pd.DataFrame (PyRadiomics) of all experiments that should be processed.
        :return: experiments that have not already been processed.
        """

        missing_experiment = None

        if self.extractor == "MIRP":
            # for MIRP is all_experiments a list
            for experiment in all_experiments:
                if missing_exp == experiment.subject:
                    missing_experiment = experiment
                    break

        elif self.extractor == "PyRadiomics":
            # for PyRadiomics is all_experiments a pd.DataFrame
            experiments = []
            for i, row_ in all_experiments.iterrows():
                experiments.append(os.path.basename(row_["Image"][:-len(".nii.gz")]) + "_" +
                                   os.path.basename(row_["Mask"])[:-len(".nii.gz")])

            for experiment in experiments:
                if missing_exp == experiment:
                    missing_experiment = experiment
                    break

        if missing_experiment is None:
            missing_experiment = missing_exp

        return missing_experiment

    def get_missing_experiments(self, done_experiments: list, all_experiments):
        """
        Returns a list of experiments that have already been processed.
        :param done_experiments: list of experiments IDs (Image+Mask) that have already been processed in feature extraction file.
        :param all_experiments: list (MIRP) or pd.DataFrame (PyRadiomics) of all experiments that should be processed in config csv file.
        :return missing_experiments: list of experiments that have not already been processed.
        """

        # TODO: Not working!
        missing_experiments_id = []  # list to store all missing experiment IDs
        missing_experiments = []  # list to store all missing experiments

        # TODO: potential problematic if we are applying it multiple times with different done experiments --> always add done experiments up
        for i, row in self.df.iterrows():
            id_ = os.path.basename(row["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row["Mask"])[
                                                                           :-len(".nii.gz")]
            found = False

            # check if done experiment is not in csv config file ?
            for experiment in done_experiments:
                if id_ in experiment:
                    found = True
                    # self.logger.info("Experiment " + id_ + " already processed.")
                    break

            if not found:
                missing_experiments_id.append(id_)

        if len(missing_experiments_id) > 0:
            print(len(missing_experiments_id), "experiments are missing!")
            self.logger.info(str(len(missing_experiments_id)) + " experiments are missing!")

            with tqdm(total=len(missing_experiments_id), desc="Config missing experiments", unit="sample") as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                    futures = {executor.submit(partial(self.config_missing_experiments,
                                                       all_experiments=all_experiments), missing): missing for missing
                               in
                               missing_experiments_id}
                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)
                        missing_experiments.append(future.result())
                        gc.collect()  # Force garbage collection
            del futures
            
        return missing_experiments

    def concat_csv_file_to_df(self, csv_file_path: str, df: pd.DataFrame = pd.DataFrame(), ignore_index=True):
        """
        :param df: Data where the csv files should get attached
        :param csv_file_path: Path to csv file to add to the df
        :param ignore_index: concat with ignore index
        :return df: Data where the csv files got attached to
        """

        if self.extractor == "MIRP":
            if ignore_index:
                df_ = pd.read_csv(csv_file_path, sep=";")
            else:
                df_ = pd.read_csv(csv_file_path, index_col=0, sep=";")

        else:
            df_ = pd.read_csv(csv_file_path)
            if len(df_.columns) < 3:
                df_ = pd.read_csv(csv_file_path, sep=";")

        df = pd.concat([df, df_], ignore_index=ignore_index)

        return df

    def concat_extraction(self, csv_files: list, ignore_index=True):
        """
        Get all csv files from output folder and concat them into one csv file
        :param csv_files: list of csv files
        :param ignore_index: if index is syncronized or not
        :return df: concatenated pd.DataFrame
        """

        df = pd.DataFrame()
        with Pool(processes=self.n_cpu) as pool:
                # Use tqdm with imap to show progress
                for result in tqdm(pool.imap(self.concat_csv_file_to_df, csv_files), total=len(csv_files), desc="Concatenating feature files"):
                    df = pd.concat([df, result], ignore_index=ignore_index)

        self.logger.info("Concatenated all csv files from extraction into one csv file:" + str(df.shape))
        print("Concatenated all csv files from extraction into one csv file:" + str(df.shape))
        
        # drop duplicated samples
        if "Image" in df.columns:
            if df.duplicated(subset=['Image', 'Mask']).any():
                ewdf = df.drop_duplicates(subset=['Image', 'Mask'], keep='first')  # .reset_index(drop=True)
                df = ewdf
        else:
            
            df.to_csv(self.out_path +"/extracted_features/tmp/features_without_config.csv")
            
            # get configuration
            df = self.add_config_cols_to_df(experiments=df, df=None)

            if "Image" not in df.columns:
                self.error.error("Image column is missing in extracted file!")
                raise ValueError("Image column is missing in extracted file!")
            self.test = df

        # print("Image in df.columns", "Image" in df.columns)

        return df

    def add_normalized_image_transformation_prefix(self, df: pd.DataFrame):
        """
        Get all normalized Samples for MR images and add normalization as prefix to image transformation column
        :return: Dataframe with added prefix in Image_Transformation column
        """
        if self.modality == "MR":
            self.logger.info("Adding normalization as prefix to image transformation column")
            for i, row in df.iterrows():
                if "Image" in df.columns:
                    # Add prefix to normalized image description
                    if "N4BF_normalized" in row["Image"]:
                        if df.loc[i, "Image_Transformation"] != "N4BF_normalized":
                            df.loc[i, "Image_Transformation"] = "N4BF_normalized_" + df.loc[i, "Image_Transformation"]
                    elif "z_score_normalized" in row["Image"]:
                        if df.loc[i, "Image_Transformation"] != "Z-score_normalized":
                            df.loc[i, "Image_Transformation"] = "Z-score_normalized_" + df.loc[
                                i, "Image_Transformation"]
                else:
                    self.error.error("No Image column found in csv file!")
        return df

    @staticmethod
    def extract_filename(path):
        """
        Extracts the mask filename pattern from a path
        :param path: path to file in .nii.gz format
        :return: filename
        """
        if isinstance(path, str):  # Single String
            return path.split('/')[-1].replace('.nii.gz', '')
        elif isinstance(path, pd.Series):  # Pandas Series
            return path.str.split('/').str[-1].str.replace('.nii.gz', '')
        else:
            raise ValueError("Unsupported type. Path must be a str or pd.Series.")
    
    @staticmethod
    def extract_image_filename(path):
        """
        Extracts the image filename pattern from a path without the configuration
        :param path: path to file in .nii.gz format
        :return: filename
        """
        pattern = r'_nan_roiL_\d+_roiT_\d+_roiN_\d+_' # pattern to remove

        if isinstance(path, str):  # Single string case
            filename = path.split('/')[-1]
            filename = re.sub(pattern, '', filename)  # Remove dynamic pattern
            filename = filename.replace('.nii.gz', '')  # Remove file extension
            return filename
        elif isinstance(path, pd.Series):  # Pandas Series case
            return (path.str.split('/').str[-1]
                    .str.replace(pattern, '', regex=True)  # Remove pattern dynamically
                    .str.replace('.nii.gz', '', regex=True))  # Remove file extension
        else:
            raise ValueError("Unsupported type. Path must be a str or pd.Series.")
        

    def process_mirp(self):
        """
        Process MIRP non-processed experiment and complete them from subject folder or output summary file
        :return: pandas df with samples of all processed experiments
        """

        missing_experiments_4_exe = []
        all_experiments_id = []

        # check if individual optimization of bin size is necessary
        # print("Max bin count:", self.data_fingerprint['Number_of_bins'].max())
        # check if extreme values are included may cause Mem problems!
        if self.data_fingerprint['Number_of_bins'].max() > self.rptk_config["Feature_extraction_config"]["max_num_bin"]:
            # self.self_optimize = True
            self.get_bin_borders()
            if not self.self_optimize:
                self.error.warning("Bin count of samples may cause memory leaks: " + str(
                    self.data_fingerprint['Number_of_bins'].max()) + "! Recommend to configure discretization!")
                print("Warning: Bin count of samples may cause memory leaks: " + str(
                    self.data_fingerprint['Number_of_bins'].max()) + "! Recommend to configure discretization!")

        # Get experiment objects
        self.experiments = self.generate_mirp_experiments()

        # search for existing experiments in subject folder
        os.makedirs(self.subject_dir + "/", exist_ok=True)
        for exp in self.experiments:
            all_experiments_id.append(exp.subject)

        Missing_detector = MissingExperimentDetector(extractor=self.extractor,  # Either PyRadiomics or MIRP
                                                     out_path=self.extracted_features_dir,  # Path to output directory
                                                     verbose=self.verbose,  # Verbose output level
                                                     logger=self.logger,  # Logger object
                                                     subject_dir=self.subject_dir,  # Path to subject directory
                                                     all_experiments_id=all_experiments_id,  # List of all experiments
                                                     RunID=self.RunID,
                                                     num_cpus=self.n_cpu,
                                                     error=self.error,
                                                     )
        # Enable automatic garbage collection
        gc.enable()
        missing_experiment_ids, done_experiments, failed = Missing_detector.execute()

        if self.df.empty:
            self.error.error("No configuration file found!")
            print("Error: No configuration file found!")
            self.df = self.get_info_from_csv(path2csv=self.path2confCSV)

        # add all configs from the preprocessing file to the extraction file
        
            
        mirp_preprocess = self.df.copy()
        # generate unique ID for each subject
        mirp_preprocess['id_subject'] = mirp_preprocess.apply(lambda row: f"{Extractor.extract_image_filename(row['Image'])}_{Extractor.extract_filename(row['Mask'])}", axis=1)
        
        if "ID" not in mirp_preprocess.columns:
            mirp_preprocess['ID'] = mirp_preprocess['id_subject'].str.split('_').str[0]

        mirp_preprocess = mirp_preprocess.set_index('id_subject')
        
        if len(done_experiments) > 0:       
            if "id_subject" in done_experiments.columns:
                done_experiments = done_experiments.set_index('id_subject')
            elif done_experiments.index.name != "id_subject":
                self.error.error("Index name is not id_subject!")
                print("Index name is not id_subject!")

            # done_experiments = pd.concat([mirp_preprocess, done_experiments], axis=1)
            
            if not os.path.exists(self.outfile):
                done_experiments.to_csv(self.outfile)


                done_experiments_with_config = self.add_config_cols_to_mirp(result=done_experiments.copy())

            if not os.path.exists(self.outfile):
                done_experiments_with_config.to_csv(self.outfile)
                self.logger.info("Saved collected done experiments to " + str(self.outfile))

        
        if len(missing_experiment_ids) > 0:
            
            for id in tqdm(missing_experiment_ids, desc="Setting up MIRP experiments", unit="sample"):
                for experiment in self.experiments:
                    if id == experiment.subject:
                        missing_experiments_4_exe.append(experiment)
                        break
            
            if len(missing_experiments_4_exe) == 0:
                self.error.error("Format error in Image/Seg config file! Delete the output from previous runs!")

            print("Starting MIRP ...")

            gc.collect()  # Force garbage collection

            result = self.extract_features(func=self.execute_mirp_experiment,
                                           entries=missing_experiments_4_exe,
                                           extractor="MIRP",
                                           returning=False)

            del missing_experiments_4_exe
            del done_experiments
            done_experiments = pd.DataFrame()
            # checking (tmp) results in subject_dir
            csv_files = glob.glob(self.subject_dir + "/*.csv")
            print("Found {} csv files in tmp folder".format(len(csv_files)))

            self.logger.info("Gathering results!")
            if len(done_experiments) == 0:
                print("No out file found in output folder!")

                done_experiments = self.concat_extraction(csv_files=csv_files, ignore_index=True)
                # done_experiments = pd.concat([done_experiments, pd.read_csv(csv_file, sep=";")])
            else:
                df = self.concat_extraction(csv_files=csv_files, ignore_index=True)
                done_experiments = pd.concat([done_experiments, df], ignore_index=True)
                del df

            if done_experiments.index.name == 'id_subject':
                Missing_detector = MissingExperimentDetector(extractor=self.extractor,  # Either PyRadiomics or MIRP
                                                             out_path=self.extracted_features_dir,
                                                             # Path to output directory
                                                             verbose=self.verbose,  # Verbose output level
                                                             logger=self.logger,  # Logger object
                                                             subject_dir=self.subject_dir,  # Path to subject directory
                                                             all_experiments_id=done_experiments.index.tolist(),
                                                             RunID=self.RunID,
                                                             num_cpus=self.n_cpu,
                                                             error=self.error,
                                                             )

                new_exp, new_samples, failed = Missing_detector.get_done_experiments_from_subject_folder(
                    done_samples=done_experiments.index.tolist(),
                    done_experiments=done_experiments)
            else:
                Missing_detector = MissingExperimentDetector(extractor=self.extractor,  # Either PyRadiomics or MIRP
                                                             out_path=self.extracted_features_dir,
                                                             # Path to output directory
                                                             verbose=self.verbose,  # Verbose output level
                                                             logger=self.logger,  # Logger object
                                                             subject_dir=self.subject_dir,  # Path to subject directory
                                                             all_experiments_id=done_experiments["id_subject"].tolist(),
                                                             RunID=self.RunID,
                                                             num_cpus=self.n_cpu,
                                                             error=self.error,
                                                             )

                new_exp, new_samples, failed = Missing_detector.get_done_experiments_from_subject_folder(
                    done_samples=done_experiments["id_subject"].tolist(),
                    done_experiments=done_experiments)

            done_experiments = pd.concat([done_experiments, new_exp])

        else:
            self.logger.info("All experiments are complete!")
            print("All experiments are complete!")

        # print("ID in done_experiments.columns", "ID" in done_experiments.columns)

        return done_experiments, Missing_detector

    def add_config_cols_to_mirp(self, result: pd.DataFrame):
        """
        Add configuration columns to the output file of MIRP
        """

        columns_should_have_nan = ["Image_Transformation", "Mask_Transformation"]

        #  1. add configuration parameters to the output file
        if self.df.empty:
            self.error.error("No configuration file found!")
            print("Error: No configuration file found!")
            self.df = self.get_info_from_csv(path2csv=self.path2confCSV)

        if len(result) == 0:
            self.error.error("Feature extraction failed!")
            raise ValueError("Feature extraction failed!")

        if "id_subject" not in result.columns:
            
            if ("Image" not in result.columns) or ("Mask" not in result.columns):
                if self.df.index.name != result.index.name:
                    if result.index.name in self.df.columns:
                        print("Adapter index name to config file!")
                        result = result.set_index(self.df.index.name, drop=False)
                    elif self.df.index.name in result.columns:
                        print("Adapter index name to config file!")
                        result = result.set_index(self.df.index.name, drop=False)
                    else:
                        if "id_subject" in result.columns:
                            result = result.set_index("id_subject", drop=False)
                        elif "id_subject" != result.index.name:
                            print("Missing id_subject in extraction file!")
                            self.error.error("Missing id_subject in extraction file!")
                            raise ValueError("Missing id_subject in extraction file!")

                        if "id_subject" != self.df.index.name:
                            if "id_subject" in self.df.columns:
                                self.df = self.df.set_index("id_subject", drop=False)
                            else:
                                if ("Image" in self.df.columns) and ("Mask" in self.df.columns):
                                    self.df['id_subject'] = self.df.apply(lambda row: f"{Extractor.extract_image_filename(row['Image'])}_{Extractor.extract_filename(row['Mask'])}", axis=1)
                                    self.df = self.df.set_index("id_subject", drop=False)
                                else:
                                    self.error.error("No Image and Mask column found in preprocessing csv file!")
                                    raise ValueError("No Image and Mask column found in preprocessing csv file!")

            result = pd.concat([result, self.df.copy()[self.df.index.isin(result.index)]], axis=1)
        
            if ("Image" in result.columns) and ("Mask" in result.columns):
                # ID generated from Image and Mask file name
                result['id_subject'] = result.apply(lambda row: f"{Extractor.extract_image_filename(row['Image'])}_{Extractor.extract_filename(row['Mask'])}", axis=1)

            else:
                self.error.error("No Image and Mask column found in csv file!")
                raise ValueError("No Image and Mask column found in csv file!")

        # result.to_csv(self.out_path + "error_extraction.csv")

        if result["id_subject"].isnull().values.any():
            self.error.error("Column id_subject has NaN for sample {}!".format(str(result["img_data_roi"].values)))
            raise ValueError("Column id_subject has NaN for sample {}!".format(str(result["img_data_roi"].values)))

        # synchronize index
        if isinstance(result["id_subject"].iloc[0], float):
            print("Data in wrong format:", result["id_subject"].iloc[0])
            self.error.error("Data in wrong format:", result["id_subject"].iloc[0])

        if "cropped" in result["id_subject"].iloc[0]:
            self.df["id_subject"] = self.df["Image"]
            self.df["id_subject"] = self.df["id_subject"].apply(os.path.basename)
            self.df = self.df.set_index("id_subject", drop=False)
            idx = self.df.index.str[:-len(".nii.gz")]
            self.df.index = idx
            # result = result.set_index("id_subject", drop=False)
            # result.index.str[:-len(".nii.gz")]

        if "id_subject" in result.columns:
            result.index = result["id_subject"]
        elif "img_data_roi" in result.columns:
            result.index = result["img_data_roi"]
        else:
            self.error.error("Could not find either id_subject nor img_data_roi for index in output file!")
            print("Error: Could not find either id_subject nor img_data_roi for index in output file!")

        # TODO: Problem: syncronization of index between these files
        # result.to_csv(self.out_path + "df_before_error_extraction.csv")
        preprocessing_index = self.df.index.name
        if self.df.index.name != result.index.name:
            if result.index.name in self.df.columns:
                print("Adapter index name to config file!")
                self.df = self.df.set_index(result.index.name, drop=False)

        if result.index[0] in self.df.index:
            # check for duplicates
            if result.duplicated().sum() > 0:
                self.logger.info("Detected " + str(result.duplicated().sum()) + " duplicates in results!")
                result = result[~result.duplicated(keep='first')]
            if result.index.duplicated().sum() > 0:
                self.logger.info("Detected " + str(result.index.duplicated().sum()) + " duplicates in results index!")
                result = result[~result.index.duplicated(keep='first')]

            if self.df.duplicated().sum() > 0:
                self.logger.info("Detected " + str(self.df.duplicated().sum()) + " duplicates in config file!")
                self.df = self.df[~self.df.duplicated(keep='first')]

            if self.df.index.duplicated().sum() > 0:
                self.logger.info(
                    "Detected " + str(self.df.indexduplicated().sum()) + " duplicates in config file index!")
                self.df = self.df[~self.df.index.duplicated(keep='first')]

            # Only include config data if it is not there already
            for col in self.df.columns.to_list():
                if col not in result.columns.to_list():
                    if "Unnamed" not in col:
                        result[col] = self.df[col]
            try:
                # expecting columns to contain nan values --> make 0 later to nan
                for col in columns_should_have_nan:
                    result[col] = result[col].fillna(0)
            except:
                # result.to_csv(self.out_path + "error_extraction.csv")
                self.error.warning("Image_Transformation or Mask_Transformation are not in data!")
                result["Image_Transformation"] = 0
                result["Mask_Transformation"] = 0
                # raise ValueError("Image_Transformation or Mask_Transformation are not in data!")

            self.logger.info("Checking for features only containing NaN ")

            # get all columns only containing nan
            nan_features = result.columns[result.isnull().all(0)]
            if len(nan_features) > 0:
                self.logger.info("There are " + str(len(nan_features)) + " features only containing NaN values!")
                for col in nan_features:
                    self.logger.info("Only NaN values in column: " + str(col))
                    result.drop([col], axis=1, inplace=True)

            # get all columns containing Unnamed in the name
            unnamed_features = rptk.get_unnamed_cols(result)
            if len(unnamed_features) > 0:
                self.logger.info(
                    "There are " + str(len(unnamed_features)) + " features containing Unnamed in the name!")
                for col in unnamed_features:
                    self.logger.info("Unnamed feature: " + str(col))
                    result.drop([col], axis=1, inplace=True)

            # get all columns containing any nan
            nan_result = result[result.columns[result.isnull().any()]]

            if len(nan_result) > 0:
                self.error.warning("There are " + str(result.isnull().sum().sum()) + " NaN values in the outfile!")
                if len(nan_result.index) == len(result.index):
                    self.error.warning("All rows contain NaN values!")
                    for col in nan_result.columns:
                        self.error.warning("Extraction of " + str(
                            nan_result[col].isna().sum()) + " samples failed for feature: " + str(col))

                if len(nan_result.columns) == len(result.columns):
                    self.error.warning("All columns contain NaN values!")

                # for i in nan_result.index:
                # self.error.error("Nan values in row: " + str(i))

                # self.logger.info("Drop all nan values!")
                # drop all experiments (rows) which are having nan values (not done yet)
                # result = result.dropna()
        else:
            print(f"Error: No matching index found between configuration file and mirp extraction! {result.index[0]} not in Preprocessing!")
            self.error.error("No matching index found between configuration file and mirp extraction! {result.index[0]} not in {self.df.index}!")
            self.error.error("Config file index: " + str(self.df.index[0]))
            self.error.error("MIRP extraction index: " + str(result.index[0]))
            raise ValueError("No matching index found between configuration file and mirp extraction! {result.index[0]} not in Preprocessing !")
        
        if  self.df.index.name != result.index.name:
            preprocessing_index = self.df.index.name
            if preprocessing_index in result.columns:
                result = result.set_index(preprocessing_index, drop=False)

        # put NAN back to columns
        for col in columns_should_have_nan:
            result[col] = result[col].replace(0, np.nan)
        
        return result

    def generating_mirp_output_file(self):

        

        # generating output files and checking for
        result, missing_detector = self.process_mirp()
        result = self.add_config_cols_to_mirp(result)

        return result, missing_detector

    def exe_mirp(self):
        """
        Executes the MIRP pipeline.
        """
        print("Configure MIPR Extraction ...")
        
        self.logger.info("Configuration of MIRP Feature Extraction:" +
                         "\n\t\t\t\t\t\t\t\t\t\tInput: " + self.path2confCSV +
                         "\n\t\t\t\t\t\t\t\t\t\tOutput directory: " + self.extracted_features_dir +
                         "\n\t\t\t\t\t\t\t\t\t\tTmp output directory: " + self.subject_dir +
                         "\n\t\t\t\t\t\t\t\t\t\tOutput file: " + self.outfile +
                         "\n\t\t\t\t\t\t\t\t\t\tRunID: " + self.RunID +
                         "\n\t\t\t\t\t\t\t\t\t\tNumber of CPU: " + str(self.n_cpu) +
                         "\n\t\t\t\t\t\t\t\t\t\tConfig: " + self.extraction_yaml
                         )

        # Get Extractor configuration
        self.extractor_setting = self.get_extractor_config() 

        # self.outfile = self.out_path + "mirp_extraction.csv"
        # result = pd.DataFrame()
        print("MIRP output file:", self.outfile)
        # if there is no outfile
        if not os.path.exists(self.outfile):
            result, missing_detector = self.generating_mirp_output_file()

        # if there is an outfile from previous run but it is empty
        elif os.path.getsize(self.outfile) == 0:
            self.logger.info("Redo MIRP feature extraction as it failed previously!")
            result, missing_detector = self.generating_mirp_output_file()
        # if there is an outfile from previous run and it is not empty
        else:
            result, missing_detector = self.process_mirp()

        # drop duplicated rows
        result.drop_duplicates(inplace=True)
        # drop duplicated columns with same name
        result = result.loc[:, ~result.columns.duplicated()]
        # drop duplicated columns with same values
        # result = result.T.drop_duplicates().T # drops too many features
        # print("ID in result.columns", "ID" in result.columns)
        if (len(missing_detector.missing_samples) == 0) and (missing_detector.found_output_file_name == ""):
            self.logger.info("Could not find any MIRP extraction file")
            self.logger.info("Write MIRP extraction to outfile: " + os.path.basename(self.outfile) + " !")
            result.to_csv(self.outfile)

        if missing_detector.found_output_file_name != "":
            self.logger.info("MIRP extraction file already exists: " + missing_detector.found_output_file_name + " !")

        if len(missing_detector.missing_samples) > 0:
            self.error.warning(
                "MIRP extraction failed for " + str(len(missing_detector.missing_samples)) + " samples!")
            self.error.warning("Missing samples: " + str(missing_detector.missing_samples))

        if "Image_Transformation" not in self.df.columns:
            self.df["Image_Transformation"] = np.nan

        if "Mask_Transformation" not in self.df.columns:
            self.df["Mask_Transformation"] = np.nan

        if "ID" not in result.columns:
            if "id_subject" in result.columns:
                result['ID'] = result['id_subject'].str.split('_').str[0]
            elif result.index.name == "id_subject":
                result['ID'] = result.index.str.split('_').str[0]
            elif "Image" in result.columns:
                result['ID'] = result["Image"].apply(lambda x: x.split('/')[-1].split('_').str[0])
            else:
                self.error.error("No id_subject column found in result file!")
                raise ValueError("No id_subject column found in result file!")

        # check for completeness - get problems to concat the data configuration from preprocessing into the feature space
        if len(result.loc[result["ID"].isna()]) > 0:
            # complete data with preprocessing data
            if "id_subject" in result.columns:
                for i, row in tqdm(result.loc[result["ID"].isna()].iterrows(),
                                   total=len(result.loc[result["ID"].isna()]), desc='MIRP Config Checking'):
                    Image_Transformation = False

                    # first check image Transformation
                    for image_path in self.df.loc[~self.df["Image_Transformation"].isna(), "Image"]:
                        image = os.path.basename(image_path)[:-len(".nii.gz")]

                        # find the correct data
                        if image in row["id_subject"]:
                            Image_Transformation = True
                            SampleIO = self.df.loc[self.df["Image"] == image_path]

                            for mask_path in SampleIO["Mask"]:
                                mask = os.path.basename(mask_path)[:-len(".nii.gz")]

                                if mask == row["img_data_roi"]:
                                    result.loc[i, "ID"] = SampleIO.loc[SampleIO["Mask"] == mask_path, "ID"].values[0]
                                    result.loc[i, "Image"] = \
                                        SampleIO.loc[SampleIO["Mask"] == mask_path, "Image"].values[0]
                                    result.loc[i, "Mask"] = SampleIO.loc[SampleIO["Mask"] == mask_path, "Mask"].values[
                                        0]
                                    result.loc[i, "Modality"] = \
                                        SampleIO.loc[SampleIO["Mask"] == mask_path, "Modality"].values[0]
                                    result.loc[i, "Prediction_Label"] = \
                                        SampleIO.loc[SampleIO["Mask"] == mask_path, "Prediction_Label"].values[0]
                                    result.loc[i, "Mask_Transformation"] = \
                                        SampleIO.loc[SampleIO["Mask"] == mask_path, "Mask_Transformation"].values[0]
                                    result.loc[i, "Image_Transformation"] = \
                                        SampleIO.loc[SampleIO["Mask"] == mask_path, "Image_Transformation"].values[0]

                                    if SampleIO.loc[SampleIO["Mask"] == mask_path, "ROI_Label"].values[0] == "nan":
                                        result.df.loc[i, "ROI_Label"] = 1.0
                                    else:
                                        result.loc[i, "ROI_Label"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "ROI_Label"].values[0]
                                    break
                            break

                    if not Image_Transformation:
                        for image_path in self.df.loc[self.df["Image_Transformation"].isna(), "Image"]:
                            image = os.path.basename(image_path)[:-len(".nii.gz")]

                            if image in row["id_subject"]:
                                SampleIO = self.df.loc[self.df["Image"] == image_path]

                                for mask_path in SampleIO["Mask"]:
                                    mask = os.path.basename(mask_path)[:-len(".nii.gz")]

                                    if mask == row["img_data_roi"]:
                                        result.loc[i, "ID"] = SampleIO.loc[SampleIO["Mask"] == mask_path, "ID"].values[
                                            0]
                                        result.loc[i, "Image"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "Image"].values[0]
                                        result.loc[i, "Mask"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "Mask"].values[
                                                0]
                                        result.loc[i, "Modality"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "Modality"].values[0]
                                        result.loc[i, "Prediction_Label"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "Prediction_Label"].values[0]
                                        result.loc[i, "Mask_Transformation"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "Mask_Transformation"].values[0]
                                        result.loc[i, "Image_Transformation"] = \
                                            SampleIO.loc[SampleIO["Mask"] == mask_path, "Image_Transformation"].values[
                                                0]

                                        if SampleIO.loc[SampleIO["Mask"] == mask_path, "ROI_Label"].values[0] == "nan":
                                            result.loc[i, "ROI_Label"] = 1.0
                                        else:
                                            result.loc[i, "ROI_Label"] = \
                                                SampleIO.loc[SampleIO["Mask"] == mask_path, "ROI_Label"].values[0]
                                        break
                                break

        self.logger.info("MIRP extraction finished")
        print("MIRP extraction finished")

        return result

    def try_pyradiomics_extraction(self, experiment, py_extractor):
        """ 
        Try to extract features with PyRadiomics
        :param experiment: experiment object
        :param py_extractor: PyRadiomics extractor object
        :param exp: experiment object
        :return: True if extraction was successful, False
        """


        py_formated = pd.DataFrame()
        try:
            self.logger.info("Extracting Features from " + str(experiment["Image"]) + " and " + str(experiment["Mask"]))

            #if experiment["ROI_Label"] != 1:
            #    py_result = py_extractor.execute(imageFilepath=str(experiment["Image"]), maskFilepath=str(experiment["Mask"]), label=int(experiment["ROI_Label"]))
            #else:
            py_result = py_extractor.execute(imageFilepath=str(experiment["Image"]), maskFilepath=str(experiment["Mask"]))

            py_formated = pd.DataFrame([py_result])

        except Exception as exc:
            return False, exc
        
        return True, py_formated

    def extract_pyradiomics(self, py_extractor, experiment, exp):

        py_formated = pd.DataFrame()
        py_result = pd.DataFrame()
        # self.logger.info("Extract Features from experiment {}".format(str(experiment["ID"])))

        status, result_ = self.try_pyradiomics_extraction(experiment, py_extractor)
        
        if not status:
            if "Image/Mask geometry mismatch." in str(result_):
                self.error.warning("Trying to correct Image/Mask geometry mismatch between {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))
                print("Trying to correct Image/Mask geometry mismatch between {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                py_extractor.settings["correctMask"] = True
                py_extractor.settings["geometryTolerance"] = 1e-4

                # setting correctMask to True for resample the mask to image reference space
                status, result_ = self.try_pyradiomics_extraction(experiment, py_extractor)
                
                if not status:

                    if "Bounding box of ROI is larger than image space." in str(result_):
                        self.error.warning("Trying to correct Bounding box of ROI is larger than image space between {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))
                        print("Trying to correct Bounding box of ROI is larger than image space between {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                        py_extractor.settings["resampledPixelSpacing"] = None
                        py_extractor.settings["interpolator"] = None
                        py_extractor.settings["padDistance"] = None
                        py_extractor.settings["correctMask"] = True

                        status, result_ = self.try_pyradiomics_extraction(experiment, py_extractor)

                        if not status:
                            self.error.error("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                            print("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                            failed = pd.DataFrame({"ID": experiment["ID"], "Image": experiment["Image"], "Mask": experiment["Mask"]}, index=[0])
                            if self.failed_extractions.empty:
                                self.failed_extractions = failed
                            # if Image and Mask are already in the failed extraction in the same row
                            elif self.failed_extractions.loc[(self.failed_extractions["Image"] == experiment["Image"]) & (self.failed_extractions["Mask"] == experiment["Mask"])].empty:
                                self.failed_extractions = pd.concat([self.failed_extractions, failed], ignore_index=True)
                        else:
                            py_formated = result_

                    elif "Resegmentation excluded too many voxels" in str(result_):
                        self.error.warning("Trying to correct Resegmentation excluded too many voxels between {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))
                        print("Trying to correct Resegmentation excluded too many voxels between {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                        py_extractor.settings["resegmentShape"] = False
                        py_extractor.settings["resegmentRange"] = None
                        py_extractor.settings["correctMask"] = False
                        

                        status, result_ = self.try_pyradiomics_extraction(experiment, py_extractor)

                        if not status:
                            self.error.error("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                            print("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                            failed = pd.DataFrame({"ID": experiment["ID"], "Image": experiment["Image"], "Mask": experiment["Mask"]}, index=[0])
                            if self.failed_extractions.empty:
                                self.failed_extractions = failed
                            # if Image and Mask are already in the failed extraction in the same row
                            elif self.failed_extractions.loc[(self.failed_extractions["Image"] == experiment["Image"]) & (self.failed_extractions["Mask"] == experiment["Mask"])].empty:
                                self.failed_extractions = pd.concat([self.failed_extractions, failed], ignore_index=True)
                        else:
                            py_formated = result_



                    else:
                        self.error.error("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                    str(experiment["Mask"]),
                                                                                    str(result_)))

                        print("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                        str(experiment["Mask"]),
                                                                                        str(result_)))

                        failed = pd.DataFrame({"ID": experiment["ID"], "Image": experiment["Image"], "Mask": experiment["Mask"]}, index=[0])
                        if self.failed_extractions.empty:
                            self.failed_extractions = failed
                        # if Image and Mask are already in the failed extraction in the same row
                        elif self.failed_extractions.loc[(self.failed_extractions["Image"] == experiment["Image"]) & (self.failed_extractions["Mask"] == experiment["Mask"])].empty:
                            self.failed_extractions = pd.concat([self.failed_extractions, failed], ignore_index=True)

                        py_result = pd.DataFrame()
                else:
                    py_formated = result_

            elif "Bounding box of ROI is larger than image space." in str(result_):
                self.error.warning("Trying to correct Bounding box of ROI is larger than image space between {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))
                print("Trying to correct Bounding box of ROI is larger than image space between {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                py_extractor.settings["resampledPixelSpacing"] = None
                py_extractor.settings["interpolator"] = None
                py_extractor.settings["padDistance"] = None
                py_extractor.settings["correctMask"] = True

                status, result_ = self.try_pyradiomics_extraction(experiment, py_extractor)

                if not status:
                    self.error.error("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                    print("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                    failed = pd.DataFrame({"ID": experiment["ID"], "Image": experiment["Image"], "Mask": experiment["Mask"]}, index=[0])
                    if self.failed_extractions.empty:
                        self.failed_extractions = failed
                    # if Image and Mask are already in the failed extraction in the same row
                    elif self.failed_extractions.loc[(self.failed_extractions["Image"] == experiment["Image"]) & (self.failed_extractions["Mask"] == experiment["Mask"])].empty:
                        self.failed_extractions = pd.concat([self.failed_extractions, failed], ignore_index=True)

            elif "Resegmentation excluded too many voxels" in str(result_):
                self.error.warning("Trying to correct Resegmentation excluded too many voxels between {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))
                print("Trying to correct Resegmentation excluded too many voxels between {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                py_extractor.settings["resegmentShape"] = False
                py_extractor.settings["resegmentRange"] = None
                py_extractor.settings["correctMask"] = False
                
                
                status, result_ = self.try_pyradiomics_extraction(experiment, py_extractor)

                if not status:
                    self.error.error("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                    print("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                    failed = pd.DataFrame({"ID": experiment["ID"], "Image": experiment["Image"], "Mask": experiment["Mask"]}, index=[0])
                    if self.failed_extractions.empty:
                        self.failed_extractions = failed
                    # if Image and Mask are already in the failed extraction in the same row
                    elif self.failed_extractions.loc[(self.failed_extractions["Image"] == experiment["Image"]) & (self.failed_extractions["Mask"] == experiment["Mask"])].empty:
                        self.failed_extractions = pd.concat([self.failed_extractions, failed], ignore_index=True)
                else:
                    py_formated = result_
            
            else:
                self.error.error("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                            str(experiment["Mask"]),
                                                                            str(result_)))

                print("Failed extraction from {} and {}: {}".format(str(experiment["Image"]),
                                                                                str(experiment["Mask"]),
                                                                                str(result_)))

                failed = pd.DataFrame({"ID": experiment["ID"], "Image": experiment["Image"], "Mask": experiment["Mask"]}, index=[0])
                if self.failed_extractions.empty:
                    self.failed_extractions = failed
                # if Image and Mask are already in the failed extraction in the same row
                elif self.failed_extractions.loc[(self.failed_extractions["Image"] == experiment["Image"]) & (self.failed_extractions["Mask"] == experiment["Mask"])].empty:
                    self.failed_extractions = pd.concat([self.failed_extractions, failed], ignore_index=True)

                py_result = pd.DataFrame()
        else:
            py_formated = result_

        # Free Memory
        del py_result
        del py_extractor

        py_formated.index = exp.index

        # Add experiment configuration to the features
        result = pd.concat([exp, py_formated], axis=1)

        if py_formated.isnull().values.sum() > 0:
            self.error.warning(
                "PyRadiomics Feature Extraction Failed for Image: " + str(os.path.basename(experiment["Image"])))

        del py_formated

        if "cropped" in experiment["Image"] or "cropped" in experiment["Mask"]:
            # Save performed extraction - Image name contains already segmentation
            result.to_csv(self.subject_dir + str(self.extractor) + "_extraction_" + str(
                os.path.basename(experiment["Image"][:-len(".nii.gz")])) + ".csv")
        else:
            # Save performed extraction
            result.to_csv(self.subject_dir + str(self.extractor) + "_extraction_" + str(
                os.path.basename(experiment["Image"][:-len(".nii.gz")])) + "_" + str(
                os.path.basename(experiment["Mask"][:-len(".nii.gz")])) + ".csv")

        if result.isnull().values.sum() > 5:  # Some config Parameters like Timepoint or ROI Labels might be NaN
            self.error.warning(
                "PyRadiomics Feature Concat Failed for Image: " + str(os.path.basename(experiment["Image"])))
            raise ValueError(
                "PyRadiomics Feature Concat Failed for Image: " + str(os.path.basename(experiment["Image"])))

        del result

    def execute_pyradiomics_experiment(self, experiment: pd.Series, extractor_setting=None, py_extractor=None):
        """
        Execute RyRadiomics Experiments
        :param experiment: (pd.Series) including Image as a path to Image and Mask as a path to Mask
        :return: extracted radiomics features with data data from config
        """

        exp = pd.DataFrame([experiment])

        if py_extractor is None:
            if extractor_setting is None:
                self.error.warning("No Settings defined for the PyRadiomics Extractor!")
            else:
                ### Config Extractor
                py_extractor = featureextractor.RadiomicsFeatureExtractor(extractor_setting)

        # configure PyRadiomics Feature Extractor:
        if self.self_optimize:

            if ("Raw_Image" in self.df.columns) and ("Raw_Mask" in self.df.columns):
                number_of_bins = self.data_fingerprint.loc[
                    (self.data_fingerprint["Image"] == str(experiment["Raw_Image"])) & (
                                self.data_fingerprint["Mask"] == str(experiment["Raw_Mask"])), "Number_of_bins"].values
            else:
                number_of_bins = self.data_fingerprint.loc[
                    (self.data_fingerprint["Image"] == str(experiment["Image"])) & (
                                self.data_fingerprint["Mask"] == str(experiment["Mask"])), "Number_of_bins"].values

            # print("Number of Bins {} with bin width of {}.".format(str(number_of_bins), str(self.extractor_setting["setting"]["binWidth"])))
            # print(self.data_fingerprint.loc[(self.data_fingerprint["Image"] == str(experiment["Image"])) & (self.data_fingerprint["Mask"] == str(experiment["Mask"])), "Number_of_bins"].values)

            if not hasattr(self, 'min_bin_number'):
                self.get_bin_borders()

            # get the number of bins for the sample
            if number_of_bins < self.min_bin_number:
                self.logger.info(
                    "Need to optimize settings for " + str(os.path.basename(experiment["Image"])) + " to " +
                    str(self.min_bin_number) + " as " + str(number_of_bins) + " is very small!")

                # self.extractor_setting["setting"].pop("binWidth", None)
                # self.extractor_setting["setting"]["binCount"] = self.min_bin_number

                py_extractor.settings["binCount"] = self.min_bin_number
                py_extractor.settings["binWidth"] = None

            if number_of_bins > self.max_bin_number:
                self.logger.info(
                    "Need to optimize settings for " + str(os.path.basename(experiment["Image"])) + " to " +
                    str(self.max_bin_number) + " as " + str(number_of_bins) + " is very large!")

                # self.extractor_setting["setting"].pop("binWidth", None)
                # self.extractor_setting["setting"]["binCount"] = self.max_bin_number

                py_extractor.settings["binCount"] = self.max_bin_number
                py_extractor.settings["binWidth"] = None

        if not self.resegmentation:
            self.logger.info("Do not perform Resegmentation for PyRadiomics extraction. Reset resegmentRange and resegmentShape!")
            py_extractor.settings["resegmentRange"] = None
            py_extractor.settings["resegmentShape"] = False

        try:
            # extract pyradiomics features and write it to folder
            self.extract_pyradiomics(py_extractor=py_extractor,
                                     experiment=experiment.copy(),
                                     exp=exp.copy())

        except Exception as ex:
            print("Failed experiment. Try without resegmentation ...")
            self.error.warning("Failed experiment. Try without resegmentation ... ")
            self.error.warning(ex.__class__.__name__, str(ex), traceback.format_exc())
            
            # Resegmentation might not be feasable for this sample - retry without resegmentation
            print(ex.__class__.__name__, str(ex), traceback.format_exc())
            print("Trying to repeat extraction with changed settings.")
            
            if extractor_setting is not None:
                if "resegmentShape" in extractor_setting["setting"]:

                    if extractor_setting["setting"]["resegmentShape"]:
                        py_extractor.settings["resegmentShape"] = False
            else:
                py_extractor.settings["resegmentShape"] = False

            try:
                self.extract_pyradiomics(py_extractor=py_extractor,
                                        experiment=experiment.copy(),
                                        exp=exp.copy())
            except Exception as ex:
                self.error.error(ex.__class__.__name__, str(ex))
                raise Exception(ex.__class__.__name__, str(ex))


            if "ID" in exp.columns:
                self.error.warning("Resegmentation disabled for experiment " + str(experiment["ID"]) + " " +
                                 str(os.path.basename(experiment["Image"])) + " " +
                                 str(os.path.basename(experiment["Mask"])) + ": " + str(ex))
                self.error.warning(traceback.format_exc())
                #raise TypeError("Error processing experiment " + str(experiment["ID"]) + " " +
                #                str(os.path.basename(experiment["Image"])) + " " +
                #                str(os.path.basename(experiment["Mask"])) + ": " + str(ex))

            else:
                self.error.warning("Resegmentation disabled for extracting features from image " + str(experiment["Image"]) + ": " + str(ex))
                self.error.warning(traceback.format_exc())
                # raise TypeError("Error extracting features from image " + str(experiment["Image"]) + ": " + str(ex))

        del exp
        del py_extractor

    def extract_features(self, func, entries=None, extractor=None, returning=False):
        """
        High efficient Pyradiomics feature extraction
        :param entries: Samples that need to get processed
        :param extractor: Name of the extractor MIRP or PyRadiomics
        :param func: Function to call for feature extraction of the sample
        :param returning: Parameter for returning any parameter
        :return:
        """
        if entries is None:
            self.error.error("Wrong function call for executing PyRadiomics feature extraction!")

        # self.mem_consumption = pd.DataFrame({"Iteration":[], "Memory[MB]":[]})
        mem_consumption_sum_file = self.extracted_features_dir + "Memory_usage_profile_" + str(self.RunID) + ".csv"

        # Refresh Memory usage profile
        if os.path.exists(mem_consumption_sum_file):
            os.remove(mem_consumption_sum_file)

        # MIRP
        if isinstance(entries, list):
            result = []
            try:
                with tqdm(total=len(entries), desc=extractor + ' Feature Extraction', unit="sample") as pbar:
                    iter = 0
                    if self.fast_mode:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                            for results in executor.map(func, [row for row in entries], chunksize=self.n_cpu):
                                pbar.update(1)
                                iter += 1
                                mem_tmp = pd.DataFrame({"Iteration": [str(iter)], "Memory[MB]": [
                                    str(psutil.Process().memory_info().rss / 1e6)]},
                                                       index=[str(iter)])
                                if not os.path.isfile(mem_consumption_sum_file):
                                    mem_tmp.to_csv(mem_consumption_sum_file)
                                else:  # else it exists so append without writing the header
                                    with open(mem_consumption_sum_file, 'a') as f:
                                        mem_tmp.to_csv(f, mode='a', header=False)

                                if returning:
                                    result.append(results)
                                gc.collect()  # Force garbage collection

                    else:
                        for entrie in entries:
                            func(entrie)
                            
                            pbar.update(1)
                            iter += 1
                            mem_tmp = pd.DataFrame({"Iteration": [str(iter)], "Memory[MB]": [str(psutil.Process().memory_info().rss / 1e6)]})

                        #chunk_size = self.n_cpu  # Adjust the chunk size based on your memory constraints

                        #for start in range(0, len(entries), chunk_size):
                        #    chunk = entries[start:start + chunk_size]
                        #    with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                        #        futures = {executor.submit(func, row): row for row in chunk}
                                
                        #        for future in concurrent.futures.as_completed(futures):
                        #            mem_tmp = pd.DataFrame({"Iteration": [str(iter)],
                        #                                    "Memory[MB]": [
                        #                                        str(psutil.Process().memory_info().rss / 1e6)]},
                        #                                   index=[str(iter)])
                        #            try:
                        #                result_ = future.result()
                        #                if result_ is not None:
                        #                    if returning:
                        #                        result.append(result_)
                        #                if not returning:
                        #                    del result_

                        #            except Exception as exc:
                        #                self.error.error(f"Generated an exception: {exc}")

                        #            pbar.update(1)
                        #            iter += 1

                        #            if not os.path.isfile(mem_consumption_sum_file):
                        #                mem_tmp.to_csv(mem_consumption_sum_file)
                        #            else:
                        #                with open(mem_consumption_sum_file, 'a') as f:
                        #                    mem_tmp.to_csv(f, mode='a', header=False)

                        #        if not returning:
                        #            del futures
                                    

                    # Free memory after chunk
                    # del chunk
                    # del futures
                    gc.collect()  # Force garbage collection


            except MemoryError as ex:
                self.error.error(
                    extractor + " Feature Extraction Failed based on Memory overload! To prevent this please crop your samples (set cropping to true in the RPTK config json) or disable fast_mode" + str(
                        ex))
                self.error.error(traceback.format_exc())
                raise MemoryError(
                    "RPTK needs more memory to execute or reduce the number of CPUs to use. Maybe try to run it on a bigger machine. " + str(
                        ex))

            except Exception as ex:
                self.error.error(extractor + " Feature Extraction Failed! " + str(ex))
                self.error.error(traceback.format_exc())
                raise Exception("Extraction Failed! " + str(ex))

        # PyRadiomics
        if isinstance(entries, pd.DataFrame):
            # check for index configuration
            if "ID" not in entries.columns:
                if "ID" == entries.index.name:
                    entries["ID"] = entries.index

            result = pd.DataFrame()
            try:
                with tqdm(total=len(entries), desc=extractor + ' Feature Extraction', unit="sample") as pbar:
                    iter = 0
                    if self.fast_mode:
                        chunk_size = self.n_cpu
                        for start in range(0, len(entries), chunk_size):
                            chunk = entries.iloc[start:start + chunk_size]
                            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:

                                futures = {executor.submit(func, row): row for i, row in chunk.iterrows()}

                                for future in concurrent.futures.as_completed(futures):
                                    mem_tmp = pd.DataFrame({"Iteration": [str(iter)],
                                                            "Memory[MB]": [
                                                                str(psutil.Process().memory_info().rss / 1e6)]},
                                                           index=[str(iter)])
                                    try:
                                        result_ = future.result()
                                        if result_ is not None:
                                            if returning:
                                                result = pd.concat([result, result_], ignore_index=True)
                                    except Exception as exc:
                                        logging.error(f"Generated an exception: {exc}")

                                    pbar.update(1)
                                    iter += 1

                                    if not os.path.isfile(mem_consumption_sum_file):
                                        mem_tmp.to_csv(mem_consumption_sum_file)
                                    else:
                                        with open(mem_consumption_sum_file, 'a') as f:
                                            mem_tmp.to_csv(f, mode='a', header=False)

                                if not returning:
                                    del futures
                        
                        
                        #with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:

                        #    for results in executor.map(func, [row for i, row in entries.iterrows()],
                        #                                chunksize=self.n_cpu):
                        #        pbar.update(1)
                        #        iter += 1
                        #        mem_tmp = pd.DataFrame({"Iteration": [str(iter)], "Memory[MB]": [
                        #            str(psutil.Process().memory_info().rss / 1e6)]},
                        #                               index=[str(iter)])

                        #    if not os.path.isfile(mem_consumption_sum_file):
                        #        mem_tmp.to_csv(mem_consumption_sum_file)
                        #    else:
                        #        with open(mem_consumption_sum_file, 'a') as f:
                        #            mem_tmp.to_csv(f, mode='a', header=False)

                        #    if returning:
                        #        result = pd.concat([result, results], ignore_index=True)
                        #    else:
                        #        del results

                        #    gc.collect()  # Force garbage collection
                    else:
                        #for entrie in entries:
                        #    func(entrie)
                            
                        #    pbar.update(1)
                        #    iter += 1
                        #    mem_tmp = pd.DataFrame({"Iteration": [str(iter)], "Memory[MB]": [str(psutil.Process().memory_info().rss / 1e6)]})
                        # with tqdm(total=len(entries), desc=extractor + ' Feature Extraction') as pbar:
                        for i, row in entries.iterrows():
                            func(row)
                            pbar.update(1)
                            iter += 1
                            mem_tmp = pd.DataFrame({"Iteration": [str(iter)], "Memory[MB]": [
                                str(psutil.Process().memory_info().rss / 1e6)]},
                                                   index=[str(iter)])

                            if not os.path.isfile(mem_consumption_sum_file):
                                    mem_tmp.to_csv(mem_consumption_sum_file)
                            else:
                                with open(mem_consumption_sum_file, 'a') as f:
                                    mem_tmp.to_csv(f, mode='a', header=False)

                            gc.collect()  # Force garbage collection


                        #with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                        #    for results in executor.map(func, [row for i, row in entries.iterrows()], chunksize=1):
                        #        pbar.update(1)
                        #        iter += 1

                        #        if returning:
                        #            if results is not None:
                        #                result = pd.concat([result, results], ignore_index=True)
                        #        else:
                        #            del results

                        #        mem_tmp = pd.DataFrame(data={"Iteration": [str(iter)], "Memory[MB]": [
                        #                                                        str(psutil.Process().memory_info().rss / 1e6)]},
                        #                                                    index=[str(iter)])

                        #        if not os.path.isfile(mem_consumption_sum_file):
                        #            mem_tmp.to_csv(mem_consumption_sum_file)
                        #        else:
                        #            with open(mem_consumption_sum_file, 'a') as f:
                        #                mem_tmp.to_csv(f, mode='a', header=False)

                        #        gc.collect()  # Force garbage collection

                    # Free memory after chunk
                    # del chunk
                    # del futures
                    # gc.collect()  # Force garbage collection

            except MemoryError as ex:
                self.error.error(
                    extractor + " Feature Extraction Failed based on Memory overload! To prevent this please crop your samples (set cropping to true in the rptk config json)" + str(
                        ex))
                self.error.error(traceback.format_exc())
            except Exception as ex:
                self.error.error(extractor + " Feature Extraction Failed! " + str(ex))
                self.error.error(traceback.format_exc())
                raise

        return result

    def check_input_format(self, df: pd.DataFrame):
        """
        Check format of input df for extraction if there are cropped images from cropped images or double samples
        :param df: Dataframe containing all img and msk paths also to the cropped samples
        :return: Cleaned DatFrame
        """
        if "raw_Image" in df.columns:
            substring = "_cropped_resample"
            df = df[~df['raw_Image'].str.contains(substring)]
        if "raw_Mask" in df.columns:
            substring = "_cropped_resample"
            df = df[~df['raw_Mask'].str.contains(substring)]

        if ("Image" in df.columns) and ("Mask" in df.columns):
            df = df.drop_duplicates(subset=["Image", "Mask"])

        return df

    def exe_pyradiomics(self):
        """
        Executes the pyradiomics feature extraction.
        """

        # only take PyRadiomics errors:
        radiomics.setVerbosity(30)
        # logger = radiomics.logger
        # logger.setLevel(logging.DEBUG)

        if self.df.empty:
            self.df = self.get_info_from_csv(path2csv=self.path2confCSV)

        # Get Extractor configuration
        # self.extractor_setting = self.get_extractor_config()

        print("Extracting Radiomics features with PyRadiomics ...")
        
        #loader = SpinningLoader(text="Config PyRadiomics extraction")
        #loader.start()
        # check if individual optimization of bin size is necessary
        # check if extreme values are included may cause Mem problems!
        if self.data_fingerprint['Number_of_bins'].max() > self.rptk_config["Feature_extraction_config"]["max_num_bin"]:
            # self.self_optimize = True
            self.get_bin_borders()
            if not self.self_optimize:
                self.error.warning("Bin count of samples may cause memory leaks: " + str(
                    self.data_fingerprint['Number_of_bins'].max()) + "! Recommend to configure discretization!")
                print("WARNING: Bin count of samples may cause memory leaks: " + str(
                    self.data_fingerprint['Number_of_bins'].max()) + "! Recommend to configure discretization!")

        self.logger.info("Configuration of PyRadiomics Feature Extraction:" +
                         "\n\t\t\t\t\t\t\t\t\t\tInput: " + self.path2confCSV +
                         "\n\t\t\t\t\t\t\t\t\t\tOutput directory: " + self.extracted_features_dir +
                         "\n\t\t\t\t\t\t\t\t\t\tTmp output directory: " + self.subject_dir +
                         "\n\t\t\t\t\t\t\t\t\t\tTmp output file: " + self.outfile +
                         "\n\t\t\t\t\t\t\t\t\t\tRunID: " + self.RunID +
                         "\n\t\t\t\t\t\t\t\t\t\tNumber of CPU: " + str(self.n_cpu) +
                         "\n\t\t\t\t\t\t\t\t\t\tConfig: " + self.extraction_yaml
                         )

        complete_result = pd.DataFrame()

        # collecting all experiment IDs
        all_experiments_id = []
        for index, row_ in self.df.iterrows():
            all_experiments_id.append(
                os.path.basename(row_["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row_["Mask"])[
                                                                          :-len(".nii.gz")])
        
        #loader.stop()
        
        Missing_detector = MissingExperimentDetector(extractor=self.extractor,  # Either PyRadiomics or MIRP
                                                     out_path=self.extracted_features_dir,  # Path to output directory
                                                     verbose=self.verbose,  # Verbose output level
                                                     logger=self.logger,  # Logger object
                                                     subject_dir=self.subject_dir,  # Path to subject directory
                                                     all_experiments_id=all_experiments_id,  # List of all experiments
                                                     RunID=self.RunID,
                                                     num_cpus=self.n_cpu,
                                                     error=self.error,
                                                     found_output_file_name=self.outfile
                                                     )

        missing_experiment_ids, done_experiments, failed_extraction = Missing_detector.execute()
        
        #if "diagnostics_Versions_PyRadiomics" in done_experiments.columns:
            #failed_samples = len(done_experiments[done_experiments["diagnostics_Versions_PyRadiomics"].isnull()])
            #if failed_samples > 0:
                #self.error.warning("Found {} samples with failed extraction! Try to redo extraction!".format(str(failed_samples)))
                #print("Found {} samples with failed extraction! Try to redo extraction!".format(str(failed_samples)))
                #failed_extraction = done_experiments[done_experiments["diagnostics_Versions_PyRadiomics"].isnull()]
                #done_experiments = done_experiments[~done_experiments["diagnostics_Versions_PyRadiomics"].isnull()]
                
                #remove extraction files from tmp folder
        #else:
            #print("Warning: PyRadiomics extraction Failed! Could not find basic features!")
            #self.error.warning("PyRadiomics extraction Failed! Could not find basic features!")
            
                  
        print("Found {} done experiments.".format(str(len(done_experiments))))

        # print(done_experiments.index.tolist())
        # print(missing_experiment_ids)

        # get done experiments from subject folder
        # Missing_detector = MissingExperimentDetector(extractor=self.extractor,  # Either PyRadiomics or MIRP
        #                                              out_path=self.extracted_features_dir,  # Path to output directory
        #                                              verbose=self.verbose,  # Verbose output level
        #                                              logger=self.logger,  # Logger object
        #                                              subject_dir=self.subject_dir,  # Path to subject directory
        #                                              all_experiments_id=done_experiments.index.tolist(),
        #                                              RunID=self.RunID,
        #                                              num_cpus=self.n_cpu,
        #                                              error=self.error,
        #                                              )
        #
        # subject_exp, subject_samples = Missing_detector.get_done_experiments_from_subject_folder(
        #                                             done_samples=done_experiments.index.tolist(),
        #                                             done_experiments=done_experiments)
        #
        # done_experiments = pd.concat([done_experiments, subject_exp])

        # TODO: the ids got replaced to Unnamed: 0
        if "Unnamed: 0" in done_experiments.columns:
            done_experiments.set_index("Unnamed: 0", inplace=True)
            
            if len(failed_extraction) > 0:
                failed_extraction.set_index("Unnamed: 0", inplace=True)

        # remove done experiments from missing experiments
        missing_experiment_ids = [missing_exp for missing_exp in missing_experiment_ids if missing_exp not in done_experiments.index.tolist()]
        
        if len(failed_extraction) > 0:
            for failed_exp in tqdm(failed_extraction.index.tolist(), desc="Setting up failed PyRadiommics experiments", unit="sample"):
                 if failed_exp not in missing_experiment_ids:
                    missing_experiment_ids.append(failed_exp)

        # check if outfile exists and containes processed files
        # missing_samples, done_file_paths, done_exps = self.check_for_missing_experiments()
        #
        # open(self.outfile, 'a').close()  # create empty file
        # # open(self.out_path + "logs/pyradiomics_feature_calculation_run.log", 'a').close()  # create empty file
        missing_df = pd.DataFrame()
        
        print("Starting PyRadiomics feature extraction ...")
        if len(missing_experiment_ids) > 0:
            print("Need to extract features for", len(missing_experiment_ids), "samples.")
            self.logger.info("Need to extract features for " + str(len(missing_experiment_ids)) + " samples.")

            # process all samples which are not included in the outfile!
            self.logger.info("Starting PyRadiomics feature extraction of missing/failed samples ...")
            self.logger.info("Extracting features for " + str(len(missing_experiment_ids)) + " samples.")
            # print("Extraction of samples: " + str(len(missing_experiment_ids)))

            # shrink self.df to only containing missing samples in the config
            if "config" in self.df.columns:
                # check if missing_samples are presnet in the config to get the Image and the Mask
                missing_df = self.df[self.df['config'].isin(missing_experiment_ids)].copy()
            else:
                # if config is not there generate it
                image = self.df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
                mask = self.df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

                id = []
                for x, y in zip(image, mask):
                    id.append(x + "_" + y)

                self.df["config"] = id

                for sample in missing_experiment_ids:
                    if sample not in self.df["config"].to_list():
                        missing_experiment_ids.remove(sample)

                missing_df = self.df[self.df['config'].isin(missing_experiment_ids)].copy()
                # missing_df.to_csv("missing_df.csv")
                # drop config column
                self.df.drop(columns=["config"], inplace=True)
            
            if len(failed_extraction) > 0:
                if "Image" in  failed_extraction.columns:
                    # if config is not there generate it
                    image = failed_extraction["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
                    mask = failed_extraction["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

                    id = []
                    for x, y in zip(image, mask):
                        id.append(x + "_" + y)

                    failed_extraction["config"] = id
                    
                    print("Add failed extractions to list of experiments for extraction.")
                    
                    # add failed extraction to missing samples
                    missing_df = pd.concat([missing_df,failed_extraction])
            
            # missing_path = self.out_path + self.extractor + "_complete_processing_" + self.RunID + ".csv"
            # missing_df.to_csv(missing_path, index=False)

            extractor_setting = self.get_extractor_config()

            if self.fast_mode:
                py_extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_setting)
                execute_pyradiomics_experiment_partial = partial(self.execute_pyradiomics_experiment,
                                                                 py_extractor=py_extractor)
            else:
                execute_pyradiomics_experiment_partial = partial(self.execute_pyradiomics_experiment,
                                                                 extractor_setting=extractor_setting)

            missing_df = self.check_input_format(df=missing_df)
            result = self.extract_features(func=execute_pyradiomics_experiment_partial,
                                           entries=missing_df,
                                           extractor="PyRadiomics",
                                           returning=False)
            # try:
            #     with tqdm(total=len(missing_df), desc='PyRadiomics Feature Extraction') as pbar:
            #         chunk_size = self.n_cpu  # Adjust the chunk size based on your memory constraints
            #         for start in range(0, len(missing_df), chunk_size):
            #             chunk = missing_df.iloc[start:start + chunk_size]
            #
            #             with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            #                 futures = {executor.submit(self.execute_pyradiomics_experiment, row): row for i, row in
            #                            chunk.iterrows()}
            #                 for future in concurrent.futures.as_completed(futures):
            #                     pbar.update(1)
            #                     result = future.result()
            #                     del result  # Free memory immediately
            #                     gc.collect()  # Force garbage collection
            #
            #             del chunk  # Free memory for the chunk
            #             gc.collect()  # Force garbage collection

            # with tqdm(total=len(missing_df), desc='PyRadiomics Feature Extraction') as pbar:
            #     with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            #         futures = {executor.submit(self.execute_pyradiomics_exper Feature Extractioniment, row): row for i, row in missing_df.iterrows()}
            #         for future in concurrent.futures.as_completed(futures):
            #             pbar.update(1)
            #             #result = future.result()
            #             #del result
            #             # done_experiments = pd.concat([done_experiments, future.result()], ignore_index=True)
            #             gc.collect()  # Force garbage collection
            # except Exception as ex:
            #     self.error.error("PyRadiomics Feature Extraction Failed! " + str(ex))

            if len(done_experiments) != len(glob.glob(self.subject_dir + "/*.csv")):
                # put result of missing samples together with results of already processed samples in tmp folder
                complete_result, failed = Missing_detector.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"))
                # complete_result = self.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"),
                #                                         ignore_index=False)
            else:
                complete_result = done_experiments

            complete_result.to_csv(self.outfile)

            # if self.verbose == 0:
            #     os.system("pyradiomics " + os.path.abspath(missing_path) + " -o " + self.outfile + " -f csv -v 2 -j " +
            #               str(self.n_cpu) + " --param " + os.path.abspath(self.extraction_yaml) + " &> " + self.out_path +
            #               "logs/pyradiomics_feature_calculation_run.log")
            #     self.pyradiomics_wait_for_output()
            # else:
            #     os.system("pyradiomics " + os.path.abspath(missing_path) + " -o " + self.outfile + " -f csv -j " +
            #               str(self.n_cpu) + " --param " + os.path.abspath(
            #         self.extraction_yaml) + " &> " + self.out_path +
            #               "logs/pyradiomics_feature_calculation_run.log")
            #     self.pyradiomics_wait_for_output()

            self.logger.info("PyRadiomics feature extraction finished.")
            self.logger.info("Include processed feature extraction and compete results.")

            # if len(done_exps) > 0:
            #     # put result of missing samples together with results of already processed samples
            #     csv_files = [self.outfile] + done_file_paths
            #     complete_result = self.concat_extraction(csv_files=csv_files)

        else:
            print("Found {} done experiments!".format(str(len(done_experiments))))
            # complete_result = done_experiments

            # if there is no outfile and no tmp files --> process all samples
            if not os.path.exists(self.outfile) and (len(glob.glob(self.subject_dir + "/*.csv")) == 0):
                print("No out file nor experiment files... process all!")
                self.df = self.check_input_format(df=self.df)
                result = self.extract_features(func=self.execute_pyradiomics_experiment,
                                               entries=self.df,
                                               extractor="PyRadiomics",
                                               returning=False)

                # try:
                #     with tqdm(total=len(self.df), desc='PyRadiomics Feature Extraction') as pbar:
                #         chunk_size = self.n_cpu  # Adjust the chunk size based on your memory constraints
                #         for start in range(0, len(self.df), chunk_size):
                #             chunk = self.df.iloc[start:start + chunk_size]
                #
                #             with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                #                 futures = {executor.submit(self.execute_pyradiomics_experiment, row): row for i, row in
                #                            chunk.iterrows()}
                #                 for future in concurrent.futures.as_completed(futures):
                #                     pbar.update(1)
                #                     # result = future.result()
                #                     # del result  # Free memory immediately
                #                     gc.collect()  # Force garbage collection
                #
                #             del chunk  # Free memory for the chunk
                #             gc.collect()  # Force garbage collection
                #
                #     # with tqdm(total=len(self.df), desc='PyRadiomics Feature Extraction') as pbar:
                #     #     with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                #     #         futures = {executor.submit(self.execute_pyradiomics_experiment, row): row for i, row in
                #     #                    self.df.iterrows()}
                #     #         for future in concurrent.futures.as_completed(futures):
                #     #             pbar.update(1)
                #     #             #result = future.result()
                #     #             #del result
                #     #             # done_experiments = pd.concat([done_experiments, future.result()], ignore_index=True)
                #     #             gc.collect()  # Force garbage collection
                # except Exception as ex:
                #     self.error.error("PyRadiomics Feature Extraction Failed! " + str(ex))

                if len(done_experiments) != len(glob.glob(self.subject_dir + "/*.csv")):
                    # put result of missing samples together with results of already processed samples in tmp folder
                    complete_result, failed = Missing_detector.concat_extraction(
                        csv_files=glob.glob(self.subject_dir + "/*.csv"))
                    # complete_result = self.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"),
                    #                                         ignore_index=False)
                else:
                    complete_result = done_experiments
                # complete_result = self.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"),
                #                                          ignore_index=False)
                complete_result.to_csv(self.outfile)

            #     os.system("pyradiomics " + os.path.abspath(self.path2confCSV) + " -o " + self.outfile + " -f csv -v 2 -j " +
            #               str(self.n_cpu) + " --param " + os.path.abspath(
            #         self.extraction_yaml) + " &> " + self.out_path +
            #               "logs/pyradiomics_feature_calculation_run.log")
            #
            #     self.pyradiomics_wait_for_output()
            #     self.logger.info("PyRadiomics feature extraction finished.")
            # # if there is an outfile from previous run but it is empty

            elif (os.path.getsize(self.outfile) == 0) and (len(glob.glob(self.subject_dir + "/*.csv")) == 0):
                print("Out file empty and no experiment files... process all!")
                self.df = self.check_input_format(df=self.df)
                result = self.extract_features(func=self.execute_pyradiomics_experiment,
                                               entries=self.df,
                                               extractor="PyRadiomics",
                                               returning=False)
                # try:
                #     with tqdm(total=len(self.df), desc='PyRadiomics Feature Extraction') as pbar:
                #         chunk_size = self.n_cpu  # Adjust the chunk size based on your memory constraints
                #         for start in range(0, len(self.df), chunk_size):
                #             chunk = self.df.iloc[start:start + chunk_size]
                #
                #             with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                #                 # Maybe faster and mem efficient if using map and chunksize=1
                #                 # for result in executor.map(self.read_file, [file for file in chunk],
                #                 #                            chunksize=1):
                #                 #     pbar.update(1)
                #                 #     if result is not None:
                #                 #         df = pd.concat([df, result], ignore_index=True)
                #                 #     # del file
                #                 #         gc.collect()
                #
                #                 futures = {executor.submit(self.execute_pyradiomics_experiment, row): row for i, row in
                #                            chunk.iterrows()}
                #                 for future in concurrent.futures.as_completed(futures):
                #                     pbar.update(1)
                #                     # result = future.result()
                #                     # del result  # Free memory immediately
                #                     gc.collect()  # Force garbage collection
                #
                #             del chunk  # Free memory for the chunk
                #             gc.collect()  # Force garbage collection
                #
                # except Exception as ex:
                #     self.error.error("PyRadiomics Feature Extraction Failed! " + str(ex))

                if len(done_experiments) != len(glob.glob(self.subject_dir + "/*.csv")):
                    complete_result, failed = Missing_detector.concat_extraction(
                        csv_files=glob.glob(self.subject_dir + "/*.csv"))
                    # complete_result = self.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"),
                    #                                         ignore_index=False)

                complete_result.to_csv(self.outfile)

            elif (os.path.getsize(self.outfile) == 0) and (len(glob.glob(self.subject_dir + "/*.csv")) > 0):
                print("Out file empty. Collecting Results from experiment files...")
                self.logger.info("Collecting Results from tmp! PyRadiomics feature extraction already done ...")
                complete_result, failed = Missing_detector.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"))
                # complete_result = self.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"),
                #                                         ignore_index=False)
                complete_result.to_csv(self.outfile)

            elif os.path.getsize(self.outfile) > 0:
                print("Out file not empty. Extraction done!")
                complete_result = pd.read_csv(self.outfile, index_col=0)

                # check if it is not only the header but extraction
                if len(complete_result) == len(glob.glob(self.subject_dir + "/*.csv")):
                    self.logger.info("Skipping! PyRadiomics feature extraction already done: " + self.outfile)
                else:
                    self.logger.info(
                        "PyRadiomics feature extraction failed previously! Gathering results from checkpoints ...")

                    if len(done_experiments) > len(complete_result):
                        complete_result = pd.concat([complete_result, done_experiments], ignore_index=True)
                        complete_result.drop_duplicates(inplace=True)

                    if not len(complete_result) == len(glob.glob(self.subject_dir + "/*.csv")):
                        complete_result, failed = Missing_detector.concat_extraction(
                            csv_files=glob.glob(self.subject_dir + "/*.csv"))
                        # complete_result = self.concat_extraction(csv_files=glob.glob(self.subject_dir + "/*.csv"),
                        #                                     ignore_index=False)
                    complete_result.to_csv(self.outfile)

            #     print("Redo PyRadiomics feature extraction as it failed previously!")
            #     self.logger.info("Redo PyRadiomics feature extraction as it failed previously!")
            #     os.system("pyradiomics " + os.path.abspath(self.path2confCSV) + " -o " + self.outfile + " -f csv -v 2 -j " +
            #               str(self.n_cpu) + " --param " + os.path.abspath(
            #         self.extraction_yaml) + " > " + self.out_path +
            #               "logs/pyradiomics_features.log")
            #     self.pyradiomics_wait_for_output()

            # elif os.path.getsize(self.outfile) > 0:
            #     self.logger.info("Skipping! PyRadiomics feature extraction already done: " + self.outfile)
            #
            #     if len(complete_result) == 0:
            #         complete_result = pd.read_csv(self.outfile)

            elif self.use_previous_output:
                self.error.warning("PyRadiomics feature extraction failed previously! No Output file found!")
            else:
                self.error.warning("PyRadiomics feature extraction failed! No Output file found!")

        # ensure that all samples are included in the feature extraction
        csv_files = glob.glob(self.subject_dir + "/*.csv")

        if (len(done_experiments) == len(csv_files)) and (len(complete_result) < len(csv_files)):
            complete_result = done_experiments

        if (len(complete_result) < len(csv_files)) or (len(complete_result) == 0):
            self.logger.info(
                "Not all extractions are included in feature out file! " + str(len(complete_result)) + " " + str(
                    len(csv_files)))
            # if there is not outfile in the tmp folder
            if len(csv_files) == 0:
                self.error.error("Feature extraction failed. No out files found in {}!".format(str(self.subject_dir)))
                raise ValueError("Feature extraction failed. No out files found in {}!".format(str(self.subject_dir)))
            else:
                complete_result, failed = Missing_detector.concat_extraction(csv_files=csv_files)

        return complete_result

    def pyradiomics_wait_for_output(self):
        if not os.path.exists(self.outfile):
            size = os.path.getsize(self.outfile)
        while (not os.path.exists(self.outfile)) and (not os.path.getsize(self.outfile) > 0):
            time.sleep(10)

        if os.path.isfile(self.outfile):
            pyradiomics_output = pd.read_csv(self.outfile)
            self.logger.info("PyRadiomics extracted {} features for {} samples".format(
                str(len(pyradiomics_output.columns)), str(len(pyradiomics_output))))
        else:
            raise ValueError("%s isn't a file!" % self.outfile)

    def postprocessing(self):
        """
        Postprocessing of the extracted features.
        Filtering for Variance of features and correlation between features.
        """

        self.logger.info("Starting postprocessing")

        # radiomics_filter = RadiomicsFilter(path_to_data=self.outfile,
        #                         path_to_img_seg_csv=self.path2confCSV,
        #                         variance_threshold=self.variance_threshold,
        #                         correlation_threshold=self.correlation_threshold,
        #                         extractor=self.extractor,
        #                         logger=self.logger,
        #                         RunID=self.RunID,
        #                         n_cpu=self.n_cpu)

        # self.features = radiomics_filter.run()

        self.logger.info("### Postprocessing finished ###\n")

    @staticmethod
    def delta_calculation(df, pos1, pos2, logger):
        """
        Compute the delta radiomics features between two time points.
        :param df: DataFrame containing the radiomics features.
        :param pos1: The first time point.
        :param pos2: The second time point.
        :return: DataFrame containing the delta radiomics features.
        """

        # check if Timepoint column exists
        df1 = df[df["Timepoint"] == pos1]
        df1 = df1.sort_values(['ID', 'Timepoint'], ascending=[True, True])
        df1 = df1.set_index('ID')

        logger.info(pos1, "radiomics:")
        logger.info(str(df1.shape[0]) + " Samples")
        logger.info(str(df1.shape[1]) + " Features\n")

        df2 = df[df["Timepoint"] == pos2]
        df2 = df2.sort_values(['ID', 'Timepoint'], ascending=[True, True])
        df2 = df2.set_index('ID')

        logger.info(pos2, "radiomics:")
        logger.info(str(df2.shape[0]) + " Samples")
        logger.info(str(df2.shape[1]) + " Features\n")

        # save response for later
        df12 = pd.concat([df1, df2])  # , join='outer')
        df3 = df12[~df12.index.duplicated(keep='first')] # TODO: check if this is correct

        # only include values with int or float
        df1 = df1.select_dtypes(include=['float', 'int'])
        df2 = df2.select_dtypes(include=['float', 'int'])

        delta = df2.subtract(df1, axis='columns', level=None, fill_value=None).dropna()

        delta_label = list(itertools.repeat(str(pos2) + "-" + str(pos1), delta.shape[0]))

        delta.insert(0, 'ID', delta.index)
        delta.insert(1, 'Delta', delta_label)

        delta = delta.drop(['Timepoint'], axis=1)

        logger.info("Delta_" + str(pos1) + "_" + str(pos2) + ":")
        logger.info(str(delta.shape[0]) + " Samples")
        logger.info(str(delta.shape[1]) + " Features\n")

        if delta.isnull().values.any():
            logger.info("Samples with NaN:")
            logger.info(delta[delta.isna().empty(axis=1)])
        else:
            logger.info("No Samples with NaN!\n")

        return delta

    @staticmethod
    def remove_timepoint_from_ID(df, id_col='ID'):
        """
        Removes the '-Timepoint' postfix from the ID column.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the modified ID column.
        id_col (str): The name of the ID column. Default is 'ID'.

        Returns:
        pd.DataFrame: A DataFrame with the original ID values.
        """

        df[id_col] = df[id_col].astype(str).str.rsplit('-', n=1).str[0]
        return df

    @staticmethod
    def calc_differences(df, t1, t2, logger, delta_radiomics:pd.DataFrame, numeric_cols, _id, take_label_changes, string_cols):
        """
        Calculate the differences between the timepoints.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the radiomics features.
        timepoints (list): A list of timepoints to calculate the differences for.

        Returns:
        pd.DataFrame: A DataFrame containing the differences between the timepoints.
        """

        delta = pd.DataFrame()

        # Sort by 'Image' and then 'Mask' to have same iterations for random contour change
        df = df.sort_values(by=['Image', 'Mask'])

        df_t1 = df[(df["Timepoint"] == float(t1))]
        df_t2 = df[(df["Timepoint"] == float(t2))]

        # check for image transformation
        if df_t1["Image_Transformation"].to_list() != df_t2["Image_Transformation"].to_list():
            logger.error("Image transformations differ between timepoints {} and {} for sample {}: {} {}".format(t1, t2, _id, df_t1["Image_Transformation"].to_list(), df_t2["Image_Transformation"].to_list()))
            raise ValueError("Image transformations differ between timepoints {} and {} for sample {}: {} {}".format(t1, t2, _id, df_t1["Image_Transformation"].to_list(), df_t2["Image_Transformation"].to_list()))
        else:
            img_trans = df_t1["Image_Transformation"].values[0]

        # check for mask transformation
        if df_t1["Mask_Transformation"].to_list() != df_t2["Mask_Transformation"].to_list():
            logger.error("Mask transformations differ between timepoints {} and {} for sample {}!".format(t1, t2, _id, _id, df_t1["Mask_Transformation"].to_list(), df_t2["Mask_Transformation"].to_list()))
            raise ValueError("Mask transformations differ between timepoints {} and {} for sample {}!".format(t1, t2, _id, _id, df_t1["Mask_Transformation"].to_list(), df_t2["Mask_Transformation"].to_list()))
        else:
            mask_trans = df_t1["Mask_Transformation"].values[0]

        df_t1_label = df_t1.copy().loc[:, "Prediction_Label"]
        df_t2_label = df_t2.copy().loc[:, "Prediction_Label"]

        df_t1 = df_t1.drop(columns=["Prediction_Label"])
        df_t2 = df_t2.drop(columns=["Prediction_Label"])

        # Skip if one of the timepoints is missing for this transformation
        if df_t1.empty or df_t2.empty:
            print(f"Could not find timepoints {t1} and {t2} for sample {_id} in data!")
            return delta_radiomics
            # continue  

        # if multiple samples are included for each timepoint
        if df_t1.shape[0] > 1 or df_t2.shape[0] > 1:
            if df_t1.shape[0] != df_t2.shape[0]:
                logger.error("Timepoints {} and {} for sample {} have failed extractions!".format(t1, t2, _id))
                if df_t1.shape[0] > df_t2.shape[0]:
                    logger.error("Timepoint {} has more than one extraction for sample {}!".format(t1, _id))
                else:
                    logger.error("Timepoint {} has more than one extraction for sample {}!".format(t2, _id))
                raise ValueError("Timepoints {} and {} for sample {} have failed extractions!".format(t1, t2, _id))
            

        # Compute delta using only numeric feature columns
        delta_features = df_t2[numeric_cols].values - df_t1[numeric_cols].values
        delta = pd.DataFrame(delta_features, columns=numeric_cols)

        # Assign metadata information
        #delta.insert(loc=0, column='ID', value=f"{_id}-delta-{t1}-{t2}")
        #delta.insert(loc=1, column='Image_Transformation', value=img_trans if pd.notna(img_trans) else np.nan)
        #delta.insert(loc=2, column='Mask_Transformation', value=mask_trans if pd.notna(mask_trans) else np.nan)

        delta["ID"] = f"{_id}-delta-{t1}-{t2}"  # New ID format 
        delta["Image_Transformation"] = img_trans if pd.notna(img_trans) else np.nan
        delta["Mask_Transformation"] = mask_trans if pd.notna(mask_trans) else np.nan

        if not df_t1_label.isna().all() and not df_t2_label.isna().all():
            if take_label_changes:
                if df_t1_label.values[0] == df_t2_label.values[0]: # if labels at timepoints are equal
                    delta["Prediction_Label"] = df_t1_label.values[0]
                elif (df_t1_label.values[0] == 1) and (df_t2_label.values[0] == 0): # if label at timepoint 1 is 1 and timepoint 2 is 0 --> 2
                    delta["Prediction_Label"] = 2
                else: # if label at timepoint 1 is 0 and timepoint 2 is 1 --> -1
                    delta["Prediction_Label"] = df_t1_label.values[0] - df_t2_label.values[0]
            else:
                if df_t1_label.values[0] == df_t2_label.values[0]: # if labels at timepoints are equal
                    delta["Prediction_Label"] = df_t1_label.values[0]
                else: # if labels at timepoints defer take the label from the latest timepoint
                    delta["Prediction_Label"] = df_t2_label.values[0]
            
        else:
            logger.warning("Could not find Prediction_Label for sample {} at timepoints {} and {}!".format(_id, t1, t2))
            raise ValueError("Could not find Prediction_Label for sample {} at timepoints {} and {}!".format(_id, t1, t2))
            # delta["Prediction_Label"] = np.nan

        # Merge string columns correctly
        for col in string_cols:
            val_t1 = df_t1[col].values if not df_t1[col].isna().all() else np.nan
            val_t2 = df_t2[col].values if not df_t2[col].isna().all() else np.nan

            # if one of both values are nan --> skip
            if (val_t1 is np.nan) or (val_t2 is np.nan):
                delta[col] = np.nan
                continue

            if list(val_t1) == list(val_t2):
                delta[col] = val_t1
                continue

            if len(val_t1) == len(val_t2):
                del_str = []
                for str_t1, str_t2 in zip(val_t1, val_t2):
                    del_str.append(f"{str_t1} - {str_t2}")
                delta[col] = del_str
            else:
                logger.warning("Unequal size for feature {} for sample {} at timepoints {} and {}!".format(col,_id, t1, t2))
                raise ValueError("Unequal size for feature {} for sample {} at timepoints {} and {}!".format(col,_id, t1, t2))

        # keep the format where stirng columns are on the fisr positions
        #other_cols = [col for col in delta.columns if col not in string_cols]

        # Reorder the DataFrame
        #delta = delta[string_cols + other_cols]

        # Append to the result
        delta_radiomics = pd.concat([delta, delta_radiomics], ignore_index=True)

        return delta_radiomics

    @staticmethod
    def compute_delta_radiomics(features, logger, take_label_changes):
        """
        Computes delta radiomics by subtracting later timepoints from earlier ones, considering
        Image_Transformation and Mask_Transformation while ensuring consistent ID tracking.

        Args:
            features (pd.DataFrame): DataFrame containing ID, Timepoint, Image_Transformation, and Mask_Transformation.
            logger: logging
            take_label_changes: Consider changing label over time and adapt the label to the label of the last timepoint

        Returns:
            pd.DataFrame: A DataFrame containing delta radiomics with correctly formatted IDs.
        """

        # Ensure Timepoint is numeric
        features["Timepoint"] = pd.to_numeric(features["Timepoint"], errors='coerce')

        # check for NaN values in Timepoint column
        if features["Timepoint"].isnull().any():
            self.error.error("Timepoint column contains NaN values! Please ensure all samples have a Timepoint value!")
            raise ValueError("Timepoint column contains NaN values! Please ensure all samples have a Timepoint value!")
            # features.dropna(subset=["Timepoint"], inplace=True)

        features["Timepoint"] = features["Timepoint"].astype(int)

        if "Image_Transformation" not in features.columns:
            features["Image_Transformation"] = np.nan
        if "Mask_Transformation" not in features.columns:
            features["Mask_Transformation"] = np.nan

        # Sort by ID, Timepoint, and transformations (if available)
        features = features.sort_values(["ID", "Timepoint", "Image_Transformation", "Mask_Transformation"], 
                                        ascending=[True, True, True, True])

        # Identify column types dynamically
        metadata_cols = ["ID", "Timepoint", "Image_Transformation", "Mask_Transformation", "Prediction_Label"]
        numeric_cols = features.select_dtypes(include=[np.number]).columns.difference(metadata_cols).tolist()
        string_cols = features.select_dtypes(exclude=[np.number]).columns.difference(metadata_cols).tolist()

        # Initialize new dataframe
        delta_radiomics = pd.DataFrame()

        # Get unique IDs
        unique_ids = features["ID"].unique().tolist()

        for _id in tqdm(unique_ids, desc="Calculating Delta Radiomics"):
            sample_df = features[features["ID"] == _id]  # Get all samples for this ID

            # Get unique timepoints in sorted order
            timepoints = sorted(sample_df["Timepoint"].unique())
            if len(timepoints) < 2:
                continue  # Skip if only one timepoint exists

            for i in range(len(timepoints) - 1):
                t1, t2 = timepoints[i], timepoints[i + 1]  # Previous and following timepoints

                # 1. get raw samples without Mask Transformation and Image Trabsformation
                raw_samples = sample_df.copy().loc[(sample_df["Image_Transformation"].isna()) & (sample_df["Mask_Transformation"].isna())]
                
                # replacing different values which were marked as nan
                raw_samples["Image_Transformation"] = np.nan
                raw_samples["Mask_Transformation"] = np.nan
                raw_samples['Image_Transformation']=raw_samples['Image_Transformation'].fillna("")
                raw_samples['Mask_Transformation']=raw_samples['Mask_Transformation'].fillna("")

                delta_radiomics = Extractor.calc_differences(df = raw_samples, 
                                                            t1 = t1, 
                                                            t2 = t2, 
                                                            logger = logger, 
                                                            delta_radiomics = delta_radiomics, 
                                                            numeric_cols = numeric_cols,
                                                            _id=_id,
                                                            take_label_changes=take_label_changes,
                                                            string_cols=string_cols)
                
                # 2. get samples with only Mask Transformation
                raw_mask_trans_samples = sample_df.copy().loc[(sample_df["Image_Transformation"].isna()) & (~sample_df["Mask_Transformation"].isna())]
                for msk_trans in raw_mask_trans_samples["Mask_Transformation"].unique():
                    raw_mask_trans_samples_single_trans = raw_mask_trans_samples.loc[raw_mask_trans_samples["Mask_Transformation"]==msk_trans]
                    
                    # replacing different values which were marked as nan
                    raw_mask_trans_samples_single_trans["Image_Transformation"] = np.nan
                    raw_mask_trans_samples_single_trans['Image_Transformation']=raw_mask_trans_samples_single_trans['Image_Transformation'].fillna("")

                    # checking for mask transformations and image transformations present in both timepoints
                    df_t1 = raw_mask_trans_samples_single_trans[(raw_mask_trans_samples_single_trans["Timepoint"] == float(t1))]
                    df_t2 = raw_mask_trans_samples_single_trans[(raw_mask_trans_samples_single_trans["Timepoint"] == float(t2))]

                    if (msk_trans in df_t1["Mask_Transformation"].to_list()) and (msk_trans in df_t2["Mask_Transformation"].to_list()):
                        delta_radiomics = Extractor.calc_differences(df = raw_mask_trans_samples_single_trans, 
                                                                    t1 = t1, 
                                                                    t2 = t2, 
                                                                    logger = logger, 
                                                                    delta_radiomics = delta_radiomics, 
                                                                    numeric_cols = numeric_cols,
                                                                    _id=_id,
                                                                    take_label_changes=take_label_changes,
                                                                    string_cols=string_cols)
                    else:
                        logger.warning(f"Could not find Mask Transformation {msk_trans} in both timepoints {t1} and {t2} for sample {_id}!")
                        # print(f"Could not find Mask Transformation {msk_trans} in both timepoints {t1} and {t2} for sample {_id}!")

                # 3. get samples with only Image Transformation
                raw_img_trans_samples = sample_df.copy().loc[(~sample_df["Image_Transformation"].isna()) & (sample_df["Mask_Transformation"].isna())]
                for img_trans in raw_img_trans_samples["Image_Transformation"].unique():
                    raw_img_trans_samples_single_trans = raw_img_trans_samples.loc[raw_img_trans_samples["Image_Transformation"]==img_trans]
                    
                    # replacing different values which were marked as nan
                    raw_img_trans_samples_single_trans["Mask_Transformation"] = np.nan
                    raw_img_trans_samples_single_trans['Mask_Transformation']=raw_img_trans_samples_single_trans['Mask_Transformation'].fillna("")

                    # checking for mask transformations and image transformations present in both timepoints
                    df_t1 = raw_img_trans_samples_single_trans[(raw_img_trans_samples_single_trans["Timepoint"] == float(t1))]
                    df_t2 = raw_img_trans_samples_single_trans[(raw_img_trans_samples_single_trans["Timepoint"] == float(t2))]

                    if (img_trans in df_t1["Image_Transformation"].to_list()) and (img_trans in df_t2["Image_Transformation"].to_list()):

                        delta_radiomics = Extractor.calc_differences(df = raw_img_trans_samples_single_trans, 
                                                                    t1 = t1, 
                                                                    t2 = t2, 
                                                                    logger = logger, 
                                                                    delta_radiomics = delta_radiomics, 
                                                                    numeric_cols = numeric_cols,
                                                                    _id=_id,
                                                                    take_label_changes=take_label_changes,
                                                                    string_cols=string_cols)
                    else:
                        logger.warning(f"Could not find Image Transformation {img_trans} in both timepoints {t1} and {t2} for sample {_id}!")
                        raise ValueError(f"Failed Image Transformation {img_trans} not in both timepoints {t1} and {t2} for sample {_id}!")

                # 4. get samples with both Mask and Image Transformation
                msk_img_trans_samples = sample_df.copy().loc[(~sample_df["Image_Transformation"].isna()) & (~sample_df["Mask_Transformation"].isna())]
                for img_trans in msk_img_trans_samples["Image_Transformation"].unique():
                    for msk_trans in msk_img_trans_samples.loc[msk_img_trans_samples["Image_Transformation"]==img_trans,"Mask_Transformation"].unique():
                        raw_img_msk_trans_samples_single_trans = msk_img_trans_samples.loc[(msk_img_trans_samples["Image_Transformation"]==img_trans) & (msk_img_trans_samples["Mask_Transformation"] == msk_trans)]

                        # checking for mask transformations and image transformations present in both timepoints
                        df_t1 = raw_img_msk_trans_samples_single_trans[(raw_img_msk_trans_samples_single_trans["Timepoint"] == float(t1))]
                        df_t2 = raw_img_msk_trans_samples_single_trans[(raw_img_msk_trans_samples_single_trans["Timepoint"] == float(t2))]

                        if (img_trans in df_t1["Image_Transformation"].to_list()) and (img_trans in df_t2["Image_Transformation"].to_list()):
                            if (msk_trans in df_t1["Mask_Transformation"].to_list()) and (msk_trans in df_t2["Mask_Transformation"].to_list()):

                                delta_radiomics = Extractor.calc_differences(df = raw_img_msk_trans_samples_single_trans, 
                                                                            t1 = t1, 
                                                                            t2 = t2, 
                                                                            logger = logger, 
                                                                            delta_radiomics = delta_radiomics, 
                                                                            numeric_cols = numeric_cols,
                                                                            _id=_id,
                                                                            take_label_changes=take_label_changes,
                                                                            string_cols=string_cols)

                # Get all unique image & mask transformations for these timepoints
                #for (img_trans, mask_trans) in sample_df.groupby(["Image_Transformation", "Mask_Transformation"]).groups.keys():
                #    if pd.isna(mask_trans):
                #        if pd.isna(img_trans):
                #            df_t1 = sample_df[(sample_df["Timepoint"] == float(t1))]
                #            df_t2 = sample_df[(sample_df["Timepoint"] == float(t2))]
                #        else:
                #            df_t1 = sample_df[(sample_df["Timepoint"] == float(t1)) & 
                #                        (sample_df["Image_Transformation"] == img_trans)]
                #            df_t2 = sample_df[(sample_df["Timepoint"] == float(t2)) & 
                #                            (sample_df["Image_Transformation"] == img_trans)]
                #    elif pd.isna(img_trans):
                #        df_t1 = sample_df[(sample_df["Timepoint"] == float(t1)) &
                #                        (sample_df["Mask_Transformation"] == mask_trans)]
                #        df_t2 = sample_df[(sample_df["Timepoint"] == float(t2)) &
                #                        (sample_df["Mask_Transformation"] == mask_trans)]
                #    else:
                #        df_t1 = sample_df[(sample_df["Timepoint"] == float(t1)) & 
                #                        (sample_df["Image_Transformation"] == img_trans) & 
                #                        (sample_df["Mask_Transformation"] == mask_trans)]
                #        df_t2 = sample_df[(sample_df["Timepoint"] == float(t2)) & 
                #                        (sample_df["Image_Transformation"] == img_trans) & 
                #                        (sample_df["Mask_Transformation"] == mask_trans)]

        return delta_radiomics

    @staticmethod
    def get_delta_radiomics(features, extractor="", time_format="%Y-%m-%d", out_path="", logger=None, error=None, take_label_changes=False):
        """
        Compute the delta radiomics features of all time points.
        :param features: The radiomics features DataFrame.
        :param extractor: The name of the extractor.
        :param time_format: The time format.
        :param out_path: The output path.
        :param logger: The logger object.
        :param error: The error object.
        :return: The delta radiomics features DataFrame.
        """

        logger.info("Computing delta radiomics features")

        features = features.drop_duplicates(subset=['Image', 'Mask'])

        if "Timepoint" not in features.columns:
            error.warning("No Timepoint column found in radiomics features!")
            print("No Timepoint column found in radiomics features!")
            return None
        else:
            try:
                # check if Timepoint column is numeric
                features["Timepoint"] = pd.to_numeric(features["Timepoint"])
                # self.features["Timepoint"] = self.features["Timepoint"].apply(pd.to_numeric)
            except:
                error.warning("Timepoint column is not numeric!")
                print("Timepoint column is not numeric!")
            
            features = Extractor.remove_timepoint_from_ID(df=features.copy())

            # check if Timepoint column has only one digit - sort by date
            if features["Timepoint"].max() > 9:
                logger.info("Timepoint column has more than one digit! Sort by date!")
                print("Timepoint column has more than one digit! Sort by date!")
                features['Timepoint'] = pd.to_datetime(features['Timepoint'], format=time_format)
                features['RANK'] = features.groupby('ID')['Timepoint'].rank(ascending=True)
                features['RANK'] = features['RANK'].astype('int')
                features['Timepoint'] = features['RANK']
                features = features.drop(columns=['RANK'])

        features["Timepoint"] = features["Timepoint"].astype('int')
        features = features.sort_values(['ID', 'Timepoint'], ascending=[True, True])

        # calculate delta radiomics
        delta_radiomics = Extractor.compute_delta_radiomics(features=features, logger=logger, take_label_changes=take_label_changes)
        
        #if extractor != "":
            # Save delta radiomics
            # delta_radiomics.to_csv(out_path + "extracted_features/" + extractor + "_delta_radiomics.csv")
        #else:
            # Save delta radiomics
            # delta_radiomics.to_csv(out_path + "delta_radiomics.csv")

        logger.info("Delta radiomics features computed and saved")

        return delta_radiomics

    def crop_exe(self, entry: pd.Series):
        """
        Execute Cropping the Images and Masks
        :param entry: entry from config file to crop and write img and mask to cropping_img/msk_folder_path
        """
        if ("_cropped" not in entry["Raw_Image"]) and ("_cropped" not in entry["Raw_Mask"]):
            # define outout files
            img_path = self.cropping_img_folder_path + "/" + os.path.basename(entry["Raw_Image"])[
                                                             :-len(".nii.gz")] + "_" + os.path.basename(
                entry["Raw_Mask"])[
                                                                                       :-len(
                                                                                           ".nii.gz")] + "_cropped_resample.nii.gz"

            msk_path = self.cropping_msk_folder_path + "/" + os.path.basename(entry["Raw_Mask"])[
                                                             :-len(".nii.gz")] + "_" + os.path.basename(
                entry["Raw_Image"])[
                                                                                       :-len(
                                                                                           ".nii.gz")] + "_cropped_resample.nii.gz"

            # check if they exist
            img_exist = os.path.exists(img_path)
            msk_exist = os.path.exists(msk_path)

            if (not img_exist) or (not msk_exist):
                if not pd.isna(entry["ROI_Label"]):
                    label = int(entry["ROI_Label"])
                else:
                    label = 1

                image = sitk.ReadImage(entry["Raw_Image"])
                mask = sitk.ReadImage(entry["Raw_Mask"])

                # transform img in int img:
                image_array = sitk.GetArrayFromImage(image)
                image_array = image_array.astype('int16')
                image = sitk.GetImageFromArray(image_array)

                # try:
                #     # check segmentation quality
                #     bouding_box = imageoperations.checkMask(image, mask)
                #
                # except Exception as ex:
                #     self.error.warning("Segmentation Quality check failed! " + str(ex))

                try:
                    # cropping and resampling
                    (res_cropped_img, res_cropped_msk) = imageoperations.resampleImage(image, mask,
                                                                                       resampledPixelSpacing=
                                                                                       self.rptk_config[
                                                                                           "Feature_extraction_config"][
                                                                                           "resampledPixelSpacing"],
                                                                                       label=label)
                    # write cropped img/msk
                    if not img_exist:
                        sitk.WriteImage(res_cropped_img, img_path)

                    if not msk_exist:
                        sitk.WriteImage(res_cropped_msk, msk_path)

                    entry["Image"] = img_path
                    entry["Mask"] = msk_path

                except Exception as ex:
                    self.error.warning(
                        "Resampling failed for " + os.path.basename(entry["Raw_Image"]) + " " + os.path.basename(
                            entry["Raw_Mask"]) + " : " + str(ex) + " Try cropping only!")

                    # raise Exception(ex)

                    try:
                        (res_cropped_img, res_cropped_msk) = imageoperations.cropToTumorMask(image,
                                                                                             mask,
                                                                                             bouding_box)

                        # write cropped img/msk
                        if not img_exist:
                            sitk.WriteImage(res_cropped_img, img_path)

                        if not msk_exist:
                            sitk.WriteImage(res_cropped_msk, msk_path)

                        entry["Image"] = img_path
                        entry["Mask"] = msk_path

                    except Exception as ex:
                        self.error.warning(
                            "RPTK cropping failed for " + os.path.basename(entry["Raw_Image"]) + " " + os.path.basename(
                                entry["Raw_Mask"]) + " : " + str(ex) + " Using raw data!")

                del image
                del mask

            else:
                entry["Image"] = img_path
                entry["Mask"] = msk_path

        # replace data in Image/Mask column
        # self.df.loc[entry["Raw_Image"] == self.df["Raw_Image"], "Image"] = img_path
        # self.df.loc[(entry["Raw_Image"] == self.df["Raw_Image"]) and (entry["Raw_Mask"] == self.df["Raw_Mask"]), "Mask"] = msk_path

        return entry

    def img_seg_cropping(self):
        """
        Crop Image and Mask to ROI and resample to spacing from self.df and add the paths to new created columns
        "Cropped_Image" and "Cropped_Mask"
        """

        self.logger.info("Cropping Image and Mask for less Mem consumption!")

        # generate a folder of storing cropped img and msk
        cropping_folder_path = os.path.abspath(os.path.join(os.path.dirname(self.out_path),
                                                            '..',
                                                            'preprocessed_data',
                                                            'cropped'))

        self.cropping_img_folder_path = cropping_folder_path + "/img"
        self.cropping_msk_folder_path = cropping_folder_path + "/msk"

        if not os.path.exists(self.cropping_img_folder_path):
            Path(self.cropping_img_folder_path).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.cropping_msk_folder_path):
            Path(self.cropping_msk_folder_path).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(cropping_folder_path + "/cropped_preprocessed_data_" + str(self.RunID) + ".csv"):

            # save raw img and msk so cropped img and msk can replace Image and Mask column
            if "Raw_Image" not in self.df.columns:
                self.df['Raw_Image'] = self.df['Image']

            if "Raw_Mask" not in self.df.columns:
                self.df['Raw_Mask'] = self.df['Mask']

            # Create an empty list
            Row_list = []

            # Iterate over each row
            for index, rows in self.df.iterrows():
                # append the list to the final list
                Row_list.append(rows)

            with Pool(processes=self.n_cpu) as pool:
                # Use tqdm with imap to show progress
                for result in tqdm(pool.imap(self.crop_exe, Row_list),
                                   total=len(self.df), desc="Cropping ROI", unit="masks"):
                    self.df.loc[(result["Raw_Image"] == self.df["Raw_Image"]) & (
                            result["Raw_Mask"] == self.df["Raw_Mask"]), "Image"] = result["Image"]
                    self.df.loc[(result["Raw_Image"] == self.df["Raw_Image"]) & (
                            result["Raw_Mask"] == self.df["Raw_Mask"]), "Mask"] = result["Mask"]

            # with tqdm(total=len(self.df), desc="Cropping ROI") as pbar:
            #     with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            #         # futures = {executor.submit(func, row): row for i, row in chunk.iterrows()}
            #         # for future in concurrent.futures.as_completed(futures):
            #         for results in executor.map(self.crop_exe, [row for i, row in self.df.iterrows()], chunksize=self.n_cpu):
            #             self.df.loc[(results["Raw_Image"] == self.df["Raw_Image"]) & (results["Raw_Mask"] == self.df["Raw_Mask"]), "Image"] = results["Image"]
            #             self.df.loc[(results["Raw_Image"] == self.df["Raw_Image"]) & (results["Raw_Mask"] == self.df["Raw_Mask"]), "Mask"] = results["Mask"]
            #             pbar.update(1)

            # for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Cropping ROI"):
            #     try:
            #         entry = self.crop_exe(entry=row)
            #         self.df.loc[i, :] = entry
            #     except Exception as ex:
            #         self.error.error("Cropping Failed: " + str(ex))

            cropped = self.df[self.df["Image"].str.endswith("_cropped_resample.nii.gz")]
            self.logger.info(
                "Cropped {} from {} samples sucessfully".format(str(cropped.shape[0]), str(self.df.shape[0])))
            del cropped
            self.df.to_csv(cropping_folder_path + "/cropped_preprocessed_data_" + str(self.RunID) + ".csv")
        else:
            self.logger.info("Loading cropping csv from previous run ...")
            self.df = pd.read_csv(cropping_folder_path + "/cropped_preprocessed_data_" + str(self.RunID) + ".csv",
                                  index_col=0)

    def adapt_optimized_feature_configuration(self, features: pd.DataFrame):
        """
        Syncronize feature nameing of dynamically optimized extractions to reduce the amount of NaN
        :param features: features with non-unique nameing including different configurations from MIRP
        """

        # affected features from configuration: 'cm', 'dzm', 'ih', 'ngl', 'ngt', 'rlm', 'szm'
        
        configuration_cols = ["ID", "Image", "Mask", "Modality", "Image_Transformation","Mask_Transformation","Rater", "Prediction_Label", "ROI_Label"]
        configed_cols = []
        
        
        if "id_subject" in features.columns:
            features.index = features["id_subject"]
        
        # check if all configs are in data
        configuration_cols_in_data = []
        for config in configuration_cols:
            if config not in features.columns:
                self.error.warning("Configuration {} is missing in extracted features!".format(str(config)))
                print("Configuration {} is missing in extracted features!".format(str(config)))
            else:
                configuration_cols_in_data.append(config)
                
        config_data = features[configuration_cols_in_data]
        
        # Configuration specific Parameter
        config_specific_param = ['ngl_dc_perc_d1_a0.0_2d_fb',]

        # Get all features with nan
        nan_features = features.columns[features.isna().any()].tolist()

        if len(nan_features)>0:
            
            # these columns are not considered as they contain nan by nature
            columns_with_nan = ["Rater", "Image_Transformation", "Mask_Transformation", "Timepoint", "ROI_Label"]
            nan_feature_ = []
            configs = []
            
            # get configs of these features
            for feat in nan_features:
                if feat not in columns_with_nan:
                    nan_feature_.append(feat)
                    
                    # no configuration of morphological features
                    if "morph" not in feat:
                        configs.append(feat.split("_")[-1])
                        
            # Remove feature configuration from feature name and put it into new column called "MIRP_config"
            checked_configs = []
            features_without_config_in_col = pd.DataFrame()
            
            # If we have different feature configurations
            if len(configs) > 0:
                
                configs = list(set(configs))
                print("Found {} different feature configurations: {}".format(str(len(configs)), str(configs)))

                nan_feature_ = list(set(nan_feature_))
                # print("Found {} different features with NaN: {}".format(str(len(nan_feature_)), str(nan_feature_)))
                
                for config in tqdm(configs, desc="Configure optimized features", unit="features"):
                    
                    if config not in checked_configs:
                        config_features = []
                        
                        # get all features with config in the ending
                        for feat in nan_feature_:
                            if feat.endswith(config):
                                config_features.append(feat)
                        
                        # get features with same configuration
                        feat_df = features[config_features]
                        
                        # drop all samples which do not have this configuration
                        feat_df = feat_df.dropna(axis = 0, how = 'all')
                        
                        feat_df["MIRP_config"] = config
                        checked_configs.append(config)

                        
                        feat_df_=feat_df.copy()
                        
                        for col in feat_df.columns:
                            if col.endswith(config):
                                # remove total configuration from the end of the feature name
                                if "_fb" in col:
                                    ext_config = re.search("_fb([_0-9a-zA-z.]*)",col)[1]
                                    new_col = col.removesuffix(ext_config)
                                    
                                else:
                                    new_col = col.removesuffix("_" + str(config))
                                    
                                feat_df_ = feat_df_.rename(columns={col:new_col})
                                configed_cols.append(col)
                        
                        feat_df = feat_df_.copy()
                        del feat_df_
                        
                        self.feat_df = feat_df.copy()
                        
                        if len(features_without_config_in_col) == 0:
                            features_without_config_in_col = feat_df
                        else:
                            features_without_config_in_col = pd.concat([features_without_config_in_col, feat_df])

                print("Configured {} Features".format(len(configed_cols)))
                
                # check if config specific parameter are in feature space
                conf_feat_ = []
                for conf_feat in config_specific_param:
                    for feat in features_without_config_in_col.columns:
                        if conf_feat in feat:
                            conf_feat_.append(conf_feat)

                config_specific_param = conf_feat_
                del conf_feat_
                
                if features_without_config_in_col[config_specific_param].isnull().values.any():
                    self.logger.info("Need to drop config specific features")
                    print("Need to drop config specific features")

                    for nan_feat in config_specific_param:
                        for col in  features_without_config_in_col.columns:
                            if nan_feat in col:
                                if features_without_config_in_col[col].isnull().any():
                                    print("Need to drop feature " + str(col))
                                    self.logger.info("Need to drop feature " + str(col))
                                    features_without_config_in_col.drop(col, axis=1, inplace=True)

                self.features_without_config_in_col = features_without_config_in_col
                nan_features = features_without_config_in_col.columns[features_without_config_in_col.isna().any()].tolist()
                if len(nan_features) > 0:
                    # check if configuration has been tracked correctly
                    
                    for nan_feature in nan_features:
                        if nan_feature not in columns_with_nan:
                            self.error.warning("Feature contains NaN! " + str(nan_feature))
                            print("Warning! Feature contains NaN! " + str(nan_feature))
                            #features_without_config_in_col.drop(nan_feature, axis=1, inplace=True)

                # check if sample size of before is the same
                if features_without_config_in_col.shape[0] != features.shape[0]:
                    self.error.warning("Feature configuration Failed! Something is wrong with the Feature configuration. Please check Extraction file.")
                    print("Feature configuration Failed! Something is wrong with the Feature configuration. Please check Extraction file.")
                    features_without_config_in_col = features

                    #features_without_config_in_col = pd.concat([features_without_config_in_col, features])
            else:
                features_without_config_in_col = features
            
            # add dataset configuration
            features_without_config_in_col = pd.concat([config_data, features_without_config_in_col], axis=1) 
            
            # check if features are not included in out features
            
            # features_without_config_in_col = pd.concat([config_data, features], axis=1) 
        else:
           features_without_config_in_col = features
           
           
        # if feature from before not in configed_cols, not in columns_with_nan and not in features_without_config_in_col --> missing feature
        missing_features = []
        for feature in features.columns:
            if feature not in configed_cols:
                if feature not in columns_with_nan:
                    if feature not in features_without_config_in_col.columns:
                        missing_features.append(feature)
                        
        print("Add {} non configurable features ...".format(str(len(missing_features))))

        features_without_config_in_col = pd.concat([features_without_config_in_col, features[missing_features]], axis=1)
        
        if features_without_config_in_col.index.name == "id_subject":
            if "id_subject" in features_without_config_in_col.columns:
                features_without_config_in_col.drop(['id_subject'], axis = 1, inplace = True) 
        
        print("MIRP Extraction Configuration done")
        return features_without_config_in_col

    # @profile(precision=4)
    def exe(self):
        """
        Execute Extractor for Radiomics feature extraction.
        """
        features = pd.DataFrame()

        # Get data configuration
        self.df = self.get_info_from_csv(path2csv=self.path2confCSV)
        
        
        # check if individual optimization of bin size is necessary
        # check if extreme values are included may cause Mem problems!
        if self.data_fingerprint['Number_of_bins'].max() > self.rptk_config["Feature_extraction_config"]["max_num_bin"]:
            # self.self_optimize = True
            self.get_bin_borders()
            if not self.self_optimize:
                self.error.warning(
                    "Recommend to set self_optimize to true as ROI seems to be very heterogeneous and can cause memory problems and performance drop.")
        # TODO: optimize cropping to safe memory
        self.rptk_config["Feature_extraction_config"]["cropping"] = False

        # cropping to the ROI
        if self.rptk_config["Feature_extraction_config"]["cropping"]:
            self.img_seg_cropping()

        if self.extractor == "MIRP":
            self.logger.info("### Starting MIRP Feature extraction ###\n")
            print("### Starting MIRP Feature extraction ###\n")
            features = self.exe_mirp()
        elif self.extractor == "PyRadiomics":
            self.logger.info("### Starting PyRadiomics extraction ###\n")
            print("### Starting PyRadiomics extraction ###\n")
            features = self.exe_pyradiomics()

        unnamed_cols = rptk.get_unnamed_cols(features)
        if len(unnamed_cols) > 0:
            self.logger.info("Unnamed columns found: " + str(unnamed_cols))
            features = features.drop(columns=unnamed_cols)

        # check for nan values in features
        failed = features[features.isnull()].copy()
        if failed.shape[0] > 0:
            self.error.warning("Failed extraction with NaN:" + str(len(failed)))
            
            # print("Failed extraction with NaN:" + str(len(failed['Mask'])))
            features = features[~features.isnull()]

        # check if index is unique else make new index
        if not features.index.is_unique:
            features = features.reset_index(drop=True)

        # if configuration not in features add it
        if "Image" not in features.columns:
            if features.index.name == self.df.index.name:
                if features.index in self.df.index:
                    if "Image" in self.df.columns:
                        features["Image"] = self.df.loc[features.index,"Image"]
                    if "Mask" in self.df.columns:
                        features["Mask"] = self.df.loc[features.index,"Mask"]
                    if "Image_Transformation" in self.df.columns:
                        features["Image_Transformation"] = self.df.loc[features.index,"Image_Transformation"]
                    if "Mask_Transformation" in self.df.columns:
                        features["Mask_Transformation"] = self.df.loc[features.index,"Mask_Transformation"]
                    if "Rater" in self.df.columns:
                        features["Rater"] = self.df.loc[features.index,"Rater"]
                    if "Modality" in self.df.columns:
                        features["Modality"] = self.df.loc[features.index,"Modality"]
                    if "Prediction_Label" in self.df.columns:
                        features["Prediction_Label"] = self.df.loc[features.index,"Prediction_Label"]
                    if "ROI_Label" in self.df.columns:
                        features["ROI_Label"] = self.df.loc[features.index,"ROI_Label"]
                    if "ID" in self.df.columns:
                        eatures["ID"] = self.df.loc[features.index,"ID"]
                else:
                    self.error.warning("Index from extraction is not included in preprocessing!")
            else:
                self.error.warning(f"Index {features.index.name} is not in preprocessing file!")
                    

        dropped = 0
        for i, row in features.iterrows():
            # if there is a row with names of the columns feature extraction has been appended with column names
            if str(row["Image"]) == "Image":
                features = features[features["Image"] != row["Image"]]
                continue

            if row['Image'] == "nan" or row['Mask'] == "nan":
                dropped += 1
                features.drop(i, inplace=True)
                continue

            if row["Image"] not in self.df["Image"].values:
                self.error.warning("Image " + str(row["Image"]) + " not found in config csv!")
                self.logger.info("Removing Image " + str(row["Image"]) + " from feature space!")
                dropped += len(features[features["Image"] == row["Image"]])
                features = features[features["Image"] != row["Image"]]

            if row["Mask"] not in self.df["Mask"].values:
                self.error.warning("Mask " + str(row["Mask"]) + " not found in config csv!")
                self.logger.info("Removing Mask " + str(row["Mask"]) + " from feature space!")
                features = features[features["Mask"] != row["Mask"]]

        if dropped > 0:
            self.logger.info("Dropped {} samples without Image/Mask path in config file!".format(str(dropped)))
            print("Dropped {} samples without Image/Mask path in config file!".format(str(dropped)))
            # overwrite results
            features.to_csv(self.outfile)

        if self.delta:
            print("Calculate Delta Radiomics ...")
            # save features wihtout delta calculation
            features.to_csv(self.extracted_features_dir + "tmp/" + str(self.extractor) + "_raw_extraction_" + str(self.RunID) + ".csv")

            self.features = features
            features = Extractor.get_delta_radiomics(features = self.features, 
                                                     extractor=self.extractor, 
                                                     time_format=self.time_format, 
                                                     out_path=self.out_path, 
                                                     logger=self.logger, 
                                                     error=self.error, 
                                                     take_label_changes=self.take_label_changes)

        # checking for duplicated samples
        features = features[~features.duplicated()]

        # features['Mask_Transformation'].replace(['NaN'], '')
        # features["Mask_Transformation"] = features["Mask_Transformation"].fillna("")

        # check for constant amsk correction otherwise remove it as it is just based on feature extraction adaption to the dataset
        if 'diagnostics_Mask-corrected' in features.columns:
            if features['diagnostics_Mask-corrected'].isnull():
                print("Not consitent Mask corrected features created!")
                # remove non informative columns absed on extractor configuration
                features = self.drop_columns_by_keyword(df=features, keyword="diagnostics_Mask-corrected")

        # drop duplicates
        features = features.drop_duplicates(subset=['Image', 'Mask'])

        # overwrite results
        features.to_csv(self.outfile)
        
        # check for IBSI coverage
        IBSIFeatureFormater(extractor=self.extractor,
                            features=features,
                            RunID=self.RunID,
                            logger=self.logger,
                            error=self.error,
                            output_path=self.out_path + "extracted_features/IBSI_profile/").format_features()
                        
        if self.self_optimize:
            if self.extractor == "MIRP":
                # features.to_csv(self.outfile[:-len(".csv")]+"_raw.csv")
                features.to_csv(self.extracted_features_tmp_dir + str(self.extractor) + "_extraction_without_mirp_config" + str(self.RunID) + ".csv")
                
                print(3*"#","Config MIRP Extraction",3*"#")
                features = self.adapt_optimized_feature_configuration(features)
                features.to_csv(self.outfile)

        self.failed_extractions.to_csv(self.out_path + "extracted_features/" + self.extractor + "_failed_extraction.csv")

        self.logger.info("### " + self.extractor + " Extraction Finished ###\n")
        print("### " + self.extractor + " Extraction Finished ###\n")

        return features

        #### TODO: Normalize Features before filtering

        # self.postprocessing()

        # TODO: calculate delta radiomics features before filtering
        # if self.delta:
        #    self.get_delta_radiomics()

        # self.logger.info("### Radiomics Feature Calculation Finished! ###")

        # filtered_radiomics_output = self.out_path + self.extractor + "_features_filtered.csv"
        # self.features.to_csv(filtered_radiomics_output)

        # self.logger.info("Filtered Radiomics Features in " + filtered_radiomics_output)
