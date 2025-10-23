import glob
import os
import sys

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np

import re
from math import isnan
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from detect_delimiter import detect
import datetime
import warnings
import multiprocessing
from multiprocessing import Pool
from pandas.api.types import is_string_dtype
from functools import partial
import math
from tqdm import tqdm

from rptk.src.segmentation_processing.SegProcessor import SegProcessor
from rptk.src.feature_filtering.Feature_stability_filter import FeatureStabilityFilter
from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.feature_filtering.Feature_formater import FeatureFormatter
import concurrent.futures
import traceback

# from rptk.src.feature_filtering.Radiomics_Filter_exe import RadiomicsFilter

# from rptk import rptk
# from rptk.rptk import RPTK

class DataConfigurator:
    """
    Configure the data for Filtering.
    1. Remove corrupted columns from extraction
    2. Separate Data for multiple ROIs or Time-points
    3. Add configuration of data to extracted features
    4. Add transformed Features as columns to effected sample and separate mask transformations
    5. Create unique sample identifier
    6. Drop Features which are not changing at all
    7. Extract non-numeric Features and save them in a separate file
    :param out_path: Path to output directory
    :param logger: Logger
    :param data: Dataframe with extracted features
    :param multiple_ROIs: Boolean if multiple ROIs are present
    :param longitudinal_data: Boolean if longitudinal data is present
    :param extractor: Name of the extractor
    :param path_to_img_seg_csv: Path to the csv file with the image and segmentation paths
    :param n_cpu: Number of CPUs to use
    :param RunID: ID of the run
    :param ICC_threshold: Threshold for ICC
    :param error: Error Logger
    :param separate_peritumoral_region: Boolean if peritumoral region should be separated or integrated as featue class
    """

    def __init__(self,
                 out_path: str = None,
                 logger: logging = None,
                 data: pd.DataFrame = pd.DataFrame(),
                 multiple_ROIs: bool = False,
                 longitudinal_data: bool = False,
                 extractor: str = "MIRP",
                 path_to_img_seg_csv: str = "",
                 n_cpu: int = 1,
                 RunID: str = "",
                 stability_filtering: bool = True,
                 ICC_threshold: float = 0.9,
                 error=None,
                 separate_peritumoral_region: bool = False,
                 format_check: bool = True,
                 peritumoral: bool = True,
                 additional_rois_to_features : bool = False,
                 delta: bool = False,
                 ):

        self.out_path = out_path
        self.logger = logger
        self.data = data
        self.multiple_ROIs = multiple_ROIs
        self.longitudinal_data = longitudinal_data
        self.extractor = extractor
        self.string_parameter = pd.DataFrame()
        self.string_parameter_msk = pd.DataFrame()
        self.path_to_img_seg_csv = path_to_img_seg_csv
        self.n_cpu = n_cpu
        self.RunID = RunID
        self.stability_filtering = stability_filtering
        self.ICC_threshold = ICC_threshold
        self.error = error
        self.separate_peritumoral_region = separate_peritumoral_region
        self.format_check = format_check
        self.peritumoral = peritumoral
        self.additional_rois_to_features = additional_rois_to_features
        self.delta = delta

        # self.feature_dir = os.path.join(self.out_path, self.extractor + "_features/")

        # Get string for unique selection of transformation kernels
        self.track_config = {"gabor": "gabor_",
                             "gaussian": "gauss_",
                             "wavelet-HHH": "wavelet-HHH",
                             "wavelet-LHH": "wavelet-LHH",
                             "wavelet-HLH": "wavelet-HLH",
                             "wavelet-HHL": "wavelet-HHL",
                             "wavelet-HLL": "wavelet-HLL",
                             "wavelet-LHL": "wavelet-LHL",
                             "wavelet-LLH": "wavelet-LLH",
                             "Wavelet-LLL": "wavelet-LLL",
                             "WaveletHHH": "wavelet-HHH",
                             "WaveletLHH": "wavelet-LHH",
                             "WaveletHLH": "wavelet-HLH",
                             "WaveletHHL": "wavelet-HHL",
                             "WaveletHLL": "wavelet-HLL",
                             "WaveletLHL": "wavelet-LHL",
                             "WaveletLLH": "wavelet-LLH",
                             "WaveletLLL": "wavelet-LLL",
                             "wavelet": "wavelet-",
                             "SquareRoot": "squareroot",
                             "squareroot": "squareroot",
                             "Square": "square",
                             "square": "square",
                             "Gradient": "gradient",
                             "gradient": "gradient",
                             "nonseparable_wavelet": "wavelet_",
                             "LBP2D": "lbp-2D",
                             "lbp-2D": "lbp-2D",
                             "LBP3D": "lbp-3D",
                             "lbp-3D": "lbp-2D",
                             "Logarithm": "logarithm",
                             "mean": "mean_",
                             "Exponential": "exponential",
                             "LoG": "log-",
                             "laws": "laws_",
                             "Peritumoral": "peritumoral_"
                             }
        # Configuration of the data without transformations
        self.config_features = ["Image", "Mask", "ID", "Rater", "ROI_Label", "Modality", "Prediction_Label", "Timepoint", "Raw_Image", "Raw_Mask"]
        self.additional_ROIs = []
        self.mask_peturbations = ["Random_Walker_Mask_Change", 
                                "Connected_Component_Expansion", 
                                "Super_Voxel_Randomization",
                                "Peritumoral"]

        self.samples_not_incuded_in_all_transformations = []

        if not self.out_path is None:
            if not self.out_path.endswith("/"):
                self.out_path = self.out_path + "/"
        else:
            print("No out_path defined for Data Configuration. Please provide path to output folder.")
            self.error.error("No out_path defined for Data Configuration. Please provide path to output folder.")
            raise ValueError("No out_path defined for Data Configuration. Please provide path to output folder.")

        if not os.path.exists(self.out_path + "tmp/"):
            os.makedirs(self.out_path + "tmp/")

        # Config Logger #
        # if self.logger is None:
        self.logger = LogGenerator(
            log_file_name=self.out_path + "RPTK_data_configuration_" + self.RunID + ".log",
            logger_topic="RPTK Data Configuration"
        ).generate_log()

        # if self.error is None:
        self.error = LogGenerator(
            log_file_name=self.out_path + "RPTK_data_configuration_" + self.RunID + ".err",
            logger_topic="RPTK Data Configuration error"
        ).generate_log()

        if len(self.data) == 0:
            self.error.error("No input data provided! Please provide a pd.DataFrame as data for processing.")
            raise ValueError("No input data provided! Please provide a pd.DataFrame as data for processing.")

        # drop non-relevant parameter
        #if "Raw_Image" in self.data.columns:
        #    self.data.drop(columns=["Raw_Image"], inplace=True)

        #if "Raw_Mask" in self.data.columns:
        #    self.data.drop(columns=["Raw_Mask"], inplace=True)

    def get_correct_transformations(self,df):
        i = df[0]
        row = df[1]
        if pd.isnull(row["Image_Transformation"]):
            found = False
            # check if transformations is in the image
            for file_kernel in self.track_config.values():
                if file_kernel in os.path.basename(row["Image"]):
                    found = True
                    break
                
            if found:
                # 2. get correct kernel config from image file
                r = re.search(file_kernel, os.path.basename(row["Image"]))
                if not r is None:
                    kernel = os.path.basename(row["Image"])[r.start():]
                    kernel = kernel.replace("_resampled", "")
                    kernel = kernel.replace(".nii.gz", "")

                    self.data.loc[(self.data.index == i) & (row["Image"] == self.data["Image"]), "Image_Transformation"] = kernel
                else:
                    raise ValueError("Could not identify transformation config for {}.".format(os.path.basename(row["Image"])))
            else:
                print("Could not identify transformation config for {}.".format(os.path.basename(row["Image"])))
        else:
            # change image transformation to transforamtion how it occurs in file name add configuration of transformation
            if isinstance(row["Image_Transformation"], str):
                if row["Image_Transformation"] in self.track_config:
                    file_match = self.track_config[row["Image_Transformation"]]

                    if file_match is not None:
                        r = re.search(file_match, os.path.basename(row["Image"]))
                        if not r is None:
                            kernel = os.path.basename(row["Image"])[r.start():]
                            kernel = kernel.replace("_resampled", "")
                            kernel = kernel.replace(".nii.gz", "")
                        else:
                            self.error.warning("{} is assigned to wrong image {}. Correcting ...".format(str(row["Image_Transformation"]), os.path.basename(row["Image"])))
                            
                            # 1. find correct kernel from image file
                            for file_kernel in self.track_config.values():
                                if file_kernel in os.path.basename(row["Image"]):
                                    break

                            # 2. get correct kernel config from image file
                            r = re.search(file_kernel, os.path.basename(row["Image"]))
                            if not r is None:
                                kernel = os.path.basename(row["Image"])[r.start():]
                                kernel = kernel.replace("_resampled", "")
                                kernel = kernel.replace(".nii.gz", "") 

                            else:
                                kernel = np.nan
                                # self.data.loc[self.data.index == i, "Image_Transformation"] = np.nan
                                print("Image Transformation configuration is wrong! {} Could not find in file {} Review Radiomics feature extraction file!".format(file_match, os.path.basename(row["Image"])))
                                self.error.warning("Image Transformation configuration is wrong! {} Could not find in file {} Review Radiomics feature extraction file!".format(file_match, os.path.basename(row["Image"])))
                        
                        self.data.loc[(self.data.index == i) & (row["Image"] == self.data["Image"]), "Image_Transformation"] = kernel
                    else:
                        self.error.warning("Could not identify transformation config for {} with transformation {}.".format(os.path.basename(row["Image"]),
                                                                                                                row["Image_Transformation"]))
                        raise ValueError("Could not identify transformation config for {} with transformation {}.".format(os.path.basename(row["Image"]),
                                                                                                                                row["Image_Transformation"]))
                else:
                    # kernel with transformation config in column
                    found = False
                    for kernel in self.track_config:
                        if kernel in row["Image_Transformation"]:
                            if self.track_config[kernel] in os.path.basename(row["Image"]):
                                found = True
                                file_kernel = self.track_config[kernel]
                                break

                    if not found: 
                        for file_kernel in self.track_config.values():
                            if file_kernel in os.path.basename(row["Image"]):
                                found = True
                                break
                    if found:
                        # 2. get correct kernel config from image file
                        r = re.search(file_kernel, os.path.basename(row["Image"]))
                        if not r is None:
                            kernel = os.path.basename(row["Image"])[r.start():]
                            kernel = kernel.replace("_resampled", "")
                            kernel = kernel.replace(".nii.gz", "") 
                        else:
                            kernel = np.nan
                        
                        self.data.loc[(self.data.index == i) & (row["Image"] == self.data["Image"]), "Image_Transformation"] = kernel

                    else:
                        print("Could not get {} in Image {} or in config. Check feature extraction file!".format(row["Image_Transformation"], os.path.basename(row["Image"])))

                        if not str(row["Image_Transformation"]) in str(os.path.basename(row["Image"])):
                            print(str(row["Image_Transformation"]), str(os.path.basename(row["Image"])))
                            print("Could not find {} transformation in allowed transformation. Add this transforamtion to config!".format(str(row["Image_Transformation"])))                                                                                               
                            self.error.warning("Could not find {} transformation in allowed transformation. Add this transforamtion to config!".format(str(row["Image_Transformation"])))

    def remove_duplcated_samples_without_preprocessed_seg(self, df, img_trans, mask_trans):
        """
        Removing samples which are not preprocessed but in data. E.g. samples in folder input_reformated or converted
        """
        if df.index.name != "ID":
            if "ID" in df.columns:
                df = df.set_index("ID")
            else:
                raise ValueError("No ID found in data. Please check input data!")
        
        if "Image_Transformation" in df.columns:
            if img_trans == "nan":
                cond1 = df.copy()["Image_Transformation"].isnull()
            else:
                cond1 = df.copy()["Image_Transformation"] == img_trans

        else:
            cond1 = None

        if "Mask_Transformation" in df.columns:    
            if mask_trans == "nan":
                cond2 = df.copy()["Mask_Transformation"].isnull()
            else:
                cond2 = df.copy()["Mask_Transformation"] == mask_trans
        else:
            cond2 = None

        # raw = df.copy()[(df.copy()["Image_Transformation"]==img_trans) & (df.copy()["Mask_Transformation"]==mask_trans)]
        if (not cond1 is None) and (not cond2 is None):
            raw = df.copy()[(cond1) & (cond2)]
        else:
            raw = df.copy()
            
        if len(raw[raw.index.duplicated()]) > 0:
            df = df.reset_index()
            if "Image_Transformation" in df.columns:
                if img_trans == "nan":
                    cond1 = df.copy()["Image_Transformation"].isnull()
                else:
                    cond1 = df.copy()["Image_Transformation"] == img_trans
            else:
                cond1 = None
            
            if "Mask_Transformation" in df.columns:  
                if mask_trans == "nan":
                    cond2 = df.copy()["Mask_Transformation"].isnull()
                else:
                    cond2 = df.copy()["Mask_Transformation"] == mask_trans
            else:
                cond2 = None

            if (not cond1 is None) and (not cond2 is None):    
                raw = df.copy()[(cond1) & (cond2)]
            else:
                raw = df.copy()

            if "Mask" in df.columns:
                # dop samples not containing multilabel_seg
                drop = raw.copy()[raw.copy()["Mask"].str.contains('/input_reformated/', na=False)]
                df = df.copy().drop(drop.index)
                drop = raw.copy()[raw.copy()["Mask"].str.contains('/converted/', na=False)]
                df = df.copy().drop(drop.index)
            
            df = df.set_index("ID")

        del raw

        return df

    # 1. Configure the columns of the raw feature extractions
    def configure_columns(self):
        """
        Configure the columns of the raw feature extractions by removing useless columns
        :return: self.data as df with dropped columns
        """

        self.logger.info("### 1. Configure Columns ###")
        print("### 1. Configure Columns ###")

        if self.data.index.name == "ID":
            self.data["ID"] = self.data.index

        if 'Image_Transformation' in self.data.columns:
            # check if nan has been replaced by 0
            if 0 in self.data['Image_Transformation']:
                self.logger.info("Found 0 in Image_Transformation column. Replace 0 with nan")
                self.data["Image_Transformation"] = self.data["Image_Transformation"].replace(0, np.nan)

        if "Mask_Transformation" in self.data.columns:
            if 0 in self.data['Mask_Transformation']:
                self.logger.info("Found 0 in Mask_Transformation column. Replace 0 with nan")
                self.data["Mask_Transformation"] = self.data["Mask_Transformation"].replace(0, np.nan)

        self.logger.info("Dropping unnamed, duplicated and constant features ...")
        print("Dropping unnamed, duplicated and constant features ...")

        # drop columns which are useless
        columns2drop = ["Unnamed:", ":"]  # TODO: Check why we need this and if we are delete too much features
        dropped = 0
        dropped_cols = []

        for col in columns2drop:
            for dcol in self.data.columns:
                if col in dcol:
                    dropped += 1
                    dropped_cols.append(dcol)
                    self.data = self.data.drop(dcol, axis=1)
        
        if "Rater" in self.data.columns:
            # because Mask Transformation and Rater can have the same values if there is not other rater
            if self.data['Mask_Transformation'].equals(self.data['Rater']):

                mask_transformation = self.data['Mask_Transformation']
                rater = self.data['Rater']

                # drop duplicated columns (values) means dropping rater/mask_transformation
                self.data = self.data.T.drop_duplicates().T

                # repair the columns
                if "Mask_Transformation" not in self.data.columns:
                    self.data["Mask_Transformation"] = mask_transformation

                if "Rater" not in self.data.columns:
                    self.data["Rater"] = rater
            else:
                # drop duplicated columns (values)
                self.data = self.data.T.drop_duplicates().T
        else:
            # drop duplicated columns (values)
            self.data = self.data.T.drop_duplicates().T

        # drop duplicated columns (names)
        dropped += len(self.data.columns[self.data.columns.duplicated()].tolist())
        dropped_cols += self.data.columns[self.data.columns.duplicated()].tolist()
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        if dropped > 0:
            self.logger.info("Dropped {} features".format(dropped))
            self.logger.info("Dropped features: {}".format(dropped_cols))
        

        self.logger.info("Correct Transformation names ...")
        print("Correct Transformation names ...")

        print("Samples", self.data.shape[0])
        print("Features", self.data.shape[1])

        if "Image_Transformation" in self.data.columns:
            # correct transformation names if necessary
            for kernel in set(self.data.copy()["Image_Transformation"].to_list()):
                if "_resampled" in str(kernel):
                    corr_kern = kernel.replace("_resampled", "")
                    self.data.loc[self.data["Image_Transformation"] == kernel, "Image_Transformation"] = corr_kern

        if "Image" in self.data.columns:
        # different format does not match conversion
            if "cropped" not in self.data["Image"].iloc[0]:
                # add config of kernel to kernel name
                if "Image_Transformation" not in self.data.columns:
                    self.error.warning("Image_Transformation column is missing!")
                    # raise ValueError("Image_Transformation column is missing!. Review Radiomics feature extraction file!")

                # Adding transformation config to Image transformation column
                #try:
                #    with tqdm(total=len(self.data), desc="Config Image Transformations") as pbar:
                #        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                #            for results in executor.map(self.get_correct_transformations, [(i, row) for i, row in self.data.copy().iterrows()], chunksize=self.n_cpu):
                #                pbar.update(1)
                                

                #except Exception as ex:
                #    self.error.error("Config Image Transformation Failed! " + str(ex))
                #    self.error.error(traceback.format_exc())
                #    raise Exception("Config Image Transformation Failed! " + str(ex))

                if not os.path.exists(self.out_path + "tmp/config_image_transformation.csv"):
                    if "Image_Transformation" in self.data.columns:
                        for i, row in tqdm(self.data.copy().iterrows(), total=len(self.data), desc="Config Image Transformations") :
                            if pd.isnull(row["Image_Transformation"]):
                                found = False
                                # check if transformations is in the image
                                for file_kernel in self.track_config.values():
                                    if file_kernel in os.path.basename(row["Image"]):
                                        found = True
                                        break
                                    
                                if found:
                                    # 2. get correct kernel config from image file
                                    r = re.search(file_kernel, os.path.basename(row["Image"]))
                                    if not r is None:
                                        kernel = os.path.basename(row["Image"])[r.start():]
                                        kernel = kernel.replace("_resampled", "")
                                        kernel = kernel.replace(".nii.gz", "")

                                        self.data.loc[(self.data.index == i) & (row["Image"] == self.data["Image"]), "Image_Transformation"] = kernel
                                    else:
                                        raise ValueError("Could not identify transformation config for {}.".format(os.path.basename(row["Image"])))
                                #else:
                                #    self.error.warning("Could not find transformation for " + str(os.path.basename(row["Image"])))
                                #    print("Could not find transformation for " + str(os.path.basename(row["Image"])))
                            else:
                                # change image transformation to transformation how it occurs in file name add configuration of transformation
                                if isinstance(row["Image_Transformation"], str):
                                    if row["Image_Transformation"] in self.track_config:
                                        file_match = self.track_config[row["Image_Transformation"]]

                                        if file_match is not None:
                                            r = re.search(file_match, os.path.basename(row["Image"]))
                                            if not r is None:
                                                kernel = os.path.basename(row["Image"])[r.start():]
                                                kernel = kernel.replace("_resampled", "")
                                                kernel = kernel.replace(".nii.gz", "")
                                            else:
                                                self.error.warning("{} is assigned to wrong image {}. Correcting ...".format(str(row["Image_Transformation"]), os.path.basename(row["Image"])))
                                                
                                                # 1. find correct kernel from image file
                                                for file_kernel in self.track_config.values():
                                                    if file_kernel in os.path.basename(row["Image"]):
                                                        break

                                                # 2. get correct kernel config from image file
                                                r = re.search(file_kernel, os.path.basename(row["Image"]))
                                                if not r is None:
                                                    kernel = os.path.basename(row["Image"])[r.start():]
                                                    kernel = kernel.replace("_resampled", "")
                                                    kernel = kernel.replace(".nii.gz", "") 

                                                else:
                                                    kernel = np.nan
                                                    # self.data.loc[self.data.index == i, "Image_Transformation"] = np.nan
                                                    print("Image Transformation configuration is wrong! {} Could not find in file {} Review Radiomics feature extraction file!".format(file_match, os.path.basename(row["Image"])))
                                                    self.error.warning("Image Transformation configuration is wrong! {} Could not find in file {} Review Radiomics feature extraction file!".format(file_match, os.path.basename(row["Image"])))
                                            
                                            self.data.loc[(self.data.index == i) & (row["Image"] == self.data["Image"]), "Image_Transformation"] = kernel
                                        else:
                                            self.error.warning("Could not identify transformation config for {} with transformation {}.".format(os.path.basename(row["Image"]),
                                                                                                                                    row["Image_Transformation"]))
                                            raise ValueError("Could not identify transformation config for {} with transformation {}.".format(os.path.basename(row["Image"]),
                                                                                                                                                    row["Image_Transformation"]))
                                    else:
                                        # kernel with transformation config in column
                                        found = False
                                        for kernel in self.track_config:
                                            if kernel in row["Image_Transformation"]:
                                                if self.track_config[kernel] in os.path.basename(row["Image"]):
                                                    found = True
                                                    file_kernel = self.track_config[kernel]
                                                    break
                                            elif self.track_config[kernel] in row["Image_Transformation"]:
                                                if self.track_config[kernel] in os.path.basename(row["Image"]):
                                                    found = True
                                                    file_kernel = self.track_config[kernel]
                                                    break

                                        if not found: 
                                            for file_kernel in self.track_config.values():
                                                if file_kernel in os.path.basename(row["Image"]):
                                                    found = True
                                                    break
                                        if found:
                                            # 2. get correct kernel config from image file
                                            r = re.search(file_kernel, os.path.basename(row["Image"]))
                                            if not r is None:
                                                kernel = os.path.basename(row["Image"])[r.start():]
                                                kernel = kernel.replace("_resampled", "")
                                                kernel = kernel.replace(".nii.gz", "") 
                                            else:
                                                kernel = np.nan
                                            
                                            self.data.loc[(self.data.index == i) & (row["Image"] == self.data["Image"]), "Image_Transformation"] = kernel

                                        else:
                                            print("Could not get {} in Image {} or in config. Check feature extraction file!".format(row["Image_Transformation"], os.path.basename(row["Image"])))

                                            if not str(row["Image_Transformation"]) in str(os.path.basename(row["Image"])):
                                                print(str(row["Image_Transformation"]), str(os.path.basename(row["Image"])))
                                                print("Could not find {} transformation in allowed transformation. Add this transforamtion to config!".format(str(row["Image_Transformation"])))                                                                                               
                                                self.error.warning("Could not find {} transformation in allowed transformation. Add this transforamtion to config!".format(str(row["Image_Transformation"])))
                                else:
                                    raise ValueError("Image Transformation configuration wrong! Please check feature data." + row["Image_Transformation"])
                    self.data.to_csv(self.out_path + "tmp/config_image_transformation.csv")
                else:
                    print("Loading found processed image transforamtion config: " + self.out_path + "tmp/config_image_transformation.csv")
                    self.data = pd.read_csv(self.out_path + "tmp/config_image_transformation.csv", index_col = 0)

        # report features with nan values
        if self.data.isnull().any().any():
            self.error.warning("Number of Features with nan values: " + str(self.data.isnull().any().sum()))
            self.error.warning("Features with nan values: " + str(self.data.loc[:, self.data.isnull().any()].columns.to_list()))
            print("Features with nan values: " + str(self.data.isnull().any().sum()))
            
            if "Image" in self.data.columns:
                if self.data["Image"].isnull().any() or self.data["Mask"].isnull().any():
                    self.error.warning("{} Failed extraction. Dropping ...".format(self.data["Image"].isnull().any().sum()))
                    print("{} Failed extraction. Dropping ...".format(self.data["Image"].isnull().any().sum()))
                    self.data = self.data[self.data['Image'].notna()]
                    self.data = self.data[self.data['Mask'].notna()]
        if self.format_check:
            self.data = SegProcessor(logger=self.logger,
                                    error=self.error,
                                    RunID=self.RunID,
                                    out_path=os.path.dirname(self.out_path)
                                    ).check_format(df=self.data.copy())

    # 2.
    def config_ROI_and_Timepoint(self):
        """
        Configure the ROI and Timepoint column for longitudinal and multiple ROIs data
        :return: self.data as df with multiple rows for every ROI or Time-point
        """

        self.logger.info("### 2. Configure ROI and Timepoint ###")
        print("### 2. Configure ROI and Timepoint ###")

        if not (self.longitudinal_data or self.multiple_ROIs):
            self.logger.info("No multiple ROIs or longitudinal data processing enabled!")
        else:
            self.logger.info("Checking for ROI ...")
            # if the data is longitudinal or has multiple ROIs per sample in the same image
            if self.multiple_ROIs:
                if ("ROI" not in self.data.columns) or (self.data["ROI"].isnull().values.all()):
                    self.data["ROI"] = np.nan

            self.logger.info("Checking for Timepoint ...")
            if self.longitudinal_data:
                if ("Timepoint" not in self.data.columns) or (self.data["Timepoint"].isnull().values.all()):
                    self.data["Timepoint"] = np.nan

            if self.longitudinal_data or self.multiple_ROIs:
                for index, row in self.data.iterrows():
                    if self.multiple_ROIs:
                        if row["Mask_Transformation"].endswith("Peritumoral"):
                            self.data.loc[index, "ROI"] = 0  # "Peritumoral"
                        else:
                            self.data.loc[index, "ROI"] = 1  # "Intratumoral"

                    if self.longitudinal_data:
                        if index.startswith("BL"):
                            self.data.loc[index, "Timepoint"] = 0
                        elif index.startswith("FU_1"):
                            self.data.loc[index, "Timepoint"] = 1
                        elif index.startswith("FU_2"):
                            self.data.loc[index, "Timepoint"] = 2
                        elif index.startswith("FU_3"):
                            self.data.loc[index, "Timepoint"] = 3
                        elif index.startswith("FU_4"):
                            self.data.loc[index, "Timepoint"] = 4
                        elif index.startswith("FU_5"):
                            self.data.loc[index, "Timepoint"] = 5

        self.logger.info("Checking for missing values in data configuration ...")
        print("Checking for missing values in data configuration ...")
        print(self.data.shape)
        # if config columns are not present fill NaNs
        if "ID" not in self.data.columns:
            self.data["ID"] = np.nan
        if "Modality" not in self.data.columns:
            self.data["Modality"] = np.nan
        if "ROI_Label" not in self.data.columns:
            self.data["ROI_Label"] = np.nan
        if "Timepoint" not in self.data.columns:
            self.data["Timepoint"] = np.nan
        if "Prediction_Label" not in self.data.columns:
            self.data["Prediction_Label"] = np.nan
        if "Image" not in self.data.columns:
            self.data["Image"] = np.nan
        if "Mask" not in self.data.columns:
            self.data["Mask"] = np.nan
        if "Mask_Transformation" not in self.data.columns:
            self.data["Mask_Transformation"] = np.nan
        if "Image_Transformation" not in self.data.columns:
            self.data["Image_Transformation"] = np.nan
        if "Rater" not in self.data.columns:
            self.data["Rater"] = np.nan

    # 3.
    def add_config_to_df(self):
        """
        Add configuration information to the data frame from configuration file
        we need to configure the data accordingly.
        """

        self.logger.info("### 3. Add Configuration to Extracted Features ###")
        print("### 3. Add Configuration to Extracted Features ###")

        if self.extractor == "PyRadiomics":
            image=None
            mask=None

            # TODO: Currently only ending .nii.gz !!
            if "Image" in self.data.columns:
                image = self.data["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
            else:
                self.error.warning("Image not found in data!")
            
            if "Mask" in self.data.columns:
                mask = self.data["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
            else:
                self.error.warning("Mask not found in data!")
            
            if (not image is None) and (not mask is None):
                # generate Index with Image name and Mask name for config string
                id = []
                for x, y in zip(image, mask):
                    id.append(x + "_" + y)

                ID = pd.Series(index=self.data.index, data=id)
                self.data.index = ID
                self.data.index.rename("id_subject", inplace=True)
                self.string_parameter.index = ID
                self.string_parameter.index.rename("id_subject", inplace=True)

            # self.data, self.msk_trans = self.extend_data_for_transformations()

        # if self.extractor == "MIRP":
            # drop duplicates if there are samples processed multiple times with the same setting
            # self.data = self.data.drop_duplicates(subset=['id_subject', 'img_data_settings_id'],
            #                                      keep='first').reset_index(drop=True)

            # id_subject example: Melanoma-001_img_0_0_Melanoma-001_seg_0_0_0
            # self.data["id_subject"] = self.data["id_subject"].str.replace(r'_img', '')
            # print(self.data.loc(self.data["id_subject"].str.endswith('_img')))#.str[:-4]

            # fill dataframe with config data
            img_seg_df = pd.read_csv(self.path_to_img_seg_csv)
            image = img_seg_df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
            mask = img_seg_df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

            # generate Index with Image name and Mask name for config string
            id = []
            for x, y in zip(image, mask):
                id.append(x + "_" + y)

            ID = pd.Series(index=img_seg_df.index, data=id)
            img_seg_df.set_index(ID, inplace=True)

            for col in self.data.columns:
                # if "id_subject" is included in column name
                if "id_subject" in col:
                    self.data.rename(columns={col: "id_subject"}, inplace=True)

            needed_columns = ["ID", "Modality", "ROI_Label", "Timepoint", "Prediction_Label",
                            "Image", "Mask", "Mask_Transformation", "Image_Transformation", "Rater"]

        #if "id_subject" in self.data.columns:
        #    self.data.set_index("id_subject", inplace=True)
        #    self.string_parameter.index = self.data.index

            for col in needed_columns:
                if col not in self.data.columns:
                    self.data[col] = np.nan
                if col not in img_seg_df.columns:
                    img_seg_df[col] = np.nan

            check_needed = False
            # check if config columns are not in data
            for config in self.config_features:
                if config not in self.data.columns:
                    if config in img_seg_df.columns:
                        if img_seg_df[config].isnull().all():
                            continue
                        else:
                            check_needed = True
                            break

            if check_needed:
                for i, row in img_seg_df.iterrows():
                    id = i #os.path.basename(row["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row["Mask"])[
                                                                                #:-len(".nii.gz")]
                    # print(id, self.data.index)
                    if "id_subject" in self.data.columns:
                        # print(id, self.data.loc[self.data["id_subject"] == id, "id_subject"])
                        self.data.loc[self.data["id_subject"] == id, "ID"] = row["ID"]
                        self.data.loc[self.data["id_subject"] == id, "Modality"] = row["Modality"]
                        self.data.loc[self.data["id_subject"] == id, "ROI_Label"] = row["ROI_Label"]
                        self.data.loc[self.data["id_subject"] == id, "Timepoint"] = row["Timepoint"]
                        self.data.loc[self.data["id_subject"] == id, "Prediction_Label"] = row["Prediction_Label"]
                        self.data.loc[self.data["id_subject"] == id, "Image"] = row["Image"]
                        self.data.loc[self.data["id_subject"] == id, "Mask"] = row["Mask"]
                        self.data.loc[self.data["id_subject"] == id, "Mask_Transformation"] = row["Mask_Transformation"]
                        # self.data.loc[self.data["id_subject"] == id, "Image_Transformation"] = row["Image_Transformation"]
                        self.data.loc[self.data["id_subject"] == id, "Rater"] = row["Rater"]

                        if "Raw_Image" in self.data.columns:
                            self.data.loc[self.data["id_subject"] == id, "Raw_Image"] = row["Raw_Image"]
                            self.data.loc[self.data["id_subject"] == id, "Raw_Mask"] = row["Raw_Mask"]

                    # MAYBE Change causes error?
                    elif self.data.index.name == "id_subject":
                        self.data.loc[self.data.index == id, "ID"] = row["ID"]
                        self.data.loc[self.data.index == id, "Modality"] = row["Modality"]
                        self.data.loc[self.data.index == id, "ROI_Label"] = row["ROI_Label"]
                        self.data.loc[self.data.index == id, "Timepoint"] = row["Timepoint"]
                        self.data.loc[self.data.index == id, "Prediction_Label"] = row["Prediction_Label"]
                        self.data.loc[self.data.index == id, "Image"] = row["Image"]
                        self.data.loc[self.data.index == id, "Mask"] = row["Mask"]
                        self.data.loc[self.data.index == id, "Mask_Transformation"] = row["Mask_Transformation"]
                        # self.data.loc[self.data.index == id, "Image_Transformation"] = row["Image_Transformation"]
                        self.data.loc[self.data.index == id, "Rater"] = row["Rater"]

                        if "Raw_Image" in self.data.columns:
                            self.data.loc[self.data.index == id, "Raw_Image"] = row["Raw_Image"]
                            self.data.loc[self.data.index == id, "Raw_Mask"] = row["Raw_Mask"]

            if "img_data_roi" in self.data.columns:
                # self.data["img_data_modality"] = img_seg_df["img_data_modality"]
                self.data.index = self.data["img_data_roi"]
                # self.data.index = self.data["id_subject"]
                self.string_parameter.index = self.data["img_data_roi"]

            #elif self.extractor == "PyRadiomics":
            #    for img,seg in zip(self.data["Image"].values,self.data["Mask"].values):
            #        self.data.loc[self.data["Image"] == img and self.data["Mask"] == seg, "img_data_roi"] = os.path.basename(img)[:-len(".nii.gz")] + "_" + os.path.basename(seg)[:-len(".nii.gz")]

            for i in self.string_parameter.index:
                if i not in self.data.index:
                    self.error.error("Index of non relevant parameter not in data: " + str(i))
                    break

    # 4.2
    def separate_transformations(self, Transformation, df: pd.DataFrame):
        """
        Seperate the mask transformed samples from the data
        :param df: Dataframe containing all samples with features and transformations
        :return: 
        df_wo_transform: Dataframe containing all samples without transformations,
        df_transform: Dataframe containing all samples with transformations
        """

        print("Sepperate", Transformation)
        self.logger.info("Sepperate " + Transformation + " ...")
        self.logger.info("Total size of data before transformation separation: " + str(df.shape))
        print("Total size of data before transformation separation: " + str(df.shape))


        # Separate Transformations
        df_transform = df.copy().loc[~df[Transformation].isnull(), :]
        df_wo_transform = df.copy().loc[df[Transformation].isnull(), :]

        self.logger.info("Total size of data after transformation sepperation: " + str(df_wo_transform.shape))
        print("Total size of data after transformation sepperation: " + str(df_wo_transform.shape))

        if df_transform.shape[0] == 0:
            self.error.warning("No samples with " + Transformation + " detected!")
            print("Warning: No samples with " + Transformation + " detected!")
        else:
            self.logger.info("Detected " + str(df_transform.shape[0]) + " samples with " + Transformation)
            print("Detected " + str(df_transform.shape[0]) + " samples with " + Transformation)

        if not df_wo_transform.index.name == df_transform.index.name:
            self.error.error("Index names of dataframes do not match!")
            raise Exception("When separating the transformations ... Index names of dataframes do not match!")

        return df_wo_transform, df_transform

    # 3.1
    def set_unique_index(self, df: pd.DataFrame):
        """
        Set the index of the dataframe to a unique index
        :param df: Dataframe containing all samples with features and transformations
        :return: df: Dataframe containing all samples with features and transformations
        """

        print("3.1 set_unique_index")
        self.logger.info("3.1 set_unique_index")
        
        if df.index.name != "id_subject":
            # set index
            ids = []
            for i, row in df.iterrows():
                ids.append(os.path.basename(row["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row["Mask"])[
                                                                                    :-len(".nii.gz")])
            df.index = ids

        return df

    def configure_peritumoral_features(self, df: pd.DataFrame):
        """
        Configure the peritumoral features
        :param df: Dataframe containing all samples with features and transformations
        :return: df: Dataframe containing all samples with features and transformations
        """

        # Configure peritumoral features
        if self.peritumoral_features:
            self.logger.info("### 4.2 Configuring peritumoral features ###")
            #df = self.configure_peritumoral_features(df)

        return df

    def get_peritumoral_samples(self, df: pd.DataFrame):
        """
        Get Features with peritumoral mask transformation and remove them from df
        :parameter df: Dataframe containing all samples with features and transformations
        """
        df['Mask_Transformation'] = df['Mask_Transformation'].replace(np.nan, "", regex=True)
        # get all samples with peritumoral in column "Mask_Transformation"
        df_peritumoral = df.loc[df['Mask_Transformation'].str.contains("Peritumoral", regex=True)]
        df_not_peritumoral = df.loc[~df['Mask_Transformation'].str.contains("Peritumoral", regex=True)]

        # df_peritumoral = df.loc[df["Mask_Transformation"] == "Peritumoral", :]
        # df_not_peritumoral = df.loc[df["Mask_Transformation"] != "Peritumoral", :]
        df_not_peritumoral['Mask_Transformation'] = df_not_peritumoral['Mask_Transformation'].replace(r'^\s*$', np.nan, regex=True)

        return df_peritumoral, df_not_peritumoral

    def sepperate_config_features(self, config_features: list, df: pd.DataFrame):
        """"
        Sepperate configuration features from Radiomics features
        """
        # df.to_csv(self.out_path + "separate_config_features.csv")

        included_config = []

        # ig config feature not in df then remove it from the list of config features
        for conf in config_features:
            if conf in df.columns:
                included_config.append(conf)

        config_features = included_config

        if "Raw_Mask" in df.columns:
            # if there are missing values in the raw Image/Mask replace with Image/Mask
            if "Raw_Mask" in config_features:
                df["Raw_Mask"].fillna(df["Mask"], inplace=True)

        if "Raw_Image" in df.columns:
            if "Raw_Image" in config_features:
                df["Raw_Image"].fillna(df["Image"], inplace=True)

        if len(df) > 0:
            config_df = df[config_features].copy()
            df_wo_config = df.drop(columns=config_features)
        else:
            config_df = pd.DataFrame()
            df_wo_config = pd.DataFrame()

        return config_df, df_wo_config

    def normalize(self, df: pd.DataFrame):
        """
        Perform Z-score normalization for each feature (column) in the dataframe and rename it with adding the suffix _zscore
        :param df: Radiomics features without any sting features
        :return: normalized features
        """

        normalized = False

        # self.logger.info("Normalizing Data ...")
        for col in df.columns:
            if "_zscore" in col:
                normalized = True
            if df[col].isnull().values.any():
                self.error.warning("NaN in Feature: " + col)

        if not normalized:
            df, string_df = self.search_non_numeric_features(df=df.copy())
            if not df.index.name == "img_data_roi":
                if "img_data_roi" in df.columns:
                    if "cropped" in df["img_data_roi"].iloc[0]:
                        df = df.set_index("img_data_roi")
                        if "config" in df.columns:
                            df = df.drop(columns="config")
            else:
                if "cropped" in df.index[0]:
                    if "config" in df.columns:
                        df = df.drop(columns="config")

            if "config" in df.columns:
                df.index = df["config"]
                string_df.index = df["config"]
                df = df.drop(columns=["config"])
            df = df.astype('float64')

            # Normalization
            df_zscore = pd.DataFrame()

            cols = list(df.columns)

            # Normalize on z-scores for each column
            for col in cols:
                if '_zscore' not in col:
                    col_zscore = col + '_zscore'
                    df_zscore[col_zscore] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

            df = df_zscore

            before = df.shape[0]

            # add config
            df = pd.concat([string_df, df], axis=1)
            after = df.shape[0]

            if before != after:
                self.error.error("Normalization index configuration seems to be wrong! ")
                raise ValueError("Normalization index configuration seems to be wrong! ")
        else:
            self.logger.info("Data already normalized!")

        return df

    def config_transformations_in_features_space(self, df: pd.DataFrame, peri:bool):
        """
        Configure the transformations in the feature space
        1. Sepperate image transformed samples
        2. Remove config features
        3. Drop all columns with constant values
        4. Convert transformed samples into feature classes
        5. Add config to feature space
        :param df: feature space containing samples with and without image transformations
        :param peri: bool if peritumoral masks gets processed
        :return: formatted df with transformations as columns in feature space
        """

        if "img_data_roi" != df.index.name:
            if "img_data_roi" in df.columns:
                df = df.set_index("img_data_roi")
            else:
                if "Image" in df.columns:
                    if "cropped" in df["Image"].iloc[0]:
                        df["img_data_roi"] = df["Image"].apply(lambda path: os.path.basename(path)[:-len(".nii.gz")])
                    else:
                        df["Img_file"] = df["Image"].apply(lambda path: os.path.basename(path)[:-len(".nii.gz")])
                        df["Msk_file"] = df["Mask"].apply(lambda path: os.path.basename(path)[:-len(".nii.gz")])
                        df["img_data_roi"] = df["Msk_file"] + "_" + df["Img_file"]
                        df = df.drop(columns=["Img_file", "Msk_file"])
                else:
                    self.error.error("Data not configured! Missing format and information to repair format.")
                    raise ValueError("Data not configured! Missing format and information to repair format.")

        # 1. Sepperate image transformed samples from samples
        # df.to_csv(self.out_path + "Before_image_transformation_sepperation.csv")
        if "Image_Transformation" in df.columns:
            df_wo_img_transform, df_img_transform = self.separate_transformations(
                                                                                Transformation="Image_Transformation",
                                                                                df=df.copy())               
        else:
            self.error.warning("Image_Transformation column is missing!")
            print("Image_Transformation column is missing!")
            df_wo_img_transform = df.copy()
            df_img_transform = pd.DataFrame()
        
        # remove config features
        tmp, df_img_transform_wo_config = self.sepperate_config_features(
                                                                        config_features=self.config_features,
                                                                        df=df_img_transform.copy())

        df_wo_img_transform_config, tmp = self.sepperate_config_features(
                                                                        config_features=self.config_features,
                                                                        df=df_wo_img_transform.copy())

        if (len(df_img_transform_wo_config) > 0) and (len(df_img_transform_wo_config.columns) > 0):
            try:
                self.logger.info("Drop columns with constant values.")
                # Drop columns with constant values
                columns_to_drop = [column for column in df_img_transform_wo_config.columns if df_img_transform_wo_config[column].nunique() == 1]
                df_img_transform_wo_config.drop(columns=columns_to_drop, inplace=True)
            except Exception as e:
                self.error.warning("Could not drop constant columns: {}".format(str(e)))
                print("Could not drop constant columns: {}".format(str(e)))

        # df_img_transform_wo_config.to_csv(self.out_path + "df_img_transform_wo_config.csv")
        # put all transformations into columns with suffix
        df_img_transform_in_col = self.processing_transformation_in_column(transform_df=df_img_transform_wo_config.copy())  # Here! not all samples are included in the feature space
        self.df_img_transform_in_col = df_img_transform_in_col

        # config index
        df = df_wo_img_transform.reset_index(drop=False)
        
        if len(df) > 0:
            if not self.delta:
                for img_data_roi in df["img_data_roi"]:
                    new_roi_ID = self.configure_cropped_index_format(target=img_data_roi)

                    if new_roi_ID != img_data_roi:
                        df.loc[df["img_data_roi"] == img_data_roi, "img_data_roi"] = new_roi_ID

            if "img_data_roi" in df.columns:
                df_wo_img_transform = df.set_index("img_data_roi")

        before = df_wo_img_transform.shape[0]
        
        if len(df_wo_img_transform) == 0:
            self.error.warning("No Samples found without Transformation!")
            print("Warning: No Samples found without Transformation!")
        
        if not peri:
            if len(df_img_transform_in_col) == 0:
                self.error.warning("No Samples found with Transformation!")
                print("Warning: No Samples found with Transformation!")

        df_wo_img_transform = df_wo_img_transform[~df_wo_img_transform.index.duplicated()]
        # df_wo_img_transform.to_csv(self.out_path + "df_wo_img_transform.csv")
        # df_img_transform_in_col.to_csv(self.out_path + "df_img_transform_in_col.csv")

        # if failed samples are there
        if len(self.samples_not_incuded_in_all_transformations) > 0:
            df_wo_img_transform = df_wo_img_transform[~df_wo_img_transform["ID"].isin(self.samples_not_incuded_in_all_transformations)]

        self.df_wo_img_transform = df_wo_img_transform
        self.df_img_transform_in_col = df_img_transform_in_col

        if not peri:
            # Add to data non-transformed data
            df_total = pd.concat([df_wo_img_transform, df_img_transform_in_col], axis=1)
            # df_total.to_csv(self.out_path + "df_total.csv")
        else:
            df_total = df_wo_img_transform
            # df_total.to_csv(self.out_path + "df_total_peri.csv")
        
        if before == 0:
            self.error.warning("Data does not contain any Samples with Image Transformation!")
            print("Data does not contain any Samples with Image Transformation!")
            before = df_total.shape[0]

        if before != df_total.shape[0]:
            try:
                # check if roi is in index in correct format
                id = df_wo_img_transform["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
                ID = pd.Series(index=df_wo_img_transform.index, data=id)
                df_wo_img_transform.index = ID
                df_wo_img_transform.index.rename("img_data_roi", inplace=True)

                df_total = pd.concat([df_wo_img_transform, df_img_transform_in_col], axis=1)

            except Exception as e:
                self.error.error("Transformation index configuration seems to be wrong! " + str(e))
                print("Transformation index configuration seems to be wrong! " + str(e))

            self.df_total = df_total.copy()

            if len(df_total.loc[df_total['ID'].isnull()]) > 0:
                if "ID.1" in df_total.columns:
                    if len(df_total.loc[df_total['ID.1'].isnull()]) < len(df_total.loc[df_total['ID'].isnull()]):
                        df.drop(['ID'], axis = 1, inplace = True) 
                        df.rename(columns={'ID.1': 'ID'}, inplace=True, errors='raise')

                if len(df_total.loc[df_total['ID'].isnull()]) > 0:
                    wrong = df_total.copy()[df_total.copy()["ID"].isnull()]
                    for id in wrong.index:
                        real_id = id.split("_")[0]  # Assume that first part of config is ID!!!
                        df_total.loc[df_total.index == id,"ID"] = real_id

                if len(df_total.loc[df_total['ID'].isnull()]) > 0:
                    dopped_samples = df_total.loc[df_total['ID'].isnull()].index.values
                    self.error.warning("Need to drop failed extraction for sample/s: " + str(dopped_samples))
                    print("Need to drop failed extraction for sample/s: " + str(dopped_samples))
                    df_total = df_total.loc[~df_total['ID'].isnull()]

            if before != df_total.shape[0]:
                self.error.error("Transformation adding index configuration seems to be wrong! Failed integrating {} samples!".format(str(int(abs(df_total.shape[0]-before)))))
                raise ValueError("Transformation adding index configuration seems to be wrong! Failed integrating {} samples!".format(str(int(abs(df_total.shape[0]-before)))))

        # drop all columns containing only nan values
        df_total = df_total.dropna(axis=1, how='all')

        # config index
        df = df_wo_img_transform_config.reset_index(drop=False)
        if len(df) > 0:
            for img_data_roi in df["img_data_roi"]:
                if not self.delta:
                    new_roi_ID = self.configure_cropped_index_format(target=img_data_roi)

                    if new_roi_ID != img_data_roi:
                        df.loc[df["img_data_roi"] == img_data_roi, "img_data_roi"] = new_roi_ID

            df_wo_img_transform_config = df.set_index("img_data_roi")

        # add config
        before = df_total.shape[0]
        # df_total.to_csv(self.out_path + "df_total.csv")
        # df_wo_img_transform_config.to_csv(self.out_path + "df_wo_img_transform_config.csv")
        # if failed samples are there
        if len(self.samples_not_incuded_in_all_transformations) > 0:
            df_wo_img_transform_config = df_wo_img_transform_config[~df_wo_img_transform_config["ID"].isin(self.samples_not_incuded_in_all_transformations)]

        df_all = pd.concat([df_wo_img_transform_config, df_total], axis=1)

        if before != df_all.shape[0]:
            df_all_ = pd.DataFrame()
            try:
                if not self.delta:
                    # check if roi is in index in correct format
                    id = df_wo_img_transform_config["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
                    ID = pd.Series(index=df_wo_img_transform_config.index, data=id)
                    df_wo_img_transform_config.index = ID
                    df_wo_img_transform_config.index.rename("img_data_roi", inplace=True)
                
                if (df_wo_img_transform_config.index.name == "img_data_roi") and df_total.index.name == "img_data_roi":
                    df_all_ = pd.concat([df_wo_img_transform_config, df_total], axis=1)
                    # df_all_.to_csv(self.out_path + "df_all_.csv")
                else:
                    print("Wrong Index config between df_wo_img_transform_config and df_total!")


            except Exception as e:
                self.error.error("Config adding index configuration seems to be wrong! " + str(e))
                print("Config adding index configuration seems to be wrong! " + str(e))

            if before != df_all_.shape[0]:
                self.error.error("Config adding index configuration seems to be wrong! ")
                raise ValueError("Config adding index configuration seems to be wrong! ")
            else:
                df_total = df_all_
                del df_all

        # drop all samples which do have more then 90% missing values
        df_total = df_total.dropna(axis=1, thresh=len(df_total) * 0.9)

        # drop all duplicated columns
        df_total = df_total.loc[:, ~df_total.columns.duplicated()].copy()

        # df_total.to_csv(self.out_path + "df_total_before_norm.csv")

        # Normalize features
        if len(df_total) > 1:
            df_total = self.normalize(df=df_total)
        else:
            self.error.warning("Only one sample in data frame. No normalization possible!")

        return df_total

    def replace_double_underscore_in_index(self, df):
        """
        Replaces double underscores '__' with a single underscore '_' in the index of a pandas DataFrame.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - DataFrame with the updated index.
        """

        df.index = df.index.str.replace("__", "_", regex=False)

        return df

    def get_additional_mask_transformations(self, df, mask_peturbations=None):
        """
        Search for additional ROIs to add to the feature space
        :param df: features contatining additional ROIs
        :param mask_peturbations: list od strings in "Mask_Transformation" to know which strings should be used for stability filtering
        """

        if mask_peturbations is None:
            mask_peturbations = self.mask_peturbations
        
        # if there are no Raters in the column rater - if rater is there it should be used for stability filtering
        if "Rater" in df.columns:
            df = df[df['Rater'].isnull()]
        
        # Get unique non-null transformations
        all_transformations = df["Mask_Transformation"].dropna().unique()
        
        # Filter out ignored values
        additional = [t for t in all_transformations if not any(ignore in t for ignore in mask_peturbations)]
        
        return sorted(additional)

    def check_for_transformation_completeness(self, df: pd.DataFrame()):
        """
        Check if all samples are completly extracted for all image transformations.
        Drop samples where not all transformations are included.
        :param df: pd.DataFrame data with radiomics features
        """

        def find_elements_not_in_all_lists(kernel_dict):
            """
            Check if all values from dice as lists are in all other lists
            :param kernel_dict: dict with transformartion kernel: lsit of sample IDs
            :return list: list of IDs which are not in cluded in all lists for all transformation kernels
            """

            all_values = set().union(*kernel_dict.values())
            common_values = set(kernel_dict[next(iter(kernel_dict))])

            for v in kernel_dict.values():
                common_values &= set(v)

            return list(all_values - common_values)

        # if there are different numbers of samples for image transformation is an indicator for missing samples in the data
        number_of_transformations = list(set(df['Image_Transformation'].value_counts().values))

        if len(number_of_transformations) > 0:
            # there might be not the correct ID in the data
            df["ID_prefix"] = df["ID"].str.extract(r'^([^_]+)', expand=False)

            # go through all transformations and check for samples which are not in all transformations
            image_transformation_kernels = list(set(df.copy().loc[~df['Image_Transformation'].isnull(),"Image_Transformation"].to_list()))
            transformation_sample_IDs = {}
            for kernel in image_transformation_kernels:
                transformation_sample_IDs[kernel] = df.copy().loc[df['Image_Transformation']==kernel,"ID_prefix"].to_list()

            samples_not_incuded_in_all_transformations = find_elements_not_in_all_lists(transformation_sample_IDs)
            if len(samples_not_incuded_in_all_transformations) > 0:
                print(f"Samples where some image transformations failed: {samples_not_incuded_in_all_transformations}")
                self.error.warning(f"Need to drop samples where some image transformations failed: {samples_not_incuded_in_all_transformations}")
                print(f"WARNING: Need to drop Sample/s with failed extraction. Check feature extraction for more information!")

                df = df[~df["ID_prefix"].isin(samples_not_incuded_in_all_transformations)]
                for sample in samples_not_incuded_in_all_transformations:
                    if sample not in self.samples_not_incuded_in_all_transformations:
                        self.samples_not_incuded_in_all_transformations.append(sample)

            df.drop(columns=["ID_prefix"], inplace=True)

        return df

    # 4.
    def extend_data_for_transformations(self, df: pd.DataFrame()):
        """
        Extract all image transformation and add the features according to each transformation to the columns of the
        data frame.
        :return: data frame with added transformation columns
        """

        df = self.check_for_transformation_completeness(df)

        peri_df_total = pd.DataFrame()

        # Need ROI name and ID but not transformation to add transformations to feature space as feature class
        if "img_data_roi" in df.columns:
            if not df.index.name == "img_data_roi":
                df = df.set_index("img_data_roi")
            # self.string_parameter.index = self.data.index

        elif ("img_data_roi" not in df.index.name) and ("img_data_roi" not in df.columns):
            mask = df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

            img_data_roi = pd.Series(index=df.index, data=mask)
            df["img_data_roi"] = img_data_roi

            df = df.set_index("img_data_roi")

        elif "img_data_roi" not in df.index.name:
            self.logger.error("Index names of dataframes do not match! \"img_data_roi\" is not present in data!")
            raise Exception("When separating the transformations ... Index names of dataframes do not match!")

        # search for index string containing features and remove them
        df = self.drop_columns_with_substring(df=df, substring="img_data_roi")

        self.logger.info("### 4.1 Separate Mask Transformations from Feature matrix ###")
        # df.to_csv(self.out_path + "df_before_separate.csv")
        
        # put all peritumoral features into the feature space and append them to the origial samples
        if "Peritumoral" in set(df["Mask_Transformation"]):
            self.logger.info("Peritumoral transformations detected")
            print("additional_rois_to_features",self.additional_rois_to_features)
            
            if self.additional_rois_to_features:
                print("Calculating additional ROIs peritumoral region ...")
                peri_df_total = pd.DataFrame()

                df['Mask_Transformation'] = df['Mask_Transformation'].replace(np.nan, "", regex=True)
                self.additional_ROIs = self.get_additional_mask_transformations(df=df.copy())

                index_name = df.index.name

                for roi in additional_ROIs:
                    print("# Processing Surounding ROI", roi)
                    if roi == "":
                        peri_roi_df = df.loc[df["Mask_Transformation"] == "Peritumoral", :]
                        print("Peritumoral " + roi, df[df['Mask_Transformation'] == "Peritumoral"].shape)
                        #print("Without " + "Peritumoral " + roi, df[df['Mask_Transformation'] != "Peritumoral"].shape)

                        df = df.loc[df["Mask_Transformation"] != "Peritumoral", :]
                    else:
                        print("Processing additional ROI surrounding region", roi)
                        roi_df = df[df['Mask_Transformation'].str.contains(roi)]
                        peri_roi_df, roi_df = self.get_peritumoral_samples(df=roi_df.copy())
                        print("Peritumoral " + roi, df[df['Mask_Transformation'] == roi + "_Peritumoral"].shape)
                        #print("Without " + "Peritumoral " + roi, df[df['Mask_Transformation'] != roi + "_Peritumoral"].shape)
                        df = df[df['Mask_Transformation'] != roi + "_Peritumoral"]

                    if self.peritumoral:
                        peri_roi_df = peri_roi_df.drop(columns=["Mask_Transformation"])

                        # drop duplicates based on the same image
                        duplicate = peri_roi_df[peri_roi_df.duplicated(['Image','Mask'])]["Image"].to_list()
                        
                        if len(duplicate) >0:
                            print("Found {} duplicates need to drop: ".format(str(len(duplicate))), duplicate)
                            self.error.warning("Found {} duplicates need to drop: ".format(str(len(duplicate))) + " " + str(duplicate))

                        # drop duplicates based on the same image
                        peri_roi_df = peri_roi_df.drop_duplicates(['Image','Mask'],keep= 'first')
                        if peri_roi_df.index.name != "img_data_roi":
                            if "img_data_roi" not in peri_roi_df.columns:
                                peri_roi_df["img_data_roi"] = np.nan
                                
                                for msk in peri_roi_df["Mask"]:
                                    img_data_roi = os.path.basename(msk)[:-len(".nii.gz")]
                                    peri_roi_df.loc[peri_roi_df["Mask"] == msk, "img_data_roi"] = img_data_roi
                            
                            peri_roi_df = peri_roi_df.reset_index(drop=False)
                            peri_roi_df.index = peri_roi_df["img_data_roi"]

                        print("Config Peritumoral Features ...")
                        peri_roi_df_total = self.config_transformations_in_features_space(df=peri_roi_df.copy(), peri=True)
                        if "ID.1_peritumoral" in peri_roi_df_total.columns:
                            peri_roi_df_total.drop(["ID.1_peritumoral"], axis = 1, inplace = True) 
                    else:
                        peri_roi_df_total = pd.DataFrame()

                    peri_roi_df_total_index_name = peri_roi_df_total.index.name
                    peri_roi_df_total = peri_roi_df_total.reset_index().set_index("ID", drop=False)
                    
                    if roi != "":
                        if "Prediction_Label" in peri_roi_df_total.columns:
                            peri_roi_df_total.drop(columns=["Prediction_Label"], inplace=True)
                        if "Modality" in peri_roi_df_total.columns:
                            peri_roi_df_total.drop(columns=["Modality"], inplace=True)

                        peri_roi_df_total = peri_roi_df_total.add_suffix("_" + roi)

                    peri_df_total = pd.concat([peri_df_total, peri_roi_df_total], axis=1)
                
                peri_df_total = peri_df_total.set_index(peri_roi_df_total_index_name, drop=False)

                print(f"Peritumoral Feature space {peri_df_total.shape}")
                print(f"Non-Peritumoral Feature space {df.shape}")

                df['Mask_Transformation'] = df['Mask_Transformation'].replace(r'^\s*$', np.nan, regex=True)

            else:
                peri_df, df = self.get_peritumoral_samples(df=df.copy())

                if self.peritumoral:
                    peri_df = peri_df.drop(columns=["Mask_Transformation"])

                    # drop duplicates based on the same image
                    duplicate = peri_df[peri_df.duplicated(['Image','Mask'])]["Image"].to_list()
                    
                    if len(duplicate) >0:
                        print("Found {} duplicates need to drop: ".format(str(len(duplicate))), duplicate)
                        self.error.warning("Found {} duplicates need to drop: ".format(str(len(duplicate))) + " " + str(duplicate))

                    # drop duplicates based on the same image
                    peri_df = peri_df.drop_duplicates(['Image','Mask'],keep= 'first')
                    if peri_df.index.name != "img_data_roi":
                        if "img_data_roi" not in peri_df.columns:
                            peri_df["img_data_roi"] = np.nan
                            
                            for msk in peri_df["Mask"]:
                                img_data_roi = os.path.basename(msk)[:-len(".nii.gz")]
                                peri_df.loc[peri_df["Mask"] == msk, "img_data_roi"] = img_data_roi
                        
                        peri_df = peri_df.reset_index(drop=False)
                        peri_df.index = peri_df["img_data_roi"]
                    
                    self.peri_df = peri_df

                    print("Config Peritumoral Features ...")
                    peri_df_total = self.config_transformations_in_features_space(df=peri_df.copy(), peri=True)
                    if "ID.1_peritumoral" in peri_df_total.columns:
                        peri_df_total.drop(["ID.1_peritumoral"], axis = 1, inplace = True) 
                else:
                    peri_df_total = pd.DataFrame()

                
                print(f"Peritumoral Feature space {peri_df_total.shape}")
                print(f"Non-Peritumoral Feature space {df.shape}")
        else:
            print("No Peritumoral mask transformations detected")
            self.error.warning("No Peritumoral mask transformations detected")

        # If there are multiple rois to add to the feature space
        if self.additional_rois_to_features:
            # df['Mask_Transformation'] = df['Mask_Transformation'].replace(r'^\s*$', np.nan, regex=True)
            df['Mask_Transformation'] = df['Mask_Transformation'].replace(np.nan, "", regex=True)
            self.additional_ROIs = self.get_additional_mask_transformations(df=df.copy())

            df_total_wo_msk_trans = pd.DataFrame()
            df_total_msk_trans = pd.DataFrame()

            # for each ROI
            # df['Mask_Transformation'] = df['Mask_Transformation'].replace(np.nan, '')
            for roi in additional_ROIs:
                print("# Processing ROI", roi)
                # If roi is real roi (blank)
                if roi == "":
                    real_df = df.copy() 
                    for a_roi in additional_ROIs:
                        if a_roi != "":
                            real_df = real_df[~real_df['Mask_Transformation'].str.contains(a_roi)]

                    roi_df = real_df.copy()
                    real_df = pd.DataFrame()

                else:
                    roi_df = df[df['Mask_Transformation'].str.contains(roi)]
                    print(f"Config Mask Transformed Features for ROI {roi}...")

                # set samples without mask transformation Mask_Transformation to nan
                roi_df.loc[roi_df["Mask_Transformation"] == roi, "Mask_Transformation"] = np.nan

                if roi != "":
                    # remove roi substing from samples to concat all samples in standart form
                    roi_df['Mask_Transformation'] = roi_df['Mask_Transformation'].str.replace(roi + "_", '', regex=True)

                # get features influenced by mask transformations also including image transformations for stability analysis
                df_roi_wo_msk_trans, df_roi_msk_transform = self.separate_transformations(
                                                                                        Transformation="Mask_Transformation",
                                                                                        df=roi_df.copy())

                if len(df_roi_msk_transform) > 0:
                    df_roi_total_msk_trans = self.config_transformations_in_features_space(df=df_roi_msk_transform.copy(), peri=False)
                else:
                    self.logger.info("No Mask Perturbated Features found.")
                    df_roi_total_msk_trans = pd.DataFrame()

                if len(df_roi_total_msk_trans) > 0:

                    self.logger.info("New Size of Mask Transformed Feature matrix:")
                    self.logger.info("Samples: " + str(df_roi_total_msk_trans.shape[0]))
                    self.logger.info("Features: " + str(df_roi_total_msk_trans.shape[1]) + "\n")
                    # df_roi_total_msk_trans.to_csv(self.out_path + "df_roi_total_msk_trans.csv")

                else:
                    self.error.warning("No Mask Perturbated Features found.")
                    print("No Mask Perturbated Features found.")

                df_roi_total_wo_msk_trans = self.config_transformations_in_features_space(df=df_roi_wo_msk_trans.copy(), peri=False)

                df_roi_total_wo_msk_trans_index_name = df_roi_total_wo_msk_trans.index.name
                df_roi_total_msk_trans_index_name = df_roi_total_msk_trans.index.name
                
                if "ID" in df_roi_total_wo_msk_trans.columns:
                    df_roi_total_wo_msk_trans = df_roi_total_wo_msk_trans.reset_index().set_index("ID",drop=False)

                if "ID" in df_roi_total_msk_trans.columns:
                    df_roi_total_msk_trans = df_roi_total_msk_trans.reset_index().set_index("ID", drop=False)    

                if roi != "":
                    if "Prediction_Label" in df_roi_total_wo_msk_trans.columns:
                        df_roi_total_wo_msk_trans.drop(columns=["Prediction_Label"], inplace=True)

                    if "Modality" in df_roi_total_wo_msk_trans.columns:
                        df_roi_total_wo_msk_trans.drop(columns=["Modality"], inplace=True)
                    
                    if "Prediction_Label" in df_roi_total_msk_trans.columns:
                        df_roi_total_msk_trans.drop(columns=["Prediction_Label"], inplace=True)

                    if "Modality" in df_roi_total_msk_trans.columns:
                        df_roi_total_msk_trans.drop(columns=["Modality"], inplace=True)

                    df_roi_total_wo_msk_trans = df_roi_total_wo_msk_trans.add_suffix("_" + roi)
                    df_roi_total_msk_trans = df_roi_total_msk_trans.add_suffix("_" + roi)

                df_total_wo_msk_trans = pd.concat([df_total_wo_msk_trans, df_roi_total_wo_msk_trans], axis=1)
                df_total_msk_trans = pd.concat([df_total_msk_trans, df_roi_total_msk_trans], axis=1)
            
            self.df_total_wo_msk_trans_ = df_total_wo_msk_trans
            self.df_total_msk_trans_ = df_total_msk_trans

            df_total_wo_msk_trans = df_total_wo_msk_trans.set_index(df_roi_total_wo_msk_trans_index_name, drop=False)
            
            # if there is mask transformation
            if len(df_total_msk_trans):
                df_total_msk_trans = df_total_msk_trans.set_index(df_roi_total_msk_trans, drop=False)

        else:
            # df.to_csv(self.out_path + "df.csv")
            print("Config Mask Transformed Features ...")
            # get features influenced by mask transformations also including image transformations for stability analysis
            df_wo_msk_trans, df_msk_transform = self.separate_transformations(
                                                                    Transformation="Mask_Transformation",
                                                                    df=df.copy())

            print("Config Image Transformed Features ...")
            self.logger.info("### 4.2 Separate Image Transformations from Feature matrix ###")
            self.logger.info("Feature space with Mask Transformation: " + str(df_msk_transform.shape))
            self.logger.info("Feature space without Mask Transformation: " + str(df_wo_msk_trans.shape))
            print("Feature space with Mask Transformation: ", df_msk_transform.shape)
            print("Feature space without Mask Transformation: ", df_wo_msk_trans.shape)

            # df_msk_transform.to_csv(self.out_path + "df_msk_transform.csv")
            # df_wo_msk_trans.to_csv(self.out_path + "df_wo_msk_trans.csv")
            
            if len(df_msk_transform) > 0:
                df_total_msk_trans = self.config_transformations_in_features_space(df=df_msk_transform.copy(), peri=False)
            else:
                self.logger.info("No Mask Perturbated Features found.")
                df_total_msk_trans = pd.DataFrame()

            if len(df_total_msk_trans) > 0:

                self.logger.info("New Size of Mask Transformed Feature matrix:")
                self.logger.info("Samples: " + str(df_total_msk_trans.shape[0]))
                self.logger.info("Features: " + str(df_total_msk_trans.shape[1]) + "\n")
                # df_total_msk_trans.to_csv(self.out_path + "df_total_msk_trans.csv")

            else:
                self.error.warning("No Mask Perturbated Features found.")
                print("No Mask Perturbated Features found.")

            df_wo_msk_trans.to_csv(self.out_path + "df_wo_msk_trans.csv")
            df_total_wo_msk_trans = self.config_transformations_in_features_space(df=df_wo_msk_trans.copy(), peri=False)

        # df_total_wo_msk_trans.to_csv(self.out_path + "df_total_wo_msk_trans.csv")

        # if normalized and resampled, index might not fit
        for conf in peri_df_total.copy().index:
            if conf not in df_total_wo_msk_trans.index.to_list():
                for wo_trans_conf in df_total_wo_msk_trans.copy().index:
                    if conf.startswith(wo_trans_conf):
                        df_total_wo_msk_trans.loc[wo_trans_conf, "config"] = conf
                    elif wo_trans_conf.startswith(conf):
                        df_total_wo_msk_trans.loc[wo_trans_conf, "config"] = conf

        # Add Peritumoral sample to feature space
        if len(peri_df_total) > 0:
            # peri_df_total.to_csv(self.out_path + "peri_df_total_0.csv")
            self.logger.info("### 4.3 Add Peritumoral Features to Feature matrix")

            # add peritumoral region as another sample to the feature space
            if self.separate_peritumoral_region:

                # add all peritumoral features from all rois to dataframe
                if self.additional_rois_to_features:

                    concat_peri_df_total = pd.DataFrame()
                    for roi in peri_df_total:
                        concat_peri_df_total = pd.concat([concat_peri_df_total, peri_df_total[roi]], axis=0)

                    peri_df_total = concat_peri_df_total.copy()
                    concat_peri_df_total = pd.DataFrame()

                df_total_wo_msk_trans = pd.concat([df_total_wo_msk_trans, peri_df_total], axis=0)

            # add peritumoral region as another feature class to the feature space
            else:
                # 1. Sync the index
                peri_index = peri_df_total.index.tolist()
                
                for i in peri_index:
                    if "peritumoral" not in i:
                        # For MIRP
                        if "id_subject" in peri_df_total.columns:
                            if isinstance(peri_df_total.loc[i,"id_subject"], pd.Series):
                                i = peri_df_total.loc[i,"id_subject"].values[0]
                            else:
                                i = peri_df_total.loc[i,"id_subject"]

                                
                    match = re.findall("roiT_[0-9]*_roiN_[0-9]*[_A-Za-z]*(_[0-9]*[_]*peritumoral)", i)
                    if (match is not None) and (len(match) >0):
                        new_i = i.replace(match[0], '')
                        # new_i = i[:-len(match[0])]
                        peri_df_total.rename(index={i: new_i}, inplace=True)
                    else:
                        self.error.warning("Configuration of sample names in peritumoral extracted features is wrong! Wrong format: " + str(i))

                # 2. Drop config columns
                for config in self.config_features:
                    if config in peri_df_total.columns:
                        peri_df_total = peri_df_total.loc[:, ~peri_df_total.columns.str.contains(config)]

                # peri_df_total.to_csv(self.out_path + "peri_df_total_1.csv")

                # 3. add suffix "_peritumoral" to features
                peri_df_total = peri_df_total.add_suffix("_peritumoral")
            

                # peri_df_total.to_csv(self.out_path + "peri_df_total_2.csv")

                # config index
                if len(peri_df_total) > 0:

                    if peri_df_total.index.name == "config":
                        peri_df_total.index.name = "img_data_roi"

                        df = peri_df_total.reset_index(drop=False)
                        for img_data_roi in df["img_data_roi"]:

                            new_roi_ID = self.configure_cropped_index_format(target=img_data_roi)

                            if new_roi_ID != img_data_roi:
                                df.loc[df["img_data_roi"] == img_data_roi, "img_data_roi"] = new_roi_ID

                        peri_df_total = df.set_index("img_data_roi")
                        peri_df_total.index.name = "config"

                else:
                    self.error.warning("No Peritumoral Features found.")
                    print("No Peritumoral Features found.")

                self.peri_df_total_ = peri_df_total
                

                # drop features with NaN
                df_total_wo_msk_trans = df_total_wo_msk_trans.dropna(axis='columns')
                peri_df_total = peri_df_total.dropna(axis='columns')

                before = len(df_total_wo_msk_trans) # .shape[0]

                if peri_df_total.index.name != df_total_wo_msk_trans.index.name:
                    peri_df_total.index.name = df_total_wo_msk_trans.index.name

                # peri_df_total.to_csv(self.out_path + "peri_df_total.csv")
                # df_total_wo_msk_trans.to_csv(self.out_path + "df_total_wo_msk_trans_0.csv")

                # check if all index are there
                configuration_problem = False
                peri_df_total.index = peri_df_total.index.to_series().str.replace(r"_\d+_peritumoral$", "", regex=True)
                for peri_index in peri_df_total.index:
                    if peri_index not in df_total_wo_msk_trans.index:
                        configuration_problem = True
                        break

                if configuration_problem:
                    self.error.warning("Peritumoral index has different configuration! Trying to solve this problem ...")
                    print("Peritumoral index has different configuration! Trying to solve this problem ...")
                    print(peri_df_total.index[0])
                    print(df_total_wo_msk_trans.index[0])

                    df_total_wo_msk_trans["config"] = df_total_wo_msk_trans.copy().index.to_list()
                    peri_df_total.index = peri_df_total.index.to_series().str.replace(r"_\d+_peritumoral$", "", regex=True)

                    # if normalized and resampled, index might not fit
                    for conf in peri_df_total.copy().index:
                        if conf not in df_total_wo_msk_trans.index.to_list():
                            for wo_trans_conf in df_total_wo_msk_trans.copy().index:
                                if conf.startswith(wo_trans_conf):
                                    df_total_wo_msk_trans.loc[wo_trans_conf, "config"] = conf
                                elif wo_trans_conf.startswith(conf):
                                    df_total_wo_msk_trans.loc[wo_trans_conf, "config"] = conf

                    df_total_wo_msk_trans.set_index('config', inplace = True)

                    if "config" in df_total_wo_msk_trans.columns:
                        df_total_wo_msk_trans.drop(['config'], axis = 1, inplace = True) 
                
                self.df_total_wo_msk_trans = df_total_wo_msk_trans
                self.peri_df_total = peri_df_total

                # drop duplicates if in columns
                df_total_wo_msk_trans.loc[:,~df_total_wo_msk_trans.columns.str.contains('ID.', case=False)] 

                self.df_total_wo_msk_trans =  df_total_wo_msk_trans.copy()
                self.peri_df_total = peri_df_total.copy()

                # print(self.df_total_wo_msk_trans.index.name, self.df_total_wo_msk_trans.index)

                self.df_total_wo_msk_trans = self.replace_double_underscore_in_index(self.df_total_wo_msk_trans.copy())
                self.peri_df_total = self.replace_double_underscore_in_index(self.peri_df_total.copy())

                # check if there are peritumoral extractions which are not with the correct configuration (roiT_*_roiN_*)
                if peri_df_total.index.name == "config":
                    peri_df_total.reset_index(drop=False, inplace=True)

                if "config" in peri_df_total.columns:
                    # not syncronized index! --> not correct configuration
                    peri_df_total = peri_df_total[~peri_df_total["config"].str.contains("peritumoral")]
                    peri_df_total.set_index("config", inplace=True)
                else:
                    if len(peri_df_total) != len(df_total_wo_msk_trans):
                        peri_df_total = peri_df_total[~peri_df_total.index.str.contains("peritumoral")]

                if len(peri_df_total) != len(df_total_wo_msk_trans):
                    idx1 = set(peri_df_total.index)
                    idx2 = set(df_total_wo_msk_trans.index)

                    only_in_peri = idx1 - idx2
                    only_in_wo_msk = idx2 - idx1

                    print(f"Only in peritumoral samples: {sorted(only_in_peri)}")
                    print(f"Only in raw not transformed samples: {sorted(only_in_wo_msk)}")

                    if len(peri_df_total) > len(df_total_wo_msk_trans):
                        self.error.error(f"More samples with peritumoral feature extraction are included than without! Check for failed feature extractions! {len(peri_df_total)} Peritumoral sample but only {len(df_total_wo_msk_trans)} non peritumoral samples.")
                        raise ValueError(f"More samples with peritumoral feature extraction are included than without! Check for failed feature extractions! {len(peri_df_total)} Peritumoral sample but only {len(df_total_wo_msk_trans)} non peritumoral samples.")
                    if len(peri_df_total) < len(df_total_wo_msk_trans):
                        self.error.error("Less samples with peritumoral feature extraction are included than without! Check for failed peritumoral feature extractions!")
                        raise ValueError("Less samples with peritumoral feature extraction are included than without! Check for failed peritumoral feature extractions!")

                if df_total_wo_msk_trans.index.name != peri_df_total.index.name:
                    peri_df_total.index.name = df_total_wo_msk_trans.index.name

                df_total_wo_msk_trans = self.replace_double_underscore_in_index(df_total_wo_msk_trans.copy())
                peri_df_total = self.replace_double_underscore_in_index(peri_df_total.copy())

                # 4. add peritumoral features to feature space
                df_total_wo_msk_trans = pd.concat([df_total_wo_msk_trans, peri_df_total], axis=1)
                self.df_total_wo_msk_trans_concat = df_total_wo_msk_trans
                # df_total_wo_msk_trans.to_csv(self.out_path + "df_total_wo_msk_trans.csv")

                # drop samples with NaN if concat is wrong
                df_total_wo_msk_trans = df_total_wo_msk_trans.dropna(axis=0)  # --> getting warning and exit if so
                # remove duplicated columns
                df_total_wo_msk_trans = df_total_wo_msk_trans.loc[:, ~df_total_wo_msk_trans.columns.duplicated()].copy()
                # df_total_wo_msk_trans.to_csv(self.out_path + "df_total_wo_msk_trans_wo_dupl_col.csv")

                after = len(df_total_wo_msk_trans)#.shape[0]
                
                if before != after:
                    print(self.df_total_wo_msk_trans.index[0],  self.peri_df_total.index[0])
                    # get data from before concat and try to solve the problem
                    if self.df_total_wo_msk_trans.index[0] not in self.peri_df_total.index:
                        peri_index_name = self.peri_df_total.index.name
                        self.peri_df_total = self.peri_df_total.reset_index()

                        print("Detected index configuration problem while integrating peritumoral features ... Trying to solve it.")
                        for config in self.df_total_wo_msk_trans.index:
                            # get unique ID for sync
                            id = config.split("_")[0]
                            # get config for sync
                            peri_conf = self.peri_df_total.copy().loc[self.peri_df_total[peri_index_name].str.startswith(id), peri_index_name].values[0]
                            # replace config
                            self.peri_df_total.loc[self.peri_df_total[peri_index_name] == peri_conf, peri_index_name] = config

                        self.peri_df_total = self.peri_df_total.set_index(peri_index_name)
                        if self.peri_df_total.index.duplicated().sum() > 0:
                            print("Detected duplicated {} index in peritumoral data.".format(str(self.peri_df_total.index.duplicated().sum())))

                        # sync index names
                        self.peri_df_total.index.name = self.df_total_wo_msk_trans.index.name

                        # if self.df_total_wo_msk_trans.index.duplicated or self.peri_df_total.index.duplicated:


                        before = len(self.df_total_wo_msk_trans)
                        self.df_total_wo_msk_trans = pd.concat([self.df_total_wo_msk_trans, self.peri_df_total], axis=1)
                        after = len(self.df_total_wo_msk_trans)

                        if before != after:
                            self.error.error("Peritumoral index configuration seems to be wrong! ")
                            print("Peritumoral index configuration seems to be wrong! ")
                            raise ValueError("Peritumoral index configuration seems to be wrong! ")
                        else:
                            df_total_wo_msk_trans = self.df_total_wo_msk_trans.copy()

                        #df_total_wo_msk_trans.to_csv(self.out_path + "df_total_wo_msk_trans.csv")
                        df_total_wo_msk_trans = df_total_wo_msk_trans.dropna(axis=0)
                        df_total_wo_msk_trans = df_total_wo_msk_trans.loc[:, ~df_total_wo_msk_trans.columns.duplicated()].copy()
                        #df_total_wo_msk_trans.to_csv(self.out_path + "df_total_wo_msk_trans_wo_dupl_col.csv")

                    else:
                        self.error.error("Peritumoral index configuration seems to be wrong! ")
                        print("Peritumoral index configuration seems to be wrong! ")
                        raise ValueError("Peritumoral index configuration seems to be wrong! ")  

                # drop all features which do have more then 90% missing values
                df_total_wo_msk_trans = df_total_wo_msk_trans.dropna(axis=1, thresh=len(df_total_wo_msk_trans) * 0.9)
        else:
            self.error.warning("Could not find any peritumoral segmentation.")
            print("Could not find any peritumoral segmentation.")

        self.logger.info("New Size of Image Transformed Feature matrix:")
        self.logger.info("Samples: " + str(df_total_wo_msk_trans.shape[0]))
        self.logger.info("Features: " + str(df_total_wo_msk_trans.shape[1]) + "\n")
        
        # write filtered normalied csv files to tmp folder
        if not os.path.exists(self.out_path + "tmp/"):
            os.makedirs(self.out_path + "tmp/")

        df_total_wo_msk_trans.to_csv(self.out_path + "tmp/" + self.RunID + "_normalized_data_wo_mask_transformation.csv")
        df_total_msk_trans.to_csv(self.out_path + "tmp/" + self.RunID + "_normalized_data_mask_transformation.csv")

        # df_total_wo_msk_trans.to_csv("df_total_wo_msk_trans.csv")
        return df_total_wo_msk_trans, df_total_msk_trans

    def drop_columns_with_substring(self, df: pd.DataFrame(), substring: str):
        """
        Drop all parameter columns with a specific substring.
        :parameter df: pd.DataFrame() with parameters
        :parameter substring: str() with substring
        :return: df: pd.DataFrame() with dropped columns containing substring
        """
        drop_col = df.columns[df.columns.str.contains(substring)].values
        if len(drop_col) >= 1:
            df = df.drop(columns=drop_col)

        return df

    def processing_transformation_in_column(self, transform_df: pd.DataFrame()):
        """
        Processing the transformation in the column as suffix in the feature name
        :param transform_df: pd.DataFrame() with image transformation
        :return: tmp_total: pd.DataFrame() with all transformations in columns
        """
        processed_transformations = pd.DataFrame()
        out = []

        # if there are different numbers of samples for image transformation is an indicator for missing samples in the data
        #number_of_transformations = list(set(transform_df['Image_Transformation'].value_counts().values))

        #if len(number_of_transformations) > 0:

            # go through all transformations and check for samples which are not in all transformations
        #    image_transformation_kernels = list(set(transform_df.copy().loc[~transform_df['Image_Transformation'].isnull(),"Image_Transformation"].to_list()))
        #    transformation_sample_IDs = {}
        #    for kernel in image_transformation_kernels:
        #        transformation_sample_IDs[kernel] = df_non_peritumoral.copy().loc[df_non_peritumoral['Image_Transformation']==kernel,"ID"].to_list()

        #    def find_elements_not_in_all_lists(d):
        #        "Check if all values from dice as lists are in all other lists"

        #        all_values = set().union(*d.values())
        #        common_values = set(d[next(iter(d))])

        #        for v in d.values():
        #            common_values &= set(v)

        #        return list(all_values - common_values)


        #    samples_not_incuded_in_all_transformations = find_elements_not_in_all_lists(transformation_sample_IDs)
        #    if len(samples_not_incuded_in_all_transformations) > 0:
        #        print(f"Samples where some image transformations failed: {samples_not_incuded_in_all_transformations}")
        #        self.error.warning(f"Need to drop samples where some image transformations failed: {samples_not_incuded_in_all_transformations}")
        #        print(f"WARNING: Need to drop Sample/s with failed extraction. Check feature extraction for more information!")
        #        transform_df = transform_df[~transform_df["ID"].isin(samples_not_incuded_in_all_transformations)]
        #        for sample in samples_not_incuded_in_all_transformations:
        #            if sample not in self.samples_not_incuded_in_all_transformations:
        #                self.samples_not_incuded_in_all_transformations.append(sample)


        if len(transform_df) > 0:
            print("Processing {} Transformations and add them to the feature space: {}".format(len(list(set(transform_df["Image_Transformation"]))), 
                                                                                                list(set(transform_df["Image_Transformation"]))))
            # multi threading if we want to use more cpus
            self.transformed_samples = None
            if self.n_cpu > 1:
                if "Image_Transformation" in transform_df.columns:
                    if len(list(set(transform_df.copy()["Image_Transformation"]))) > 0:
                        try:
                            prod_df = partial(self.add_transformations_as_suffix, df=transform_df.copy())
                            
                            with tqdm(total=len(list(set(transform_df.copy()["Image_Transformation"]))), desc='Config Transformations in Feature Space') as pbar:
                                
                                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                                    for results in executor.map(prod_df, [trans for trans in list(set(transform_df.copy()["Image_Transformation"]))], chunksize=self.n_cpu):
                                        pbar.update(1)
                                        if len(results.columns.duplicated(keep='first')) > 0:
                                            # print("Found {} duplicated features!".format(len(results.columns.duplicated(keep='first'))))
                                            results = results.loc[:,~results.columns.duplicated(keep='first')]
                                        out.append(results)

                        except Exception as ex:
                            self.error.error("Config Feature Failed! " + str(ex))
                            self.error.error(traceback.format_exc())
                            raise Exception("Config Feature Failed! " + str(ex))

                # prod_df = partial(self.add_transformations_as_suffix, df=transform_df)
                #
                # p = Pool(self.n_cpu)
                # out = p.imap(prod_df,
                #              list(set(transform_df["Image_Transformation"])))
                # p.close()
                # p.join()
            else:
                if len(transform_df["Image_Transformation"].unique()) > 0:
                    
                    for transformation in transform_df["Image_Transformation"].unique():
                        out.append(self.add_transformations_as_suffix(transform=transformation, df=transform_df.copy()))

            # print("Found Image Transformations", transform_df["Image_Transformation"].unique())
            # drop duplicated column names
            # processed_transformations = processed_transformations.loc[:, ~processed_transformations.columns.duplicated()]

            #if len(processed_transformations.columns) > 1:
            #    feature = processed_transformations.columns[0]
            #    processed_transformations = processed_transformations[processed_transformations[feature] != feature]

            if len(out) > 0:

                # check if format needs syncronization
                if "_cropped_resample" in out[0].index[0]:
                    for i in tqdm(range(len(out)),  desc='Config Index'):
                        df = out[i].reset_index(drop=False)

                        kernel_IO = ""
                        for kernel in list(self.track_config.values()):
                            for roi_ID in df["img_data_roi"]:
                                if kernel in roi_ID:
                                    kernel_IO = kernel
                                    break

                            if kernel_IO != "":
                                break

                        if kernel_IO != "":
                            for img_data_roi in df["img_data_roi"]:
                                new_roi_ID = self.configure_cropped_index_format(target=img_data_roi, kernelIO=kernel_IO)

                                if new_roi_ID != img_data_roi:
                                    df.loc[df["img_data_roi"] == img_data_roi, "img_data_roi"] = new_roi_ID

                        df = df.loc[:,~df.columns.duplicated(keep='first')]
                        out[i] = df.set_index("img_data_roi")
                else:
                    for i in tqdm(range(len(out)),  desc='Config Index'):

                        df = out[i].reset_index(drop=False)

                        # drop duplicated columns
                        df = df.loc[:,~df.columns.duplicated(keep='first')]
                        df.set_index("img_data_roi", inplace=True)
                        # df = df.drop_duplicates()

                        # drop duplicated index
                        if df.index.duplicated().sum() > 0:
                            df = df.loc[~df.index.duplicated()]

                        
                        out[i] = df

                self.out = out

                self.logger.info("Concat Configured Feature Space ...")
                print("Concat Configured Feature Space ..." )

                processed_transformations = pd.concat(out, axis=1)
                print("Processed transformed samples", processed_transformations.shape)

                processed_transformations = processed_transformations.loc[:, ~processed_transformations.columns.duplicated()]
                #processed_transformations.to_csv(self.out_path + "processed_transformations.csv")

        else:
            self.error.warning("Input is empty. Check for input data to get transformed feature space.")
            processed_transformations = transform_df

        return processed_transformations

    def configure_cropped_index_format(self, target: str, kernelIO: str = ""):
        """
        Configure the index to concat data into dataframe
        :param kernelIO: Transformation kernel
        :param target: index from dataframe to configure
        :return: configured index
        """

        new_roi_ID = target

        # remove kernel from index so that it is matchable with non transformed samples
        if kernelIO != "":
            match = re.search("(" + kernelIO + "[A-Z0-9a-z-.+_]*)cropped_resample", target)
            if match:
                new_roi_ID = target.replace(match.group(1), '')

        new_roi_ID = new_roi_ID.replace("__", '_')
        new_roi_ID = new_roi_ID.replace("_CT_", '_')
        new_roi_ID = new_roi_ID.replace("_MR_", '_')
        new_roi_ID = new_roi_ID.replace("_peritumoral_", '_')
        new_roi_ID = new_roi_ID.replace("imagecropped", 'image_cropped')

        if new_roi_ID.endswith("_cr"):
            new_roi_ID = new_roi_ID.replace("_cr", '_cropped_resample')

        # syncronize peritumoral index for data attachment
        match = re.search("_roiT_[0-9]+_((roiN_[0-9]+_)[0-9]+_)", new_roi_ID)
        if match:
            new_roi_ID = new_roi_ID.replace(match.group(1), match.group(2))

        return new_roi_ID

    def add_transformations_as_suffix(self, transform: str, df: pd.DataFrame()):
        """
        Concatenate all transformations to one data frame where the kernel gets added to the name of the column
        as a suffix. Apply this function for all transformations or a specific transformation
        :param df: data frame with all transformations
        :param transform: name of transformation kernel which should be added
        :return: tmp: data frame with added transformation columns
        """
        cols_to_drop = []

        #if ("wavelet" not in transform) and ("Wavelet" not in transform):
        #    print("GOT IT!!!!  Config {} in feature space.".format(transform))

        # For msk
        # df = self.df_mask_img_transform
        # For img
        # df = self.df_wo_msk_img_trans
        # get all samples which have been transformed with the kernel
        tmp = df.copy().loc[df["Image_Transformation"] == transform]
        transformed_samples = tmp.shape[0]

        if self.transformed_samples is None: 
            self.transformed_samples = transformed_samples
        else:
            if self.transformed_samples != transformed_samples:
                # can be caused if supervoxels are better in one transforamtion than another
                self.error.warning("Unequal {} Transformations detected! Samples with this transformation {} samples with others {}.".format(transform, str(transformed_samples), str(self.transformed_samples)))
                print("Warning: Unequal {} Transformations detected! Samples with this transformation {} samples with others {}.".format(transform, str(transformed_samples), str(self.transformed_samples)))
                # aise ValueError("Failed {} Transformations detected! Samples with this transformation {} samples with others {}.".format(transform, str(transformed_samples), str(self.transformed_samples)))
        
        # df_total_trans.drop([tmp.index], axis=0)
        if self.extractor == "MIRP":
            # Drop columns which are not specific for the transformation and add suffix to the columns
            if "id_subject" in tmp.columns:
                tmp = tmp.drop(["ID",
                                "Image",
                                "Mask",
                                "Prediction_Label",
                                "Image_Transformation",
                                "Modality",
                                "ROI_Label",
                                "Mask_Transformation",
                                "Timepoint",
                                "config",  # remove config (edit 08.11)
                                "Rater",
                                "id_subject",
                                "id_cohort",
                                "img_data_settings_id",
                                "img_data_modality"], axis=1, errors='ignore')
                # features are for image configurations not interesting for transformed images
                cols_to_drop = list(tmp.columns[tmp.columns.str.contains('img_data_')])
                cols_to_drop.extend(list(tmp.columns[tmp.columns.str.contains('diag_')]))
            else:
                tmp = tmp.drop(["ID",
                                "Image",
                                "Mask",
                                "Prediction_Label",
                                "Image_Transformation",
                                "Modality",
                                "ROI_Label",
                                "Mask_Transformation",
                                "Timepoint",
                                "config",  # remove config (edit 08.11)
                                "Rater",
                                "id_cohort",
                                "img_data_settings_id",
                                "img_data_modality"], axis=1, errors='ignore')
                # features are for image configurations not interesting for transformed images
                cols_to_drop = list(tmp.columns[tmp.columns.str.contains('img_data_')])
                cols_to_drop.extend(list(tmp.columns[tmp.columns.str.contains('diag_')]))

        elif self.extractor == "PyRadiomics":

            if "id_subject" in tmp.columns:
                tmp = tmp.drop(["ID",
                                "Image",
                                "Mask",
                                "Prediction_Label",
                                "Image_Transformation",
                                "Modality",
                                "config",  # remove config (edit 08.11)
                                "ROI_Label",
                                "Mask_Transformation",
                                "Timepoint",
                                "id_subject"], axis=1, errors='ignore')

            else:
                tmp = tmp.drop(["ID",
                                "Image",
                                "Mask",
                                "Prediction_Label",
                                "Image_Transformation",
                                "Modality",
                                "config",  # remove config (edit 08.11)
                                "ROI_Label",
                                "Mask_Transformation",
                                "Timepoint"], axis=1, errors='ignore')

            # Create a boolean array indicating which columns contain the string "diagnostics_" in the column name
            # features are for image configurations not interesting for transformed images
            cols_to_drop = list(tmp.columns[tmp.columns.str.contains('diagnostics_')])
        
        # Drop the columns containing the string "diagnostics_"
        tmp.drop(cols_to_drop, axis=1, inplace=True)

        # drop duplicated cols (e.g. "ID.1")
        for col in tmp.columns:
            if len(col) > len("ID"):
                if col.startswith("ID"):
                    tmp.drop(col, axis=1, inplace=True)

        exist = False
        # check if transformation is not already included
        for col in tmp.columns:
            if col.endswith("_" + transform):
                exist = True

        if not exist:
            # add suffix to column names to indicate the transformation
            tmp = tmp.add_suffix("_" + transform)

        tmp = tmp.loc[:, ~tmp.columns.duplicated(keep='first')]

        # print("Feature space for " + transform,"features:",tmp.shape)

        return tmp

    def add_transformations_as_suffix_to_image(self, transform):
        """
        Concatenate all transformations to one data frame where the kernel gets added to the name of the column
        as a suffix
        :param transform: name of transformation kernel which should be added
        :return: tmp: data frame with added transformation columns
        """

        tmp = self.df_wo_msk_img_trans.loc[self.df_wo_msk_img_trans["Image_Transformation"] == transform]

        if self.extractor == "MIRP":
            # Drop columns which are not specific for the transformation and add suffix to the columns
            if "id_subject" in tmp.columns:
                tmp = tmp.drop(["Image",
                                "Mask",
                                "Prediction_Label",
                                "Image_Transformation",
                                "Modality",
                                "ROI_Label",
                                "Mask_Transformation",
                                "Timepoint",
                                "Rater",
                                "id_subject",
                                "id_cohort",
                                "img_data_settings_id",
                                "img_data_modality"], axis=1, errors='ignore')
            else:
                tmp = tmp.drop(["Image",
                                "Mask",
                                "Prediction_Label",
                                "Image_Transformation",
                                "Modality",
                                "ROI_Label",
                                "Mask_Transformation",
                                "Timepoint",
                                "Rater",
                                "id_cohort",
                                "img_data_settings_id",
                                "img_data_modality"], axis=1, errors='ignore')

        elif self.extractor == "pyradiomics":
            tmp = tmp.drop(["Image",
                            "Mask",
                            "Prediction_Label",
                            "Image_Transformation",
                            "Modality",
                            "ROI_Label",
                            "Mask_Transformation",
                            "Timepoint"], axis=1, errors='ignore')

        # Create a boolean array indicating which columns contain the string "diagnostics_"
        # cols_to_drop = tmp.columns[tmp.columns.str.contains('diagnostics_')]
        # Drop the columns containing the string "diagnostics_"
        # tmp.drop(cols_to_drop, axis=1, inplace=True)

        exist = False
        # check if transformation is not already included
        for col in tmp.columns:
            if col.endswith("_" + transform):
                exist = True

        if not exist:
            # add suffix to column names to indicate the transformation
            tmp = tmp.add_suffix("_" + transform)

        tmp.set_index(self.df_mask_img_transform.index.name + "_" + transform, inplace=True)
        tmp.index.rename(self.df_mask_img_transform.index.name, inplace=True)

        return tmp

        ## 2.1 Get mean of Samples with multiple ROI

    def get_config_string(self, df: pd.DataFrame, string_parameter: pd.DataFrame):
        """
        Get a unique string for each sample incorporating all information which should be considered
        for the mean calculation (image transformation, mask transformation, timepoint and seg rater).
        :param string_parameter: dataframe with config information
        :param df: dataframe with features and no config information
        """
        self.logger.info("Generate config string for identifying multiple ROI per sample.")

        conf_list = []

        # synchronize index
        index_name = df.index.name
        if index_name in string_parameter.columns:
            string_parameter.set_index(index_name, inplace=True)
        else:
            index_name = string_parameter.index.name
            if index_name in df.columns:
                df.set_index(index_name, inplace=True)

        if len(conf_list) == 0:
            # Check for info features if they are nan insert 0 instead
            for i, row in df.iterrows():
                conf_string = str(os.path.basename(row["Image"])[:-len(".nii.gz")])
                
                if "Modality" in df.columns:
                    if not pd.isnull(row["Modality"]):
                        conf_string = conf_string + "_" + str(row["Modality"])
                    else:
                        conf_string = conf_string + "_0"
                else:
                    conf_string = conf_string + "_0"

                if "ROI_Label" in df.columns:
                    if not pd.isnull(row["ROI_Label"]):
                        conf_string = conf_string + "_" + str(row["ROI_Label"])
                    else:
                        conf_string = conf_string + "_0"
                else:
                    conf_string = conf_string + "_0"

                # if not pd.isnull(row["Image_Transformation"]):
                #    conf_string = conf_string + "_" + str(row["Image_Transformation"])
                # else:
                #    conf_string = conf_string + "_0"
                # if not pd.isnull(row["Mask_Transformation"]):
                #    conf_string = conf_string + "_" + str(row["Mask_Transformation"])
                # else:
                #    conf_string = conf_string + "_0"

                if "Timepoint" in df.columns:
                    if not pd.isnull(row["Timepoint"]):
                        conf_string = conf_string + "_" + str(row["Timepoint"])
                    else:
                        conf_string = conf_string + "_0"
                else:
                    conf_string = conf_string + "_0"

                if "Rater" in df.columns:
                    if not pd.isnull(row["Rater"]):
                        conf_string = conf_string + "_" + str(row["Rater"])
                    else:
                        conf_string = conf_string + "_0"
                else:
                    conf_string = conf_string + "_0"

                conf_list.append(conf_string)

        config = pd.Series(conf_list)

        # rename column if config already exists
        if "config" in df.columns:
            df = df.rename(columns={"config": "config_raw"}, errors="raise")

        df["config"] = config.values
        string_parameter["config"] = df["config"]

        return df, string_parameter

    def get_config_for_mean_per_config(self, df: pd.DataFrame, string_parameter: pd.DataFrame):
        """
        Get mean per config for samples with multiple ROIs and therefore multiple rows.
        :param string_parameter:
        :param df: Dataframe with features
        :return: Dataframe with mean per config features
        """

        df, string_parameter = self.get_config_string(df=df,
                                                      string_parameter=string_parameter)

        if df["config"].duplicated().sum() > 0:
            self.logger.info("Found Multiple ROIs: " + str(df["config"].duplicated().sum()))
            print("Found Multiple ROIs: " + str(df["config"].duplicated().sum()))
            

        self.logger.info("Corrected Sample Size: ")
        self.logger.info("Samples: " + str(df.shape[0]))
        self.logger.info("Features: " + str(df.shape[1]) + "\n")

        return df, string_parameter

    def split_column(self, col_name):
        """
        Split Parameter which are stored as string with three values.
        Converting to values with ending _x, _y, _z.
        :param col_name: column name to split in self.data
        """

        self.data[[col_name + '_x',
                   col_name + "_y",
                   col_name + "_z"]] = self.data[col_name].astype(str).str.split(", ", expand=True)

        # remove unwanted character
        self.data[col_name + '_x'] = self.data[col_name + '_x'].str.replace(r'(', '')
        self.data[col_name + '_z'] = self.data[col_name + '_z'].str.replace(r')', '')

        # remove previous column
        self.data.drop([col_name], axis=1, inplace=True)

        # convert column data type
        self.data[col_name + '_x'].str.replace(r' ', '').astype(float)
        self.data[col_name + '_y'].str.replace(r' ', '').astype(float)
        self.data[col_name + '_z'].str.replace(r' ', '').astype(float)

    def sepperate_string_list_features(self, feature:str, df:pd.DataFrame):
        """
        Sepperate Features which are stings in form of \"(value1,value2,value3)\" into columns for each value:
        :para feature: Name of Feature in feature space
        :para df: pd.DataFrame where feature is included
        :return: reformated pd.DataFrame
        """

        if feature in df.columns:
            df[[feature + '_x', feature + '_y',feature + '_z']] = self.data[feature].astype(str).str.split(", ", expand=True)  
            # remove unwanted character
            df[feature + '_x'] = df[feature + '_x'].str.replace(r'(', '')
            df[feature + '_z'] = df[feature + '_z'].str.replace(r')', '')

            try:
                df[feature + '_x'].astype(float)
            except:
                self.error.warning("Can not convert feature {} to number. Need to drop feature!".format(str(feature + '_x')))
                print("Can not convert feature {} to number. Need to drop feature!".format(str(feature + '_x')))
                df.drop([feature + '_x'], axis=1, inplace=True)

            try:
                df[feature + '_y'].astype(float)
            except:
                self.error.warning("Can not convert feature {} to number. Need to drop feature!".format(str(feature + '_y')))
                print("Can not convert feature {} to number. Need to drop feature!".format(str(feature + '_y')))
                df.drop([feature + '_y'], axis=1, inplace=True)

            try:
                df[feature + '_z'].astype(float)
            except:
                self.error.warning("Can not convert feature {} to number. Need to drop feature!".format(str(feature + '_z')))
                print("Can not convert feature {} to number. Need to drop feature!".format(str(feature + '_z')))
                df.drop([feature + '_z'], axis=1, inplace=True)

            df.drop([feature], axis=1, inplace=True)

        return df

    def separate_string_and_calculate_delta(self, df, feature):
        """
        Separates values in the format '(a,b,c)-(x,y,z)' into individual columns
        and computes deltas.
        
        :param feature: Name of the feature in the DataFrame
        :param df: Pandas DataFrame where the feature is included
        :return: Reformatted Pandas DataFrame
        """
        
        if feature in df.columns:
            # Extract the left and right values using regex
            df[[feature + '_left', feature + '_right']] = df[feature].astype(str).str.extract(r'\((.*?)\) - \((.*?)\)')

            # Split the values into individual columns
            df[[feature + '_x', feature + '_y', feature + '_z']] = df[feature + '_left'].str.split(',', expand=True)
            df[[feature + '_right_x', feature + '_right_y', feature + '_right_z']] = df[feature + '_right'].str.split(',', expand=True)

            # Clean unwanted characters
            df[feature + '_x'] = df[feature + '_x'].str.replace(r'[()]', '', regex=True).astype(float)
            df[feature + '_y'] = df[feature + '_y'].str.replace(r'[()]', '', regex=True).astype(float)
            df[feature + '_z'] = df[feature + '_z'].str.replace(r'[()]', '', regex=True).astype(float)

            df[feature + '_right_x'] = df[feature + '_right_x'].str.replace(r'[()]', '', regex=True).astype(float)
            df[feature + '_right_y'] = df[feature + '_right_y'].str.replace(r'[()]', '', regex=True).astype(float)
            df[feature + '_right_z'] = df[feature + '_right_z'].str.replace(r'[()]', '', regex=True).astype(float)

            # Calculate deltas
            df[feature + '_delta_x'] = df[feature + '_x'] - df[feature + '_right_x']
            df[feature + '_delta_y'] = df[feature + '_y'] - df[feature + '_right_y']
            df[feature + '_delta_z'] = df[feature + '_z'] - df[feature + '_right_z']

            # Drop unnecessary columns
            df.drop(columns=[feature, feature + '_left', feature + '_right', feature + '_right_x', feature + '_right_y', feature + '_right_z', feature + '_x', feature + '_y', feature + '_z'], inplace=True)

        return df

    def separate_and_calculate_delta_extended(self, df, feature):
        """
        Separates values in the format '(a1,a2,a3,a4,a5,a6)-(b1,b2,b3,b4,b5,b6)' into individual columns
        and computes deltas with appropriate postfixes (_1, _2, ..., _6).
        
        :param feature: Name of the feature in the DataFrame
        :param df: Pandas DataFrame where the feature is included
        :return: Reformatted Pandas DataFrame
        """
        
        if feature in df.columns:
            # Extract left and right values using regex
            df[[feature + '_left', feature + '_right']] = df[feature].astype(str).str.extract(r'\((.*?)\) - \((.*?)\)')
            
            # Split values into six separate columns
            left_columns = [f"{feature}_{i}" for i in range(1, 7)]
            right_columns = [f"{feature}_right_{i}" for i in range(1, 7)]
            delta_columns = [f"{feature}_delta_{i}" for i in range(1, 7)]

            df[left_columns] = df[feature + '_left'].str.split(',', expand=True).astype(float)
            df[right_columns] = df[feature + '_right'].str.split(',', expand=True).astype(float)

            # Calculate deltas
            for i in range(6):
                df[delta_columns[i]] = df[left_columns[i]] - df[right_columns[i]]

            # Drop unnecessary columns
            df.drop(columns=[feature, feature + '_left', feature + '_right'] + left_columns + right_columns, inplace=True)

        return df

    def config_pyradiomics_parameter(self):
        """
        Configure feature Space of PyRadiomics. Every parameter should be in one column and split multi parameter columns.
        self.data is the dataframe with unconfigured features from PyRadiomics extraction
        """

        # split ,multiple value columns
        self.data = self.sepperate_string_list_features(feature='diagnostics_Image-original_Spacing',df=self.data.copy())
        self.data = self.sepperate_string_list_features(feature='diagnostics_Mask-interpolated_CenterOfMass',df=self.data.copy())
        self.data = self.sepperate_string_list_features(feature='diagnostics_Mask-original_Spacing',df=self.data.copy())

        if self.delta:
            self.data = self.separate_string_and_calculate_delta(df=self.data, feature='diagnostics_Image-original_Size')
            self.data = self.separate_string_and_calculate_delta(df=self.data, feature='diagnostics_Mask-original_Size')
            self.data = self.separate_string_and_calculate_delta(df=self.data, feature='diagnostics_Mask-original_CenterOfMassIndex')
            self.data = self.separate_string_and_calculate_delta(df=self.data, feature='diagnostics_Mask-original_CenterOfMass')

            #self.logger.info("Need to drop some features for delta calculation: diagnostics_Image-original_Size, diagnostics_Mask-original_Size, diagnostics_Mask-original_CenterOfMassIndex, diagnostics_Mask-original_CenterOfMass")
            #print("Need to drop some features for delta calculation: diagnostics_Image-original_Size, diagnostics_Mask-original_Size, diagnostics_Mask-original_CenterOfMassIndex, diagnostics_Mask-original_CenterOfMass")
            #columns2drop = ['diagnostics_Image-original_Size', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass']
            #for col in columns2drop:
            #    if col in self.data.columns:
            #        self.data.drop(columns=[col], inplace=True)

        else:
            self.data = self.sepperate_string_list_features(feature='diagnostics_Image-original_Size',df=self.data.copy())
            self.data = self.sepperate_string_list_features(feature='diagnostics_Mask-original_Size',df=self.data.copy())
            self.data = self.sepperate_string_list_features(feature='diagnostics_Mask-original_CenterOfMassIndex',df=self.data.copy())
            self.data = self.sepperate_string_list_features(feature='diagnostics_Mask-original_CenterOfMass',df=self.data.copy())

        
        self.data = self.sepperate_string_list_features(feature='diagnostics_Image-interpolated_Spacing',df=self.data.copy())
        self.data = self.sepperate_string_list_features(feature='diagnostics_Image-interpolated_Size',df=self.data.copy())
        self.data = self.sepperate_string_list_features(feature='diagnostics_Mask-interpolated_CenterOfMassIndex',df=self.data.copy())
        
        
        if 'diagnostics_Mask-interpolated_BoundingBox'in self.data.columns:
            if self.delta:
                self.data = self.separate_and_calculate_delta_extended(df=self.data, feature='diagnostics_Mask-interpolated_BoundingBox')
            else:
                # split one column contains a string into three
                self.data[['diagnostics_Mask-interpolated_BoundingBox_1',
                        "diagnostics_Mask-interpolated_BoundingBox_2",
                        "diagnostics_Mask-interpolated_BoundingBox_3",
                        'diagnostics_Mask-interpolated_BoundingBox_4',
                        'diagnostics_Mask-interpolated_BoundingBox_5',
                        'diagnostics_Mask-interpolated_BoundingBox_6']] = self.data[
                    'diagnostics_Mask-interpolated_BoundingBox'].astype(
                    str).str.split(", ", expand=True)

                # remove unwanted character
                self.data['diagnostics_Mask-interpolated_BoundingBox_1'] = self.data[
                    'diagnostics_Mask-interpolated_BoundingBox_1'].str.replace(
                    r'(', '')
                self.data['diagnostics_Mask-interpolated_BoundingBox_6'] = self.data[
                    'diagnostics_Mask-interpolated_BoundingBox_6'].str.replace(
                    r')', '')

                # remove previous column
                self.data.drop(['diagnostics_Mask-interpolated_BoundingBox'], axis=1, inplace=True)

                # convert column data type
                self.data[['diagnostics_Mask-interpolated_BoundingBox_1',
                        "diagnostics_Mask-interpolated_BoundingBox_2",
                        "diagnostics_Mask-interpolated_BoundingBox_3",
                        'diagnostics_Mask-interpolated_BoundingBox_4',
                        'diagnostics_Mask-interpolated_BoundingBox_5',
                        'diagnostics_Mask-interpolated_BoundingBox_6']].astype(float)

        if 'diagnostics_Mask-original_BoundingBox' in self.data.columns:
            if self.delta:
                self.data = self.separate_and_calculate_delta_extended(df=self.data, feature='diagnostics_Mask-original_BoundingBox')
            else:
                # split one column contains a string into three
                self.data[['diagnostics_Mask-original_BoundingBox_1',
                        "diagnostics_Mask-original_BoundingBox_2",
                        "diagnostics_Mask-original_BoundingBox_3",
                        'diagnostics_Mask-original_BoundingBox_4',
                        'diagnostics_Mask-original_BoundingBox_5',
                        'diagnostics_Mask-original_BoundingBox_6']] = self.data[
                    'diagnostics_Mask-original_BoundingBox'].astype(
                    str).str.split(", ", expand=True)

                # remove unwanted character
                self.data['diagnostics_Mask-original_BoundingBox_1'] = self.data[
                    'diagnostics_Mask-original_BoundingBox_1'].str.replace(
                    r'(', '')
                self.data['diagnostics_Mask-original_BoundingBox_6'] = self.data[
                    'diagnostics_Mask-original_BoundingBox_6'].str.replace(
                    r')', '')

                # remove previous column
                self.data.drop(['diagnostics_Mask-original_BoundingBox'], axis=1, inplace=True)

                # convert column data type
                self.data[['diagnostics_Mask-original_BoundingBox_1',
                        "diagnostics_Mask-original_BoundingBox_2",
                        "diagnostics_Mask-original_BoundingBox_3",
                        'diagnostics_Mask-original_BoundingBox_4',
                        'diagnostics_Mask-original_BoundingBox_5',
                        'diagnostics_Mask-original_BoundingBox_6']].astype(float)

        if 'diagnostics_Mask-resegmented_BoundingBox' in self.data.columns:
            if self.delta:
                self.data = self.separate_and_calculate_delta_extended( df=self.data, feature='diagnostics_Mask-resegmented_BoundingBox')
            else:
                # split one column contains a string into three
                self.data[['diagnostics_Mask-resegmented_BoundingBox_1',
                        "diagnostics_Mask-resegmented_BoundingBox_2",
                        "diagnostics_Mask-resegmented_BoundingBox_3",
                        'diagnostics_Mask-resegmented_BoundingBox_4',
                        'diagnostics_Mask-resegmented_BoundingBox_5',
                        'diagnostics_Mask-resegmented_BoundingBox_6']] = self.data[
                    'diagnostics_Mask-resegmented_BoundingBox'].astype(
                    str).str.split(", ", expand=True)

                # remove unwanted character
                self.data['diagnostics_Mask-resegmented_BoundingBox_1'] = self.data[
                    'diagnostics_Mask-resegmented_BoundingBox_1'].str.replace(
                    r'(', '')
                self.data['diagnostics_Mask-resegmented_BoundingBox_6'] = self.data[
                    'diagnostics_Mask-resegmented_BoundingBox_6'].str.replace(
                    r')', '')

                # remove previous column
                self.data.drop(['diagnostics_Mask-resegmented_BoundingBox'], axis=1, inplace=True)

                # convert column data type
                self.data[['diagnostics_Mask-resegmented_BoundingBox_1',
                        "diagnostics_Mask-resegmented_BoundingBox_2",
                        "diagnostics_Mask-resegmented_BoundingBox_3",
                        'diagnostics_Mask-resegmented_BoundingBox_4',
                        'diagnostics_Mask-resegmented_BoundingBox_5',
                        'diagnostics_Mask-resegmented_BoundingBox_6']].astype(float)

        if 'diagnostics_Mask-resegmented_Spacing' in self.data.columns:
            self.split_column("diagnostics_Mask-resegmented_Spacing")

        if "diagnostics_Mask-resegmented_Size" in self.data.columns:
            if self.delta:
                self.data = self.separate_string_and_calculate_delta(df=self.data, feature="diagnostics_Mask-resegmented_Size")
            else:
                self.split_column("diagnostics_Mask-resegmented_Size")

        if "diagnostics_Mask-resegmented_CenterOfMassIndex" in self.data.columns:
            if self.delta:
                self.data = self.separate_string_and_calculate_delta(df=self.data, feature="diagnostics_Mask-resegmented_CenterOfMassIndex")
            else:
                self.split_column("diagnostics_Mask-resegmented_CenterOfMassIndex")

        if "diagnostics_Mask-resegmented_CenterOfMass" in self.data.columns:
            if self.delta:
                self.data = self.separate_string_and_calculate_delta(df=self.data, feature="diagnostics_Mask-resegmented_CenterOfMass")
            else:
                self.split_column("diagnostics_Mask-resegmented_CenterOfMass")

    def config_pyradiomics_features(self):
        """
        This function is used to configure the pyradiomics features and make each feature a number
        """

        # drop columns with strings
        self.data = self.data[self.data.columns[~(self.data.columns.str.endswith('original_Hash'))]]
        self.data = self.data[self.data.columns[~(self.data.columns.str.startswith('diagnostics_Versions'))]]
        self.data = self.data[self.data.columns[~(self.data.columns.str.startswith('diagnostics_Configuration'))]]

        if 'diagnostics_Image-original_Dimensionality' in self.data.columns:
            self.data.drop(['diagnostics_Image-original_Dimensionality'], axis=1, inplace=True)

        self.logger.info("###  Data overview after remove Hashes and versions###")
        self.logger.info("Input Data contains:")
        self.logger.info(str(self.data.shape[0]) + " Samples")
        self.logger.info(str(self.data.shape[1]) + " Features\n")

        # check if name of feature is included as value in data frame
        dropping_samples = []

        target = self.data.loc[self.data[self.data.columns[0]] == self.data.columns[0], :]
        if len(target) > 0:
            self.error.warning("Feature name included as value in data frame!")
            for tar in target.index:
                # drop sample with feature name as value
                dropping_samples.append(tar)

        # drop all rows which contain the feature names as values in the data frame
        if len(dropping_samples) > 0:
            self.error.warning("Drop samples with feature name as value in data frame")
            self.data = self.data.drop(list(set(dropping_samples)))

            # drop duplicated samples
            self.data = self.data.drop_duplicates(["Image", "Mask"])

        # 3. make multiple feature columns in one each
        self.config_pyradiomics_parameter()

        self.logger.info("### 1. Config PyRadiomics features ###")
        self.logger.info("Corrected Sample Size:")
        self.logger.info(str(self.data.shape[0]) + " Samples")
        self.logger.info(str(self.data.shape[1]) + " Features\n")

    def outsource_string_columns(self, df: pd.DataFrame, string_parameter: pd.DataFrame):
        """
        Outsource string columns from data to string_parameter
        :param df: dataframe to be processed with string columns
        :param string_parameter: dataframe to store string columns
        :return: dataframe without string columns and string_parameter with string columns
        """

        # drop ROI column of dataframe - if it is included
        if "ROI" in df.columns:
            df.drop(["ROI"], axis=1, inplace=True)
        else:
            self.error.warning("No ROI column in dataframe")

        if "ID" in df.columns:
            # self.data.index = self.data["ID"]
            if "ID" not in string_parameter.columns:
                string_parameter["ID"] = df["ID"]
            df.drop(["ID"], axis=1, inplace=True)
        else:
            self.error.warning("No ID column in dataframe")

        # If only the same modality is present
        if "Modality" in df.columns:
            if "Modality" not in string_parameter.columns:
                string_parameter["Modality"] = df["Modality"]
            df.drop(["Modality"], axis=1, inplace=True)
        else:
            self.error.error("No Modality column in dataframe")

        if "ROI_Label" in df.columns:
            if "ROI_Label" not in string_parameter.columns:
                string_parameter["ROI_Label"] = df["ROI_Label"]
            df.drop(["ROI_Label"], axis=1, inplace=True)
        else:
            self.error.warning("No ROI_Label column in dataframe")

        if "Timepoint" in df.columns:
            if "Timepoint" not in string_parameter.columns:
                string_parameter["Timepoint"] = df["Timepoint"]
            df.drop(["Timepoint"], axis=1, inplace=True)
        else:
            self.error.warning("No Timepoint column in dataframe")

        if "Prediction_Label" in df.columns:
            if "Prediction_Label" not in string_parameter.columns:
                string_parameter["Prediction_Label"] = df["Prediction_Label"]
            df.drop(["Prediction_Label"], axis=1, inplace=True)
        else:
            self.error.error("No Prediction_Label column in dataframe")

        if "Image" in df.columns:
            if "Image" not in string_parameter.columns:
                string_parameter["Image"] = df["Image"]
            # self.data.drop(["Image"], axis=1, inplace=True)
        else:
            self.error.error("No Image column in dataframe")

        if "Mask" in df.columns:
            if "Mask" not in string_parameter.columns:
                string_parameter["Mask"] = df["Mask"]
            # self.data.drop(["Mask"], axis=1, inplace=True)
        else:
            self.error.error("No Mask column in dataframe")

        if "Mask_Transformation" in df.columns:
            if "Mask_Transformation" not in string_parameter.columns:
                string_parameter["Mask_Transformation"] = df["Mask_Transformation"]
            df.drop(["Mask_Transformation"], axis=1, inplace=True)
        else:
            self.error.warning("No Mask_Transformation column in dataframe")

        if "Image_Transformation" in df.columns:
            if "Image_Transformation" not in string_parameter.columns:
                string_parameter["Image_Transformation"] = df["Image_Transformation"]
            df.drop(["Image_Transformation"], axis=1, inplace=True)
        else:
            self.error.warning("No Image_Transformation column in dataframe")

        if "Rater" in df.columns:
            if "Rater" not in string_parameter.columns:
                string_parameter["Rater"] = df["Rater"]
            df.drop(["Rater"], axis=1, inplace=True)
        else:
            self.error.error("No Rater column in dataframe")

        self.logger.info("Corrected Sample Size: ")
        self.logger.info("Samples: " + str(df.shape[0]))
        self.logger.info("Features: " + str(df.shape[1]) + "\n")

        return df, string_parameter

    def add_configs(self):
        """
        Mean multiple ROIS and get config string for identification of unique samples and
        :return:
        """

        # synchronize index
        # if not self.data.index.name == "ID":
        #    self.data = self.data.set_index("ID")

        # if not self.df_mask_transform.index.name == "ID":
        #    self.df_mask_transform = self.df_mask_transform.set_index("ID")

        if not os.path.exists(os.path.dirname(self.out_path) + "/tmp/"):
            os.makedirs(os.path.dirname(self.out_path) + "/tmp/")

        # self.df_mask_transform.to_csv(os.path.dirname(self.out_path) +
        #                               "/tmp/" + self.RunID + "_df_mask_transform.csv")
        # self.data.to_csv(os.path.dirname(self.out_path) + "/tmp/" + self.RunID + "_data.csv")

        self.logger.info("Write Radiomics Filter config to csv: " + str(os.path.dirname(self.out_path) + "/" + self.RunID + "_Radiomics_Filter_config.csv"))

        # self.data.to_csv(self.out_path + self.RunID + "_Radiomics_Filter_config.csv")

        # if self.extractor == "MIRP":
        #    self.config_mirp_features()
        #TODO: Configure mean calculation --- but it seems like it is not happening!
        self.logger.info("Generating Config for multiple ROIs per Sample ...")
        print("Generating Config for multiple ROIs per Sample ...")
        self.data, self.string_parameter = self.get_config_for_mean_per_config(
                                                                            df=self.data.copy(),
                                                                            string_parameter=self.string_parameter.copy())

        self.df_mask_transform, self.string_parameter_msk = self.get_config_for_mean_per_config(
                                                                                            df=self.df_mask_transform.copy(),
                                                                                            string_parameter=self.string_parameter.copy())

        self.logger.info("Extract configuration information ...")
        self.data, self.string_parameter = self.outsource_string_columns(
                                                                        df=self.data,
                                                                        string_parameter=self.string_parameter)

        return self.data, self.string_parameter


    def drop_constant_column(self, df: pd.DataFrame):
        """
        Drops constant value columns of pandas dataframe.
        """

        const_cols = []

        # pre = self.data.shape[1]
        # self.data = self.data.loc[:, (self.data != self.data.iloc[0]).any()]

        for col in df.columns:
            if df[col].nunique() == 1:
                df.drop(col, axis=1, inplace=True)
                const_cols.append(col)
        
        if len(const_cols) > 0:
            self.logger.info(f"Dropped {str(len(const_cols))} Constant Features: " + str(const_cols))
            print(f"Dropped {str(len(const_cols))} Constant Features: " + str(const_cols))

        if len(const_cols) > 0:
            for col in const_cols:
                if col == const_cols[-1]:
                    self.logger.info("Dropped Constant Feature " + str(col) + "\n")
                else:
                    self.logger.info("Dropped Constant Feature " + str(col))

        self.logger.info("Corrected Sample Size: ")
        self.logger.info("Samples: " + str(df.shape[0]))
        self.logger.info("Features: " + str(df.shape[1]) + "\n")

        return df

    def search_non_numeric_features(self, df: pd.DataFrame):
        """
        :parameter: df: dataframe to be searched for non-numeric features
        :return: string_cols: list of non-numeric features
        :return: df: dataframe with numeric features
        """

        string_cols = []

        # Scan for strings in dataframe
        for column in df.columns:
            if df[column].dtype == object:
                if column in self.config_features:
                    string_cols.append(column)
                elif column != "config":
                    try:
                        if is_string_dtype(df[column]):
                            df[column] = df[column].str.replace(',','.')
                            
                        df[column] = df[column].astype(float)
                    except Exception as e:
                        # self.data[column] = self.data[column].astype(str)
                        self.error.warning("Non-numeric Feature: " + column + " " + str(e))
                        string_cols.append(column)
                    # df['column_name'] = df['column_name'].astype(float)
                    # string_cols.append(column)
            elif column in self.config_features:
                string_cols.append(column)

        # self.logger.info("Found " + str(len(string_cols)) + " Non-numeric Features in Data.")

        # for col in string_cols:
        #     if col == string_cols[-1]:
        #         self.logger.info("Found Non-numeric Feature: " + col + "\n")
        #     else:
        #         self.logger.info("Found Non-numeric Feature: " + col)

        # self.logger.info("Calculate Mean for these Samples..." + "\n")
        string_df = df[string_cols].copy()

        if "config" in string_df.columns:
            string_df["config"] = df["config"]

        # self.string_df["config"] = self.data["config"]
        df = df.drop(string_cols, axis=1)

        self.logger.info("Corrected Sample Size:")
        self.logger.info(str(df.shape[0]) + " Samples")
        self.logger.info(str(df.shape[1]) + " Features\n")

        return df, string_df

    def check_config(self, df:pd.DataFrame):
        """
        Check existing configuration of data
        """

        existing_config_cols =[]

        # checking completeness of config features
        for config in self.config_features:
            if config not in df.columns:
                self.error.warning("Config Feature " + config + " not found in Data.")
            else:
                existing_config_cols.append(config)
        
        for config in ["Image_Transformation", "Mask_Transformation"]:
            if config not in df.columns:
                self.error.warning("Config Feature " + config + " not found in String Parameter.")
            else:
                existing_config_cols.append(config)

        # checking for config column order
        # if Image is not in the first 5 columns, move it to the first 5 columns
        if "Image" not in df.iloc[:, 0:5].columns:
            df = df.copy()[existing_config_cols + [c for c in df.columns if c not in existing_config_cols]]

        df = df.copy().loc[:,~df.columns.duplicated()].copy()

        # if duplciated columns
        if df.columns.duplicated().sum() > 0:
            df = df.copy().loc[:,~df.columns.duplicated()]

        if df.index.name == "ID":
            if "ID" in df.columns:
                df.drop(["ID"], axis=1, inplace=True)

        if "ID" in df.columns:
            df.sort_values(by='ID', inplace=True)

        # convert 0 to nan in Image_Transformation and Mask_Transformation is it is there
        if "Image_Transformation" not in df.columns:
            self.error.warning("No Image_Transformation found in Data.")
            print("Warning - No Image_Transformation found in Data.")

        if "Mask_Transformation" not in df.columns:
            if "Rater" in df.columns:
                df["Mask_Transformation"] = df["Rater"]
                self.error.warning("Mask_Transformation missing in Data. Replacing Mask_Transformation with Rater.")
                print("Mask_Transformation missing in Data. Replacing Mask_Transformation with Rater.")
            else:
                self.error.warning("No Mask_Transformation found in Data.")
                print("No Mask_Transformation found in Data.")

        return df

    def config(self):
        """
        Main function for data configuration anst stability filtering
        """

        self.logger.info("### Configuring Data from Feature Extractor ###")
        print("Configure extracted Features ...")

        self.data = self.check_config(df=self.data.copy())

        # Remove non preprocessed samples from raw samples
        self.data = self.remove_duplcated_samples_without_preprocessed_seg(self.data.copy(), img_trans="nan", mask_trans="nan")
        if "Image_Transformation" in self.data.columns:
            pbar = tqdm(self.data["Image_Transformation"].unique(), desc='Remove duplications from Transformations')

            # Remove non preprocessed samples from Image transformation samples
            for trans in pbar: #tqdm(df.copy()["Image_Transformation"].unique(), desc="Remove duplications from Transformations"):
                pbar.set_description("Remove duplications from {} Transformations".format(trans))
                self.data = self.remove_duplcated_samples_without_preprocessed_seg(self.data.copy(), img_trans=trans, mask_trans="nan")

        print("Reduced Feature space", self.data.shape)

        # 1. Remove corrupted columns from extraction
        self.configure_columns()
        # self.data.to_csv(os.path.dirname(self.out_path) + "/after_configure_columns.csv")

        # 2. Separate Data for multiple ROIs or Time-points
        self.config_ROI_and_Timepoint()

        # 3. Add configuration of data to extracted features
        self.add_config_to_df()

        if self.extractor == "MIRP":

            # check if name of feature is included as value in DataFrame
            dropping_samples = []

            target = self.data.loc[self.data[self.data.columns[0]] == self.data.columns[0], :]
            if len(target) > 0:
                self.error.warning("Feature name included as value in data!")
                for tar in target.index:
                    # drop sample with feature name as value
                    dropping_samples.append(tar)

            # drop all rows which contain the feature names as values in the data
            if len(dropping_samples) > 0:
                self.error.warning("Drop samples with feature name as value in data")
                self.data = self.data.drop(list(set(dropping_samples)))

            # drop duplicated samples
            self.data = self.data.drop_duplicates(["Image", "Mask"])

            if "img_data_roi" not in self.data.columns:
                if self.data.index.name != "id_subject":
                    if "id_subject" not in self.data.columns:
                        self.data = self.set_unique_index(self.data.copy())
                        self.data.index.name = "id_subject"
                    else:
                        self.data = self.data.set_index("id_subject")
            else:
                self.data = self.data.set_index("img_data_roi")
            self.string_parameter.index = self.data.index

        if self.extractor == "PyRadiomics":
            self.config_pyradiomics_features()

        self.data.sort_values(by=['ID'], inplace=True)

        # check if column names are also in the values
        if "ID" in self.data.columns:
            if "ID" in self.data["ID"].to_list():
                self.logger.info("Found columns names in Features! Remove them...")
                print("Found columns names in Features ID! Remove them...")
                self.data = self.data[self.data["ID"] != "ID"]

        # I hate myself
        # TODO: Need to configure delta features after feature configuration here
        # self.data.to_csv(self.out_path + "configured_data.csv")

        # 4. Add transformed Features as columns to effected sample and separate mask transformations
        self.data, self.df_mask_transform = self.extend_data_for_transformations(df=self.data.copy()) 
        self.string_parameter.index = self.data.index
        self.data, self.string_parameter = self.outsource_string_columns(df=self.data.copy(),
                                                                        string_parameter=self.string_parameter.copy())
                                                                        
        # self.string_parameter.to_csv(os.path.dirname(self.out_path) + "/String_parameter_001.csv")
        # print("11", self.data.shape)
        self.config_feature_size = self.data.shape[1]
        # self.data.to_csv(self.out_path + "configured_feature_space.csv")
        self.data = self.data.dropna(axis=1, how='all')
        self.df_mask_transform = self.df_mask_transform.dropna(axis=1, how='all')

        feature_formatter = FeatureFormatter(features=self.data.columns.to_list(),
                                             extractor=self.extractor,
                                             logger=self.logger,
                                             error=self.error,
                                             additional_ROIs=self.additional_ROIs,
                                             output_path=os.path.dirname(self.out_path))
                                             
        print("Generate General Feature Profile ...")
        self.logger.info("Generate General Feature Profile ...")
        
        # clean feature names
        self.data.columns = [s.replace("_zscore", "") for s in self.data.copy().columns.to_list()]

        formatted_features = feature_formatter.exe(title="General_Feature_Profile")
        formatted_features.to_csv(self.out_path + "/" + self.extractor + "_General_Feature_Profile.csv", index=False)

        # 4. Feature Stability Filtering
        if self.stability_filtering:

            #self.df_mask_transform.to_csv(self.out_path + "tmp/df_mask_transform.csv")
            
            if "Mask_Transformation" in self.df_mask_transform.columns:

                if len(set(self.df_mask_transform["Mask_Transformation"].to_list())) < 5:
                    self.logger.info("Too few mask transformations. Trying to extend radom walker trials ...")
                    msks = self.df_mask_transform.loc[self.df_mask_transform["Mask_Transformation"] == "Random_Walker_Mask_Change"]["Mask"].to_list()
                    for msk in msks:
                        match = re.search("_random_walker_([0-9]*)_", msk)
                        if not match is None:
                            self.df_mask_transform.loc[self.df_mask_transform["Mask"] == msk, "Mask_Transformation"] = self.df_mask_transform.loc[self.df_mask_transform["Mask"] == msk, "Mask_Transformation"] + "_" + str(match.group(1))
                
                # self.df_mask_transform.to_csv(self.out_path + "df_mask_transform_ext_transform.csv")

                if not os.path.exists(str(os.path.dirname(self.out_path)) + "/unstable_features_threshold_" + str(self.ICC_threshold)  + "_" + str(self.RunID) + ".csv"):
                    stability_filter =  FeatureStabilityFilter(
                                                        #logger=self.logger,
                                                        #error=self.error,
                                                        extractor=self.extractor,
                                                        df_data=self.df_mask_transform,
                                                        ICC_threshold=self.ICC_threshold,
                                                        RunID=self.RunID,
                                                        out_path=os.path.dirname(self.out_path)
                                                        )
                    unstable_features = stability_filter.exe()
                else:
                    unstable_features = pd.read_csv(str(os.path.dirname(self.out_path)) + "/unstable_features_threshold_" + str(self.ICC_threshold)  + "_" + str(self.RunID) + ".csv")["unstable_features"].to_list()

                if len(unstable_features) > 0:
                    self.logger.info("Unstable Features: " + str(len(unstable_features)))
                    print("Found " + str(len(unstable_features)) + " Unstable Features.")

                    # drop unstable features
                    for unstable in unstable_features:
                        if unstable in self.data.columns:
                            self.data.drop(unstable, axis=1, inplace=True)

                    # write the result from feature stability filtering
                    unstable_features = pd.DataFrame({"Unstable_features": unstable_features})# 
                    unstable_features.to_csv(self.out_path + "unstable_features.csv")

                    feature_formatter = FeatureFormatter(features=unstable_features,
                                                        extractor=self.extractor,
                                                        logger=self.logger,
                                                        error=self.error,
                                                        output_path=self.out_path)

                    formatted_features = feature_formatter.exe(title="Unstable_Feature_Profile")
                    formatted_features.to_csv(self.out_path + self.extractor + "_Unstable_Feature_Profile.csv", index=False)
                else: 
                    print("No unstable features found!")
                    self.logger.info("No unstable features found!")
            else:
                self.error.warning("No Mask_Transformation found in Data. Can not perform Feature Stability Filtering.")
                print("Warning - No Mask_Transformation found in Data. Can not perform Feature Stability Filtering.")
                # raise ValueError("No Mask_Transformation found in Data. Can not perform Feature Stability Filtering.")
                self.stability_filtering = False

                # No MAsk transfomration should be in the the data anymore
                if "Mask_Transformation" in self.data.columns:
                    # no instability filtering menas we do not need mask transformations
                    self.data = self.data[self.data['Mask_Transformation'].isin(['Peritumoral']) | self.data['Mask_Transformation'].isna()]

                feature_formatter = FeatureFormatter(features=self.data.columns.to_list(),
                                                    extractor=self.extractor,
                                                    logger=self.logger,
                                                    error=self.error,
                                                    output_path=self.out_path)
                                                    
                formatted_features = feature_formatter.exe(title="General_Feature_Profile")
                formatted_features.to_csv(self.out_path + self.extractor + "_General_Feature_Profile.csv", index=False)

        self.stable_feature_size = self.data.shape[1]
        
        # 5. Create unique sample identifier
        self.logger.info("### 5. Create unique sample identifier ###")
        print("### 5. Create unique sample identifier ###")


        self.data, self.string_parameter = self.add_configs()
        # self.string_parameter.to_csv(os.path.dirname(self.out_path) + "/String_parameter_tmp_1.csv")

        # 6. Drop Features which are not changing at all
        self.logger.info("### 6. Drop constant Features ###")
        print("### 6. Drop constant Features ###")
        self.data = self.drop_constant_column(df=self.data)

        self.non_constant_feature_size = self.data.shape[1]
        
        # 7. Extract non-numeric Features and save them in a separate file
        self.logger.info("### 7. Search for non-numeric Features ###")
        print("### 7. Search for non-numeric Features ###")
        self.data, string_df = self.search_non_numeric_features(df=self.data)

        # self.string_parameter.to_csv(os.path.dirname(self.out_path) + "/String_parameter_tmp.csv")

        self.string_parameter = pd.concat([self.string_parameter, string_df], axis=1)
        
        # drop duplicated rows
        self.string_parameter = self.string_parameter.drop_duplicates()
        
        self.logger.info("### Configured Numeric Radiomics Data ###")
        self.logger.info(str(self.data.shape[0]) + " Samples")
        self.logger.info(str(self.data.shape[1]) + " Features" + "\n")

        self.logger.info("### Configured String Radiomics Data ###")
        self.logger.info(str(self.string_parameter.shape[0]) + " Samples")
        self.logger.info(str(self.string_parameter.shape[1]) + " Features" + "\n")

        # self.string_parameter.to_csv(os.path.dirname(self.out_path) + "/String_parameter.csv")

        # Separate Mask Transformations -- including config string and excluding string_parameter
        # self.data, self.df_mask_transform = self.seperate_mask_transformations(df=self.data)
        # string_parameter_mask_transform = self.string_parameter.copy()

        # self.logger.info("Drop Constant Features from mask augmentation:")
        # self.df_mask_transform = self.drop_constant_column(df=self.df_mask_transform)

        # self.logger.info("### 5. Searching for non-numeric Features ###")

        # self.logger.info("Drop Non-numeric Features from mask augmentation:")
        # self.df_mask_transform, string_df_mask = self.search_non_numeric_features(df=self.df_mask_transform)
        # string_df_mask = pd.concat([string_df_mask, string_parameter_mask_transform])

        return self.data, self.string_parameter
