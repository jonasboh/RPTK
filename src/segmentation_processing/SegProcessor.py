# +

import numpy as np
import os
import tqdm
from multiprocessing.pool import Pool
import concurrent.futures as cf
import random
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

import SimpleITK as sitk
from skimage.measure import label
from skimage.measure import regionprops
import skimage.morphology
import argparse
import operator
import path
import time
import datetime
import logging
import glob
from pandas.api.types import is_string_dtype

from sklearn.metrics import jaccard_score
from sklearn.preprocessing import Normalizer
from skimage.segmentation import expand_labels

from statistics import mean
import pandas as pd
from pathlib import Path  # handle path dirs
from tqdm.contrib.concurrent import process_map  # parallel processing bar
from functools import partial
import nibabel as nib

from sklearn.metrics import confusion_matrix

from concurrent.futures import ThreadPoolExecutor, as_completed

import gzip
import shutil
import re
from typing import Union

from rptk.mirp import *
from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.src.image_processing.Resampler import Resampler
from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.segmentation_processing.SegmentationFilters import SegmentationFilter
from rptk.src.segmentation_processing.Segmentation_perturbator import CCExtender, SupervoxelPerturbator, \
    PeriTumoralSegmentator, RandomWalker
from rptk.src.image_processing.Transform_Executer import Executor
from rptk.src.image_processing.MR_Normalizer import MR_Normalizer

from rptk import rptk


# -


# # + Modality
# # + Label --> Needed
# # + ID --> default Image Name + Seg Name
# # + timepoint --> default 0
# # + Image configuration (Reconstructionkernel,Phantom,cropped Scan) --> default 0
# # + Seg Rater --> default: 0
# # + Object description --> default: 0

# TODO: Liver segmentation atifacts are not removed --> Why?
# TODO: Config File generation --> see above
# TODO: Add Segmentation simulation with DICE checkup

class SegProcessor:
    """
    Class for processing segmentations and images. Including filtering of segmentations for small ROIs,
    image transformation, mask perturbation, peritumoral segmentation, normalization, and resampling.

    :param segPath: Path to segmentation
    :param imgPath: Path to image
    :param modality: Modality of image
    :param out_path: Path to output folder
    :param path2csv: Path to csv file
    :param kernels: List of kernels
    :param RunID: RunID
    :param chunksize: Chunksize for multiprocessing
    :param isotropic_scale: Isotropic scale for resampling
    :param normalization: Normalize images
    :param normalization_method: Normalization method
    :param segmentation_perturbation: do seg perturbation
    :param image_transformation: do img transformation
    :param perturbation_method: Perturbation method
    :param resampling: Resample images
    :param roi_threshold: Threshold for ROI selection
    :param max_num_rois: Maximum number of ROIs
    :param n_cpu: Number of CPUs
    :param perturbation_roi_adapt_size: Perturbation size for ROI
    :param perturbation_roi_adapt_type: Perturbation type for ROI
    :param roi_pert_repetition: Number of perturbations per sample
    :param label_of_interest: Label of interest
    :param dice_threshold: Dice threshold
    :param peritumoral_seg: Peritumoral segmentation
    :param peri_dist: Peritumoral distance
    :param expand_seg_dist: Distance for segmentation expansion
    :param perturbation_factor: factor to normalize the perturbation impact for small and large ROIs
    :param seg_closing_radius: Radius for segmentation closing
    :param logger: Logger
    :param error: logger for errors
    :param log_file_path: Path to log file
    """

    def __init__(self,
                 rptk_config_json: str = None,
                 segPath: str = "",
                 imgPath: str = "",  # if single files or a folder with same modality
                 modality: str = "",  # Modality if single img and mask are provided or multiple in folder
                 out_path: str = "",
                 path2csv: str = "",
                 kernels=None,
                 RunID: str = "",
                 chunksize: int = 10,  # Chunk size for multiprocessing
                 isotropic_scale: float = 1.,
                 normalization: Union[bool, str] = "auto",  # Normalize MRI Images (yes/no) default yes
                 normalization_method=None,  # Normalization method: [z_score, n4bf]
                 segmentation_perturbation: bool = True,  # Whether to execute segmentation perturbation or not
                 image_transformation: bool = True,  # Whether to execute image transformation or not
                 perturbation_method=None,  # Perturbation method: super voxel, connected_component, random_walker
                 resampling: bool = False,  # Resample Images and Segmentations
                 roi_threshold: int = 3,  # Threshold for roi selection ignoring number of voxels
                 max_num_rois: int = 1,  # How many Rois you allow per sample/ mask
                 n_cpu: int = os.cpu_count() - 1,  # CPU used for multi threading
                 perturbation_roi_adapt_size: list = None,  # Perturbation size for roi
                 perturbation_roi_adapt_type: str = "distance",  # "fraction"
                 roi_pert_repetition: int = 3,
                 # How many times the roi perturbation is repeated (Super voxel or random walker)
                 label_of_interest: int = 1,  # Label which will be considered for segmentation
                 dice_threshold: float = 0.90,  # Dice threshold for segmentation acceptance
                 peritumoral_seg: bool = True,  # Peritumoral segmentation (yes/no)
                 peri_dist: list = None,  # Distance for surrounding region: default 3mm
                 expand_seg_dist=None,  # Distance for expand segmentation: default 1
                 perturbation_factor: float = 0.05,  # Perturbation factor: default 0.05
                 seg_closing_radius: int = 10,  # Segmentation closing radius: default 10
                 logger=None,  # logger for info
                 error=None,  # logger for errors
                 log_file_path=None,  # Path to log file
                 consider_multiple_labels: bool = False,  # Consider multiple labels from segmentation (not only 1)
                 use_previous_output: bool = False,  # Use output from previous run
                 self_optimize: bool = False,  # Optimizes the parameters for better feature extraction
                 mean_slice_thickness: int = 0,
                 resample_slice_thickness_threshold: int = 6,
                 fast_mode : bool = False,  # Mode for slow processing but saving memory
                 timeout : int = 500,  #  Time [s] until to wait for perturbation processing
                 seed : int = 1234,  # Seed for random perturbation
                 pertubration_closing_correction: bool = False,  # Correct the perturbation with closing
                 pertubration_convex_correction: bool = False,  # Correct the perturbation with convex hull
                 ):
        # self optimization parameters:
        # dice_threshold
        # roi_threshold

        self.rptk_config_json = rptk_config_json
        self.segPath = segPath
        self.imgPath = imgPath
        self.out_path = out_path
        self.max_num_rois = max_num_rois
        self.n_cpu = n_cpu
        self.label_of_interest = label_of_interest  # Label which will be considered for
        self.RunID = RunID
        self.path2csv = path2csv
        self.perturbation_roi_adapt_size = perturbation_roi_adapt_size
        self.perturbation_roi_adapt_type = perturbation_roi_adapt_type
        self.roi_pert_repetition = roi_pert_repetition
        self.peritumoral_seg = peritumoral_seg
        self.peri_dist = peri_dist
        self.expand_seg_dist = expand_seg_dist
        self.modality = modality
        self.normalization = normalization
        self.normalization_method = normalization_method
        self.segmentation_perturbation = segmentation_perturbation
        self.image_transformation = image_transformation
        self.perturbation_method = perturbation_method
        self.resampling = resampling
        self.perturbation_factor = perturbation_factor
        self.kernels = kernels
        self.seg_closing_radius = seg_closing_radius
        self.chunksize = chunksize
        self.logger = logger
        self.error = error
        self.log_file_path = log_file_path
        self.consider_multiple_labels = consider_multiple_labels
        self.use_previous_output = use_previous_output
        self.self_optimize = self_optimize
        self.mean_slice_thickness = mean_slice_thickness
        self.resample_slice_thickness_threshold = resample_slice_thickness_threshold
        self.fast_mode = fast_mode
        self.timeout = timeout
        self.seed = seed
        self.pertubration_closing_correction = pertubration_closing_correction
        self.pertubration_convex_correction = pertubration_convex_correction

        if self.RunID is None:
            self.RunID = time.strftime("%Y%m%d-%H%M%S")

        if self.out_path.endswith("/"):
            self.out_path = self.out_path[:-1]

        # default values
        if self.perturbation_method is None:
            self.perturbation_method = ["supervoxel",
                                        "connected_component",
                                        "random_walker"]

        if self.normalization_method is None:
            self.normalization_method = ["z_score"]  # alternativly: n4bf

        if self.peri_dist is None:
            self.peri_dist = [3]

        if self.expand_seg_dist is None:
            self.expand_seg_dist = [1]

        if self.perturbation_roi_adapt_size is None:
            self.perturbation_roi_adapt_size = [-2.0, +2.0]

        # Kernels to execute transoramtions from pipeline
        if self.kernels is None:
            self.kernels = ["Wavelet",
                            "laplacian_of_gaussian",
                            "gabor",
                            "gaussian",
                            "laws",
                            "mean",
                            "Gradient", 
                            #'WaveletLLH', 
                            #'WaveletLLL', 
                            #'WaveletHLL', 
                            #'WaveletHHL', 
                            #'WaveletLHL', 
                            #'WaveletHHH', 
                            #'WaveletHLH', 
                            #'WaveletLHH', 
                            'Square', 
                            'SquareRoot', 
                            'Logarithm', 
                            'Exponential', 
                            'Gradient', 
                            'LBP2D', 
                            'LBP3D', 
                            'LoG']

        if self.out_path == "":
            if not os.path.isdir("preprocessed_data"):
                os.mkdir("preprocessed_data")
            self.out_path = os.getcwd() + "/preprocessed_data"

        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        self.experiments = []
        self.scanner_protocols = []

        # Config Logger #
        # if self.logger is None:
        if self.log_file_path is None:
            self.logger = LogGenerator(
                log_file_name=self.out_path + "/RPTK_preprocessing_" + self.RunID + ".log",
                logger_topic="RPTK Pre-processing"
            ).generate_log()

            self.error = LogGenerator(
                log_file_name=self.out_path + "/RPTK_preprocessing_" + self.RunID + ".err",
                logger_topic="RPTK Pre-processing error"
            ).generate_log()
        else:
            self.logger = LogGenerator(
                log_file_name=self.log_file_path + self.RunID + ".log",
                logger_topic="RPTK Pre-processing"
            ).generate_log()

            self.error = LogGenerator(
                log_file_name=self.log_file_path + self.RunID + ".err",
                logger_topic="RPTK Pre-processing error"
            ).generate_log()

        # if self.error is None:

        self.seg_spacings = []
        self.dice_threshold = dice_threshold
        self.out_file = self.out_path + "/" + self.RunID + "_preprocessing_out.csv"

        # Load Segmentation #
        self.isotropic_scale = isotropic_scale
        self.resampling = resampling
        self.roi_threshold = roi_threshold
        self.df = pd.DataFrame()
        self.normalized_out = ""
        self.integrated_kernels = ["Wavelet-HHH",
                                   "Wavelet-LHH",
                                   "Wavelet-HLH",
                                   "Wavelet-HHL",
                                   "Wavelet-HLL",
                                   "Wavelet-LHL",
                                   "Wavelet-LLH",
                                   "Wavelet-LLL",
                                   "Square",
                                   "LoG",
                                   "SquareRoot",
                                   "Logarithm",
                                   "Exponential",
                                   "Gradient",
                                   "LBP2D",
                                   "LBP3D",
                                   "laws",
                                   "gabor",
                                   "gaussian",
                                   # "nonseparable_wavelet",
                                   "separable_wavelet",
                                   "mean",
                                   "laplacian_of_gaussian"]

        self.kernels_in_files = ["wavelet-HHH",
                                 "wavelet-LHH",
                                 "wavelet-HLH",
                                 "wavelet-HHL",
                                 "wavelet-HLL",
                                 "wavelet-LHL",
                                 "wavelet-LLH",
                                 "wavelet-LLL",
                                 "square.",
                                 "log-",
                                 "squareroot",
                                 "_logarithm",
                                 "_exponential",
                                 "_gradient",
                                 "_lbp-2D",
                                 "_lbp-3D",
                                 "_laws_",
                                 "_gabor_",
                                 "gauss",
                                 # "_wavelet_",
                                 "separable_wavelet",
                                 "_mean_",
                                 "log_"]

        self.kernels_in_files_dict = {"wavelet-": "Wavelet",
                                      "square.": "Square",
                                      "squareroot": "SquareRoot",
                                      "_logarithm": "Logarithm",
                                      "_exponential": "Exponential",
                                      "_gradient": "Gradient",
                                      "_lbp-2D": "LBP2D",
                                      "_lbp-3D": "LBP3D",
                                      "_laws_": "laws",
                                      "gabor": "gabor",
                                      "gauss": "gaussian",
                                      "separable_wavelet": "separable_wavelet",  # Maybe not correct
                                      "mean": "mean",
                                      "log_": "laplacian_of_gaussian",
                                      "log-": "LoG"}


        self.needed_columns = ["ID", "Image", "Mask", "Prediction_Label",
                          "Modality", "ROI_Label", "Image_Transformation",
                          "Mask_Transformation", "Timepoint", "Rater"]
        self.trans_to_process = {}

    def check_format(self, df: pd.DataFrame = pd.DataFrame()):
        """
        Check if CSV file has correct format
        Min Format: Image, Mask, Modality
        Format Configuration: Image, Mask, Modality, ID, ROI_Label, Image_Transformation, Mask_Transformation,
        Timepoint, Rater, Prediction_Label
        :param df: pandas dataframe
        :return: configured pandas dataframe
        """

        return_ = True

        if len(df) == 0:
            return_ = False
            df = self.df

        # list_of_needed_columns = ["Image", "Mask", "Modality", "ID", "ROI_Label", "Image_Transformation",
        # "Mask_Transformation", "Timepoint", "Rater", "Prediction_Label"]
        transformation_num = 0
        if "Image" not in df.columns:
            self.error.warning("CSV FORMAT WARNING: Image is missing in CSV file")
            print("CSV FORMAT WARNING: Image is missing in CSV file")
            raise ValueError("CSV FORMAT ERROR: Please provide a CSV file with columns Image")
        elif "Mask" not in df.columns:
            self.error.warning("CSV FORMAT WARNING: Mask is missing in CSV file")
            print("CSV FORMAT WARNING: Mask is missing in CSV file")
            raise ValueError("CSV FORMAT ERROR: Please provide a CSV file with columns Mask")
        elif "Modality" not in df.columns:
            replace = False

            for col in df.columns:
                if "modality" in col:
                    df.rename(columns={col: "Modality"}, errors="raise", inplace=True)
                    replace = True
                    break

            if replace == False:
                self.error.warning("CSV FORMAT WARNING: Please provide a CSV file with columns Modality")
                print("CSV FORMAT WARNING: Modality is missing in CSV file")
                #raise ValueError("CSV FORMAT ERROR: Please provide a CSV file with columns Modality")

        if len(df) == 0:
            self.error.error("Please provide a CSV file with columns Image, Mask, Modality")
            raise ValueError("Please provide a CSV file with columns Image, Mask, Modality")
            print("CSV FORMAT WARNING: Modality is missing in CSV file")

        if "ID" not in df.columns:
            self.error.warning("No ID column found in CSV file. Adding column with Image File Name")
            df["ID"] = df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

        if "ROI_Label" not in df.columns:
            self.error.warning("No ROI_Label column found in CSV file. Adding value 1 for foreground.")
            df["ROI_Label"] = 1

        if "Image_Transformation" not in df.columns:
            self.error.warning("No Image_transformation column found in CSV file. Ignoring this variable ...")
            df["Image_Transformation"] = None
        else:
            # Check for different Scanner protocols
            # self.logger.info("Detected Image transformation.")
            if df["Image_Transformation"].unique().shape[0] > 1:
                number_of_transformation = df["Image_Transformation"].unique().shape[0]
                self.logger.info("Found " + str(number_of_transformation) + " different image settings.")
                print("Found " + str(number_of_transformation) + " different image settings.")

                # df["ID"] = df["ID"] + "_" + df["Image_Transformation"]
                for transformation in df["Image_Transformation"].unique().tolist():
                    if str(transformation) != "nan":
                        transformation_num += 1

                self.logger.info("Configured {} Image Transformations".format(str(transformation_num)))
                self.scanner_protocols = df["Image_Transformation"].unique().tolist()

            # Here!!! Check if there is the right transformation in the row which is included in the file name
            for i, row in df.iterrows():
                if row["Image"] == 0 or pd.isnull(row["Image"]) or row["Image"] == "" or row["Image"] == " ":
                    self.error.warning("Found empty Image path {} in CSV file {}.".format(row["Image"], row["ID"]))
                    raise ValueError("Found empty Image path {} in CSV file {}.".format(row["Image"], row["ID"]))

                if row["Image_Transformation"] == "wavelet-":
                    kernel = self.find_kernel_in_pattern(img_path=row["Image"])

                    if kernel is not None:
                        if kernel != row["Image_Transformation"]:
                            # self.logger.warning("Found different scanner protocols in entry " + str(img_pattern) + " of CSV file.")
                            # self.logger.info("Replace " + str(row["Image_Transformation"]) + " with " + str(kernel))
                            row["Image_Transformation"] = kernel
                            df.iloc[i] = row

        if "Mask_Transformation" not in df.columns:
            self.error.warning("No mask transformation found in CSV file. Ignoring this variable ...")
            df["Mask_Transformation"] = None

        if "Timepoint" not in df.columns:
            self.error.warning("No Timepoint found in CSV file. Ignoring this variable ...")
            df["Timepoint"] = None

        if "Rater" not in df.columns:
            self.error.warning("No Rater found in CSV file. Ignoring this variable ...")
            df["Rater"] = None

        if "Prediction_Label" not in df.columns:
            self.error.warning("No prediction label found in CSV file. Ignoring this variable ...")
            df["Prediction_Label"] = None

        # drop duplicated samples
        df = df.drop_duplicates(subset=['Image', 'Mask'], keep='first').reset_index(drop=True)

        if return_:
            return df
        else:
            self.df = df

    def add_new_csv_entry(self, ID, img_path, seg_path, modality, roi_label, image_transformation,
                          mask_transformation, timepoint, rater, prediction_label, df=None):
        """
        Create new entry for csv file
        :parameter
        ID: ID of patient
        img_path: path to image
        seg_path: path to segmentation
        modality: CT or MR
        roi_label: label of ROI
        image_transformation: transformation of image
        mask_transformation: transformation of mask
        timepoint: timepoint of image
        rater: rater who segmented the image
        prediction_label: label of prediction
        """

        entry = pd.DataFrame({"ID": [ID],
                              "Image": [img_path],
                              "Mask": [seg_path],
                              "Modality": [modality],
                              "ROI_Label": [roi_label],
                              "Image_Transformation": [image_transformation],
                              "Mask_Transformation": [mask_transformation],
                              "Timepoint": [timepoint],
                              "Rater": [rater],
                              "Prediction_Label": [prediction_label]
                              })

        entry = self.check_format(df=entry)

        try:
            if df is None:
                self.df = pd.concat([self.df, entry], ignore_index=True)
            if df is not None:
                # there is no self.df in the object calling from
                return pd.concat([df, entry], ignore_index=True)
        except AttributeError:
            if df is not None:
                # there is no self.df in the object calling from
                return pd.concat([df, entry], ignore_index=True)
            else:
                return entry

    def find_kernel_in_file(self, img_file, out_t_files, valid_kernels, previous_out, img, kernels_in_files):
        """
        Find the transformation kernels inside the name of the files to get an estimate about processed samples.
        :param img_file: transformed img files
        :param out_t_files: transformed img files in output folder
        :param valid_kernels: kernels found in output files
        :param previous_out:
        :param img: Img files of the Sample
        :param kernels_in_files: dict to link Kernel names with file endings
        :return:
        """

        if len(out_t_files) > 0:
            print("Found", len(out_t_files), "transformed files")

            # check if file in out transformed files if img_file in out_t_files:
            if img_file not in out_t_files:
                # file not in output files but in csv file --> need to process

                # get kernel from file of the csv
                for kernel in valid_kernels:

                    if kernel in img_file:
                        raw_img = previous_out.loc[previous_out["Image"] == img]
                        raw_img = raw_img.loc[raw_img["Image_Transformation"].isnull()].values[0]
                        real_kernel = list(kernels_in_files.keys())[list(kernels_in_files.values()).index(kernel)]
                        self.trans_to_process[real_kernel].append(raw_img)

                        break
        else:
            # if there is no transformed image for a kernel

            # get kernel from file of the csv
            for kernel in valid_kernels:
                raw_img = previous_out.loc[previous_out["Image"] == img]
                raw_img = raw_img.loc[raw_img["Image_Transformation"].isnull(), "Image"].values[0]

                real_kernel = list(kernels_in_files.keys())[list(kernels_in_files.values()).index(kernel)]
                if real_kernel in self.trans_to_process.keys():
                    self.trans_to_process[real_kernel].append(raw_img)
                else:
                    self.trans_to_process[real_kernel] = [raw_img]

    def csv_image_transformation(self, input_csv, kernels_in_files, transformation_out_path):
        """
        Scan input csv for image/mask transformations and add them to the input if they are not existing
        :param input_csv: pd.DataFrame with input parameters
        :param kernels_in_files: dict with formatted kernel names: name patterns in file
        :param transformation_out_path: path to output folder with transformed images/masks
        :return: input_csv with transformed images included from the output folder
        """

        # 1. Check for image transformation in the output
        # 1.1 Get all Images with transformation

        results = []
        if len(os.listdir(transformation_out_path)) > 0:

            # 1.2 check for image transformation in the output folder
            out_files = glob.glob(transformation_out_path + '*.nii.gz')

            # check if file is in input csv
            partial_function = partial(self.scan_processed_image_transformations,
                                       input_csv=input_csv,
                                       kernels_in_files=kernels_in_files,
                                       )
            if len(out_files) > 0:
                # Process files using multiple CPUs
                with cf.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                    # Use tqdm with map to show progress
                    for result in tqdm.tqdm(executor.map(partial_function, list(set(out_files))),
                                            total=len(list(set(out_files))), desc="Checking transformed Image Completeness"):
                        if result is not None:
                            results.append(result)

            if len(results) > 0:
                self.logger.info("Found {} processed Image Transformation files not included in the output.".format(
                    str(len(results))))
                print("Found {} processed Image Transformation files not included in the output.".format(
                    str(len(results))))

                for entry in results:
                    input_csv = pd.concat([input_csv, entry], ignore_index=True)

                input_csv = input_csv.drop_duplicates()

        return input_csv

    def find_image_transformation_config(self, file_kernel, trans_file):
        """
        Extract configuration of transformation kernel configuration which is in the file
        """

        # get configuration of transformation
        kernel_config = re.search(file_kernel + "[-]*([.]*[_]*[-]*[A-Z]*[a-z]*[0-9]*).nii.gz",
                                  os.path.basename(trans_file))
        if kernel_config:
            return kernel_config.group(1)
        else:
            return None

    def scan_processed_image_transformations(self,
                                             trans_file: str,
                                             input_csv: pd.DataFrame,
                                             kernels_in_files: dict,
                                             ):
        """
        Scan input csv for transformed images and add them to the input if they are not existing
        :param input_csv: pd.DataFrame with input parameters
        :param trans_file: output file where transformed images
        :param kernels_in_files: dict with the formatted kernel: kernel pattern in the file
        :return: input_csv with transformed images included from the output folder
        """

        df = pd.DataFrame()

        # need to add the missing transformation to input pd.DataFrame
        for ID in set(input_csv["ID"]):
            # get a unique ID if the ID could be a subset of another ID
            if str(str(ID) + "_") in os.path.basename(trans_file):

                # get the right kernel and add it to
                for kernel, file_kernel in kernels_in_files.items():
                    if file_kernel in os.path.basename(trans_file):

                        # get configuration of transformation
                        kernel_config = self.find_image_transformation_config(file_kernel=file_kernel,
                                                                              trans_file=trans_file)

                        # Add config to the kernel name
                        if kernel_config is not None:
                            if kernel_config not in kernel:
                                kernel = kernel + "_" + kernel_config

                        SampleIO = input_csv.loc[input_csv["ID"] == ID]

                        # Add all combinations of MaskTransformation
                        for mask in SampleIO.loc[SampleIO["Mask_Transformation"] != "Peritumoral", "Mask"]:
                            if mask not in SampleIO.loc[SampleIO["Image"] == trans_file, "Mask"]:
                                # Add file to input pd.DataFrame
                                tmp = self.add_new_csv_entry(
                                    ID=ID,
                                    img_path=trans_file,
                                    seg_path=mask,
                                    modality=SampleIO.loc[SampleIO["Mask"] == mask, "Modality"].to_list()[0],
                                    roi_label=SampleIO.loc[SampleIO["Mask"] == mask, "ROI_Label"].to_list()[0],
                                    image_transformation=kernel,
                                    mask_transformation=SampleIO.loc[
                                        SampleIO["Mask"] == mask, "Mask_Transformation"].to_list()[0],
                                    timepoint=SampleIO.loc[SampleIO["Mask"] == mask, "Timepoint"].to_list()[0],
                                    rater=SampleIO.loc[SampleIO["Mask"] == mask, "Rater"].to_list()[0],
                                    prediction_label=SampleIO.loc[
                                        SampleIO["Mask"] == mask, "Prediction_Label"].to_list()[0],
                                    df=pd.DataFrame())

                                df = pd.concat([df, tmp], ignore_index=True)
                                df = df.drop_duplicates()
                        break

        return df

    def add_missing_image_transformations(self, input_csv: pd.DataFrame, kernels_in_files: dict):
        """
        Search for missing or failed Image transformations and add them to a dict to process those image transformations
        :param input_csv: pd.DataFrame input csv
        :param kernels_in_files: dict with formatted kernel names: name patterns in file
        :return trans_to_process: dict with formatted kernel names: list of files
        """

        trans_to_process = {}

        for ID in tqdm.tqdm(list(set(input_csv["ID"].values)), total=len(set(input_csv["ID"])),
                            desc="Scanning for Missing Transformations"):

            # 1.3 check each ID for missing kernel transformation:
            SamplesOI = input_csv[input_csv["ID"] == ID]

            # check for non processed image transformations
            for formatted_kernel in kernels_in_files:

                # image transformation not performed for this sample
                if formatted_kernel not in SamplesOI["Image_Transformation"]:
                    if formatted_kernel in trans_to_process:
                        trans_to_process[formatted_kernel] += SamplesOI.loc[
                            SamplesOI["Image_Transformation"].isna(), "Image"].tolist()
                    else:
                        trans_to_process[formatted_kernel] = SamplesOI.loc[
                            SamplesOI["Image_Transformation"].isna(), "Image"].tolist()

            # check if image transformation files exist
            for trans_file in SamplesOI.loc[~SamplesOI["Image_Transformation"].isna(), "Image"].tolist():

                # check if the transformation file does not exist
                if not os.path.isfile(trans_file):
                    self.error.warning("File {} does not exist, need to redo transformation!".format(trans_file))
                    kernel = SamplesOI.loc[SamplesOI["Image"] == trans_file, "Image_Transformation"].tolist()

                    for k in kernel:
                        if k in trans_to_process:
                            for img in list(
                                    set(SamplesOI.loc[SamplesOI["Image_Transformation"].isna(), "Image"].to_list())):
                                if img not in trans_to_process[k]:
                                    trans_to_process[k].append(img)
                        else:
                            trans_to_process[k] = list(set(SamplesOI.loc[
                                                               SamplesOI[
                                                                   "Image_Transformation"].isna(), "Image"].to_list()))

        return trans_to_process

    def Seg2df2arr(self):
        """ 
        Load segmentation from csv file/single path/folder and integrate into dataframe
        :return: list of segmentation arrays
        """

        # lists for samples with IDs to process if the output of a previous run is used
        self.pert_to_process = []
        self.resample_to_process = []
        self.normalize_to_process = []
        self.trans_to_process = {}

        seg = []  # list of segmentation arrays

        print("Read & Check Input ...")

        if self.path2csv != "":
            # read csv file with config data
            self.df = pd.read_csv(self.path2csv)

            for col in self.needed_columns:
                if col not in self.df.columns:
                    self.df[col] = np.nan

            # check if multiple Raters are in the csv and check if there are segmentations without Rater
            if not self.df["Rater"].isnull().any():
                self.error.error("All segmentations do have a Rater. Please remove the Rater from the most trust " +
                                 "worthy Rater to mark it as ground truth.")
                raise ValueError("All segmentations do have a Rater. Please remove the Rater from the most trust " +
                                 "worthy Rater to mark it as ground truth.")

            # if you want to use the output of a previous run
            if self.use_previous_output:
                # check if the output of a previous run is available
                if os.path.exists(self.out_path + "/" + self.RunID + "_preprocessing_out.csv"):
                    if os.stat(self.out_path + "/" + self.RunID + "_preprocessing_out.csv").st_size != 0:
                        self.df = pd.read_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")
                    else:
                        self.error.warning(f'The preprocessing file from the previous run {self.out_path + "/" + self.RunID + "_preprocessing_out.csv"} is empty. Removing ...')
                        os.remove(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")

                missing_resample = 0
                missing_norm = 0
                missing_pert = 0
                valid_kernels = []

                # TODO: Check for already performed transformations in the CSV and the output folder
                # TODO: no transformation if kernel i input file --> find_kernel_in_file?

                # Check dataframe for each kernel transformation done for image
                # Checking transformation kernels
                self.transform_exe = Executor(rptk_config_json=self.rptk_config_json,
                                         input_csv=self.path2csv,
                                         output_dir=self.out_path + "/transformed_images/",
                                         logger=self.logger,
                                         error=self.error,
                                         RunID=self.RunID)

                # check for valid kernels in config
                for kernel in self.kernels:
                    if kernel in self.transform_exe.kernels_in_files:
                        valid_kernels.append(self.transform_exe.kernels_in_files[kernel])
                    else:
                        self.error.error("The kernel " + kernel + " is not supported. Please choose from: " + str(
                            self.transform_exe.kernels_in_files.keys()))
                        self.kernels.remove(kernel)
                        raise ValueError("The kernel " + kernel + " is not supported. Please choose from: " + str(
                            self.transform_exe.kernels_in_files.keys()))

                # if we can not get nay valid kernel, take all
                if len(valid_kernels) == 0:
                    self.error.error("No valid kernels found. Please choose from: " + str(
                        self.transform_exe.kernels_in_files.keys()))
                    raise ValueError("No valid kernels found. Please choose from: " + str(
                        self.transform_exe.kernels_in_files.keys()))

                # check in output folder for already transformed images
                out_t_imgs = glob.glob(self.out_path + "/transformed_images/*.nii.gz")

                out_t_files = []  # transformed files with valid kernel
                failed_out_files = []  # wrongly transformed files with not valid kernel
                
                if len(out_t_imgs) > 0:
                    # check for output files containing valid kernels
                    for t_img in tqdm.tqdm(out_t_imgs, desc='Scanning for Transformed Images'):
                        # check if kernel is in output files
                        kernel_in_file = False
                        found_kernel = 0

                        for kernel in valid_kernels:
                            if kernel in os.path.basename(t_img):
                                kernel_in_file = True
                                found_kernel += 1

                        if kernel_in_file and found_kernel == 1:
                            out_t_files.append(os.path.basename(t_img))
                        else:
                            # print("Cleaning transformed file as no valid kernel found:", os.path.basename(t_img))
                            # os.remove(t_img)
                            failed_out_files.append(t_img)
                            # raise ValueError("No valid kernel found in output file:", os.path.basename(t_img))

                print("Found {} transformed images in output folder".format(len(out_t_files)))

                if len(failed_out_files) > 0:
                    self.error.warning("Found {} Files with non valid kernel. These files were deleted.".format(str(len(failed_out_files))))
                    self.error.warning("Deleted non valid files: " + str(failed_out_files))

                # check if the output file exists
                if os.path.isfile(self.out_path + "/" + self.RunID + "_preprocessing_out.csv"):
                    file_size = os.stat(self.out_path + "/" + self.RunID + "_preprocessing_out.csv").st_size
                else:
                    file_size = 0

                if not (os.path.isfile(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")) or file_size == 0:
                    # if not, run from scratch and load the path2csv file
                    print("The output file from the previous run does not exist or is empty. " +
                          "\nSearching for processed data...")
                    self.error.warning(
                        "Can not load data form previous run. " +
                        "The output file from the previous preprocessing does not exist.")

                    # if previous run is empty then self.df is also empty So we should set it back to default
                    self.df = pd.read_csv(self.path2csv)

                    for col in self.needed_columns:
                        if col not in self.df.columns:
                            self.df[col] = np.nan

                    # Image Transformation
                    self.df = self.csv_image_transformation(input_csv=self.df,
                                                            kernels_in_files=self.transform_exe.kernels_in_files,
                                                            transformation_out_path=self.out_path + "/transformed_images/")

                    # self.df = self.scan_processed_transformations(input_csv=self.df,
                    #                                               transformation_out_path=self.out_path + "/transformed_images/",
                    #                                               kernels_in_files=transform_exe.kernels_in_files)

                    for img in self.df["Image"].to_list():

                        # drop sample if it has been wrongly transformed
                        if img in failed_out_files:
                            self.df = self.df[~self.df["Image"] == img]
                            continue

                        try:
                            sample = self.df.loc[self.df["Image"] == img, "ID"].values[0]
                        except:
                            self.error.error("Processing", img,
                                             "failed. The column 'ID' is missing in the csv file. Please add the column 'ID' to the csv file.")
                            raise ValueError(
                                "The column 'ID' is missing in the csv file. Please add the column 'ID' to the csv file.")

                        # TODO: Make recommendation for resampling -- Varinace of slice thickness?

                        # Resampling
                        if self.resampling:
                            img_file = os.path.basename(img)
                            if not "resampled" in img_file: # img.endswith("_resampled.nii.gz"):
                                missing_resample += 1
                                # self.error.warning("The output file from the previous preprocessing does not contain any resampled images for sample: " + sample)
                                if sample not in self.resample_to_process:
                                    self.resample_to_process.append(sample)

                        # Normalization
                        if self.self_optimize:
                            # only normalize for MR Images
                            if "MR" in list(self.df["Modality"].values):
                                print("Configure Settings for optimization. Normalize MR Images ...")
                                self.logger.info("Configure Settings for optimization. Normalize MR Images ...")
                                self.normalization = True
                            else:
                                print("Configure Settings for optimization. Do not normalize CT Images ...")
                                self.logger.info("Configure Settings for optimization. Do not normalize CT Images ...")
                                self.normalization = False

                        if (self.normalization) and (self.normalization_method is not None):

                            if "z_score" in self.normalization_method:
                                if not img.endswith("_z_score_normalized.nii.gz"):
                                    missing_norm += 1
                                    # self.error.warning("The output file from the previous preprocessing does not contain any normalized images for sample: " + sample)
                                    if sample not in self.normalize_to_process:
                                        self.normalize_to_process.append(sample)

                            if "n4bf" in self.normalization_method:
                                if not img.endswith("_n4bf_normalized.nii.gz"):
                                    missing_norm += 1
                                    # self.error.warning("The output file from the previous preprocessing does not contain any normalized images for sample: " + sample)
                                    if sample not in self.normalize_to_process:
                                        self.normalize_to_process.append(sample)

                        # Perturbation
                        if (self.roi_pert_repetition > 0) and (self.perturbation_method is not None):
                            # check if each sample has at least one perturbation
                            if len(self.df.loc[self.df["ID"] == sample, "Mask_Transformation"].dropna()) == 0:
                                missing_pert += 1
                                self.pert_to_process.append(sample)

                        # Transformation
                        # if len(self.kernels) > 0:
                        #
                        #     # Add all kernels to process for this sample
                        #     for kernel in valid_kernels:
                        #         real_kernel = list(transform_exe.kernels_in_files.keys())[
                        #             list(transform_exe.kernels_in_files.values()).index(kernel)]
                        #
                        #         if real_kernel in self.trans_to_process:
                        #             self.trans_to_process[real_kernel].append(img)
                        #         else:
                        #             self.trans_to_process[real_kernel] = [img]

                else:
                    print("Found output file from previous run: " + self.RunID + "_preprocessing_out.csv")
                    print("Loading data from previous run...")

                    # check for config: kernels, resampling, normalization, perturbation
                    previous_out = pd.read_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")

                    # filter for wrongly transformed images
                    for img in previous_out["Image"].copy():
                        # drop sample if it has been wrongly transformed
                        if img in failed_out_files:
                            previous_out = previous_out[previous_out["Image"] != img]

                    # get IDs from each sample
                    previous_samples = list(set(previous_out["ID"].to_list()))

                    ### Check for Transformed Images
                    missing_trans = 0
                    no_transformation = 0

                    # Image Transformation
                    # complete dataframe if it does not contain all transformations
                    previous_out = self.csv_image_transformation(input_csv=previous_out,
                                                                 kernels_in_files=self.transform_exe.kernels_in_files,
                                                                 transformation_out_path=self.out_path + "/transformed_images/")

                    # previous_out = self.scan_processed_transformations(input_csv=previous_out,
                    #                                                    transformation_out_path=self.out_path +
                    #                                                                            "/transformed_images/",
                    #                                                    kernels_in_files=transform_exe.kernels_in_files)

                    if len(previous_out) > 0:
                        self.df = pd.concat([self.df, previous_out], ignore_index=True)
                        # self.df.to_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")

                    # gather all missing transformations
                    self.trans_to_process = self.add_missing_image_transformations(input_csv=previous_out,
                                                                                   kernels_in_files=self.transform_exe.kernels_in_files)

                    # go through samples and get processed perturbations
                    for sample in tqdm.tqdm(previous_samples, desc="Scanning for data from previous run"):

                        # get all images for each sample
                        imgs = set(previous_out.loc[previous_out["ID"] == sample, "Image"].to_list())

                        # Resampling
                        if self.resampling:
                            # check if each segmentation and image has been resampled
                            for img in imgs:
                                img_file = os.path.basename(img)
                                if not "resampled" in img_file: # if not img.endswith("_resampled.nii.gz"):
                                    missing_resample += 1
                                    # self.error.warning("The output file from the previous preprocessing does not contain any resampled images for sample: " + sample)
                                    if sample not in self.resample_to_process:
                                        self.resample_to_process.append(sample)
                        # Normalization
                        if self.normalization:
                            # check if each segmentation and image has been normalized

                            if "z_score" in self.normalization_method:
                                # check if each image has been normalized with z normalization

                                for img in imgs:
                                    if not img.endswith("_z_score_normalized.nii.gz"):
                                        missing_norm += 1
                                        # self.error.warning("The output file from the previous preprocessing does not contain any normalized images for sample: " + sample)
                                        if sample not in self.normalize_to_process:
                                            self.normalize_to_process.append(sample)

                            if "n4bf" in self.normalization_method:
                                # check if each image has been normalized with n4bf normalization

                                for img in imgs:
                                    if not img.endswith("_n4bf_normalized.nii.gz"):
                                        missing_norm += 1
                                        # self.error.warning("The output file from the previous preprocessing does not contain any normalized images for sample: " + sample)
                                        if sample not in self.normalize_to_process:
                                            self.normalize_to_process.append(sample)

                        # Perturbation
                        if (self.roi_pert_repetition > 0) and (self.perturbation_method is not None):

                            # check if each sample has at least one perturbation
                            if len(previous_out.loc[previous_out["ID"] == sample, "Mask_Transformation"].dropna()) == 0:
                                missing_pert += 1
                                self.pert_to_process.append(sample)

                        # Transformation
                        # if len(self.kernels) > 0:
                        #
                        #     # transformed img files: out_t_files
                        #     # Img files of the Sample: imgs (transformed and not transformed)
                        #     # kernels found in output files: valid_kernels
                        #     # transform_exe.kernels_in_files: dict to link Kernel names with file endings
                        #     no_transformation = 0
                        #     transformed_samples = []
                        #
                        #     # Check if transformations for this file are not in the output folder
                        #     for out_file in out_t_files:
                        #         # Check if ID is the out file
                        #         if sample in out_file:
                        #             # Need to add to the csv
                        #             transformed_samples.append(out_file)
                        #
                        #     for img in imgs:
                        #         img_file = os.path.basename(img)
                        #
                        #         # We do not have any transformation for this sample --> add all transformations
                        #         if previous_out.loc[
                        #             previous_out["ID"] == sample, "Image_Transformation"].isnull().all():
                        #             no_transformation += 1
                        #             self.logger.info("No Transformation found for sample {} in outfile.".format(sample))
                        #
                        #             # Add all kernels to process for this sample
                        #             for kernel in valid_kernels:
                        #                 performed = False
                        #                 tmp = pd.DataFrame()
                        #
                        #                 # Check if this transformation has not been done already
                        #                 for performed_img_trans in transformed_samples:
                        #                     # check if transformation has been performed but not included into outfile
                        #                     if kernel in performed_img_trans:
                        #                         performed = True
                        #                         kernels_translate = transform_exe.kernels_in_files
                        #                         right_kernel = ""
                        #                         for trans_kernel in kernels_translate.keys():
                        #                             if kernel == kernels_translate[trans_kernel]:
                        #                                 right_kernel = trans_kernel
                        #
                        #                         if right_kernel == "":
                        #                             right_kernel = kernels_translate.keys()[kernels_translate.values().index(kernel)]
                        #
                        #                         print("Got Kernel: ", right_kernel)
                        #
                        #                         self.logger.info(
                        #                             "{} Transformation found for sample {} in outfolder.".format(right_kernel, sample))
                        #                         # Add to all mask transformations in combination with image transformations
                        #                         for msk in previous_out.loc[previous_out["ID"] == sample, "Mask"]:
                        #                             if "ROI_label" in previous_out.columns:
                        #                                 roi_label = previous_out.loc[previous_out["Mask"] == msk, "ROI_label"]
                        #                             else:
                        #                                 roi_label = "1"
                        #                             if "Timepoint" in previous_out.columns:
                        #                                 timepoint = previous_out.loc[previous_out["Mask"] == msk, "Timepoint"]
                        #                             else:
                        #                                 timepoint = np.nan
                        #
                        #                             entr = self.add_new_csv_entry(ID=sample,
                        #                                                          img_path=performed_img_trans,
                        #                                                          seg_path=msk,
                        #                                                          modality=previous_out.loc[previous_out["Mask"] == msk, "Modality"],
                        #                                                          roi_label=roi_label,
                        #                                                          image_transformation=right_kernel,
                        #                                                          mask_transformation=previous_out.loc[previous_out["Mask"] == msk, "Mask_Transformation"],
                        #                                                          timepoint=timepoint,
                        #                                                          rater=previous_out.loc[previous_out["Mask"] == msk, "Rater"],
                        #                                                          prediction_label=previous_out.loc[previous_out["Mask"] == msk, "Prediction_Label"],
                        #                                                          df=pd.DataFrame())
                        #                             tmp = pd.concat([tmp, entr])
                        #
                        #                 if len(tmp) > 0:
                        #                     previous_out = pd.concat([previous_out, tmp])
                        #
                        #                 if kernel in img_file:
                        #                     performed = True
                        #
                        #                 if not performed:
                        #                     real_kernel = list(transform_exe.kernels_in_files.keys())[
                        #                         list(transform_exe.kernels_in_files.values()).index(kernel)]
                        #
                        #                     if real_kernel in self.trans_to_process:
                        #                         self.trans_to_process[real_kernel].append(img)
                        #                     else:
                        #                         self.trans_to_process[real_kernel] = [img]
                        #
                        #         else:
                        #             kernel_found = False
                        #
                        #             # check if kernel is in img file:
                        #             for kernel in valid_kernels:
                        #                 kernel_found = False
                        #                 if kernel in img_file:
                        #                     kernel_found = True
                        #                     break
                        #
                        #             if not kernel_found:
                        #                 # get performed image transformation from out folder
                        #                 # 1. get all outfiles with sample name in file name
                        #
                        #                 # 2. get all transformations from file names
                        #                 for kernel in valid_kernels:
                        #                     kernel_found = False
                        #
                        #                     for out_file in transformed_samples:
                        #                         if kernel in out_file:
                        #                             kernel_found = True
                        #                             break
                        #
                        #                     if kernel in img_file:
                        #                         kernel_found = True
                        #
                        #                     # 3. check if all transformation are in out files
                        #                     if not kernel_found:
                        #                         # print("Processing transformation for sample: " + sample + " and kernel: " + kernel)
                        #                         real_kernel = list(transform_exe.kernels_in_files.keys())[
                        #                             list(transform_exe.kernels_in_files.values()).index(kernel)]
                        #
                        #                         if real_kernel in self.trans_to_process:
                        #                             self.trans_to_process[real_kernel].append(img)
                        #                         else:
                        #                             self.trans_to_process[real_kernel] = [img]

                    if missing_resample > 0:
                        self.error.warning(
                            "Resampling from the previous preprocessing is missing for " + str(
                                missing_resample) + " samples!")
                    if missing_norm > 0:
                        # check if MR is in modality
                        self.error.warning(
                            "Normalization from the previous preprocessing is missing for " + str(
                                missing_norm) + " perturbations!")
                    if missing_pert > 0:
                        self.error.warning(
                            "Perturbation from the previous preprocessing is missing for " + str(
                                missing_pert) + " perturbations!")
                    if missing_trans > 0:
                        self.error.warning(
                            "Transformation from the previous preprocessing is missing for " + str(
                                missing_trans) + " transformations!")

                    if no_transformation > 0:
                        print("No transformation found for " + str(no_transformation) + " samples!")

                    print("Need to process: ")
                    print("Resampling: " + str(len(self.resample_to_process)))
                    print("Normalization: " + str(len(self.normalize_to_process)))
                    print("Perturbation: " + str(len(self.pert_to_process)))

                    count = 0
                    for i in self.trans_to_process.values():
                        # print(i)
                        count += len(i)

                    print("Image Transformations: " + str(count) + "\n")

            self.check_format()

        # if you do want only one image or segmentation
        elif (self.segPath != "") and (self.imgPath != ""):

            print("Load segmentations ...")
            seg.append(SegmentationReader(logger=self.logger).loadSeg(self.segPath))

            self.df = pd.DataFrame({"Image": [self.imgPath],
                                    "Mask": [self.segPath],
                                    "Modality": [self.modality]})

            self.check_format()
            self.logger.info("Loaded " + str(len(seg)) + " segmentations")
            print("Loaded " + str(len(seg)) + " segmentations")
        # if you want to load all images and segmentations from a folder or only one file
        else:
            segs = []
            imgs = []

            # Check if Segpath is a file or a folder
            if os.path.isfile(self.segPath):
                segs = [self.segPath]
            elif os.path.isdir(segPath):
                segs = glob.glob(self.segDir + '/*.nii.gz')

            if os.path.isfile(self.imgPath):
                imgs = [self.imgPath]
            elif os.path.isdir(imgPath):
                imgs = glob.glob(self.imgDir + '/*.nii.gz')

            self.df = pd.DataFrame({"Image": [],
                                    "Mask": [],
                                    "Modality": []})

            print("Load segmentations ...")
            for seg_i in tqdm.tqdm(segs, desc='Reading Seg Files'):
                seg_file = os.path.basename(seg_i)
                for img_i in imgs:
                    if os.path.basename(img_i) == seg_file:
                        seg.append(SegmentationReader(logger=self.logger).loadSeg(seg_i))
                        tmp_df = pd.DataFrame({"Image": [img_i],
                                               "Mask": [seg_i],
                                               "Modality": [self.modality]})

                        self.df = pd.concat([self.df, tmp_df], ignore_index=True)

            self.check_format()
            self.logger.info("Loaded " + str(len(seg)) + " segmentations")
            print("Loaded " + str(len(seg)) + " segmentations")

        # drop not image-based parameters
        for col in self.df.columns:
            if col not in self.needed_columns:
                self.df = self.df.drop(columns=col)

    def check_image_transformation(self):
        """
        Check if image transformation configuration and kernel format is correct
        """

        image_transformation_samples = self.df.loc[~self.df["Image_Transformation"].isnull()]

        for i,r in tqdm.tqdm(image_transformation_samples.iterrows(), total=len(image_transformation_samples), desc='Correcting Image Kernel Confguration'):
            filename = os.path.basename(r["Image"])
            transformation = r["Image_Transformation"]

            for trans in self.kernels_in_files_dict :
                if trans in filename:
                    
                    try:
                        config = re.search(trans + "([-0-9A-Za-z-_.]*).nii.gz", filename).group(1)
                    except:
                        config = ""

                    # check if config is not correct
                    if transformation != self.kernels_in_files_dict[trans] + config:
                        self.df.loc[self.df["Image"] == r["Image"], "Image_Transformation"] = self.kernels_in_files_dict[trans] + config
                    break
        
        self.df = self.df.drop_duplicates(subset=['Image', 'Mask'])

    # Dice computation for binary 3D masks
    def _compute_dice(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute Dice coefficient between original and perturbed segmentations.
        
        Args:
            original (np.ndarray): Original segmentation mask.
            perturbed (np.ndarray): Perturbed segmentation mask.
        
        Returns:
            float: Dice coefficient.
        """
        intersection = np.logical_and(original, perturbed).sum()
        return 2.0 * intersection / (original.sum() + perturbed.sum())

    def compute_dice(self, row):
        """
        Compute Dice score for a row from the internal dataframe.

        Parameters:
        - row (Series): A row with keys 'Mask' and 'Mask_gt' among others.

        Returns:
        - dict: Dictionary with ID, Image, Mask, Mask_Transformation, and Dice score.
        """
        try:
            mask_pred = nib.load(row['Mask']).get_fdata()
            mask_gt = nib.load(row['Mask_gt']).get_fdata()
            dice = self._compute_dice(mask_pred, mask_gt)
            return {
                'ID': row['ID'],
                'Image': row['Image'],
                'Mask': row['Mask'],
                'Mask_Transformation': row['Mask_Transformation'],
                'Dice': dice
            }
        except Exception as e:
            print(f"Error processing ID {row['ID']}: {e}")
            return None

    def compute_external_dice(self, file_path, gt_df):
        """
        Compute Dice score for an external mask file against its ground truth.

        Parameters:
        - file_path (str): Path to the external segmentation file.
        - gt_df (DataFrame): DataFrame containing ground truth masks with ID.

        Returns:
        - dict or None: Dictionary with comparison results or None on failure.
        """
        try:
            file_name = os.path.basename(file_path)

             # get mask transformation algorithm
            if "random_walker" in file_name:
                mask_Transformation  = "random_walker"
            elif "morph" in file_name:
                mask_Transformation  = "super_voxel_randomization"
            elif "expanded" in file_name:
                mask_Transformation  = "Connected_Component_Expansion"
            else:
                mask_Transformation  = "Unknown"

            # find ground truth mask ID
            found = False
            for id_ in gt_df["ID"]:
                if id_ in file_name:
                    found = True
                    break

            if found:
                gt_row = gt_df[gt_df['ID'] == id_]
                if not gt_row.empty:
                    mask_pred = nib.load(file_path).get_fdata()
                    mask_gt = nib.load(gt_row.iloc[0]['Mask']).get_fdata()
                    dice = self._compute_dice(mask_pred, mask_gt)
                    return {
                        'ID': id_,
                        'Image': gt_row.iloc[0]['Image'],
                        'Mask': file_path,
                        'Mask_Transformation': mask_Transformation,
                        'Dice': dice
                    }
            else:
                print("could not find ID to ", file_name)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return None

    def analyze_dice_distribution(self, df, output_dir, num_workers=None, perturbed_folder=None):
        """
        Analyze and visualize Dice scores for segmentation perturbations.

        Parameters:
        - df (DataFrame): Input DataFrame with columns 'ID', 'Image', 'Mask', 'Mask_Transformation', and 'Image_Transformation'.
        - output_dir (str): Path to directory where outputs (CSV and plot) will be saved.
        - num_workers (int, optional): Number of threads to use for processing. Defaults to all available cores.
        - perturbed_folder (str, optional): Path to a folder with external perturbed segmentation files.

        Returns:
        - None: Outputs are saved to disk.
        """

        os.makedirs(output_dir, exist_ok=True)
        
        if not output_dir.endswith("/"):
            output_dir = output_dir + "/"
        
        if os.path.isfile(output_dir + "dice_scores.csv"):
            dice_df = pd.read_csv(output_dir + "dice_scores.csv")
        
        else:
            # Separate ground truth and perturbed masks
            gt_df = df[(df['Mask_Transformation'].isna()) & (df['Image_Transformation'].isna())]
            perturbed_df = df[df['Mask_Transformation'].notna() & (df['Mask_Transformation'] != 'Peritumoral')]

            dice_scores = []

            if perturbed_folder:
                # List all .nii and .nii.gz files
                perturbed_files = glob.glob(os.path.join(perturbed_folder, '*.nii')) + \
                                  glob.glob(os.path.join(perturbed_folder, '*.nii.gz'))

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(self.compute_external_dice, f, gt_df) for f in perturbed_files]
                    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Computing Dice Distribution"):
                        result = future.result()
                        if result:
                            dice_scores.append(result)
            else:
                # Merge perturbed masks with their ground truth masks by ID
                merged_df = perturbed_df.merge(gt_df[['ID', 'Mask']], on='ID', suffixes=('', '_gt'))

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(self.compute_dice, row) for _, row in merged_df.iterrows()]
                    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Computing Dice Distribution"):
                        result = future.result()
                        if result:
                            dice_scores.append(result)

            # Create DataFrame of Dice scores
            dice_df = pd.DataFrame(dice_scores)

            # Save Dice scores to CSV
            csv_path = os.path.join(output_dir, "dice_scores.csv")
            dice_df.to_csv(csv_path, index=False)

        # Plot distribution with individual points
        plt.figure(figsize=(16, 14))
        sns.set(style="whitegrid", context="talk")
        ax = sns.boxplot(data=dice_df, x='Mask_Transformation', y='Dice', color='lightgray', linewidth=2)
        sns.stripplot(data=dice_df, x='Mask_Transformation', y='Dice', 
                    jitter=True, dodge=True, alpha=0.6, color='black', size=5)

        plt.xticks(rotation=45, ha='right')
        plt.title("Dice Score Distribution per Mask Transformation Algorithm", fontsize=18)
        plt.xlabel("Mask Transformation Algorithms", fontsize=14)
        plt.ylabel("Dice Coefficient", fontsize=14)
        plt.ylim(0, 1.0)
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "dice_distribution_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()


    def process(self):
        """
        - Main function -
        Process the segmentation file.
        """

        self.logger.info("#### Start RPTK Preprocessing ####")
        print("#### Start RPTK Preprocessing ####")

        # ToDo: Check completeness of all steps by comparing to input variables and files in output folders
        # 1. check if there is output from previous run, if yes, use it and skip the following steps
        # 1.1 check resampling
        # 1.2 check normalization
        # 1.3 check segmentation filtering
        # 1.4 check segmentation perturbation

        # Read Input CSV
        self.Seg2df2arr()

        # Filter Segmentation for Size
        if not self.use_previous_output or not os.path.exists(self.out_file):
            Seg_filter_processing(self).filter_segmentations()
        else:
            self.logger.info("Using previous output. No filtering necessary.")
            print("Using previous output. No filtering necessary.")

        # self.roi_threshold

        # 1.1 Resample Images and Segmentations
        # if not self.use_previous_output or not os.path.exists(self.out_file):
        if self.resampling:
            print("Resample Images and Segmentations ...")
            Resample_processing(self).process_resampling()
        #else:
        #    self.logger.info("Using previous output. No resampling necessary.")
        #    print("Using previous output. No resampling necessary.")

        # Normalization recommended for MRI Images
        if not self.use_previous_output or not os.path.exists(self.out_file):
            if self.normalization:
                if self.modality == "CT":
                    print("Warning Normalization of CT Images is not recommended. CT Images should be normalized.")
                    self.error.warning("Warning Normalization of CT Images is not recommended. CT Images should be normalized.")
                print("### Starting Image Normalization ...")
                Normalization_processing(self).process_normalization()
                self.df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')
            else:
                if (self.modality == "MR") or (self.modality == "MRI"):
                    print("Warning Normalization of MR Images is recommended. MR Images are normally not normalized.")
                    self.error.warning("Warning Normalization of MR Images is recommended. MR Images are normally not normalized.")
        else:
            self.logger.info("Using previous output. No normalization necessary.")
            print("Using previous output. No normalization necessary.")

        # configure normalized images as default images
        if self.normalization:
            if "Image_Transformation" not in self.df.columns:
                self.df["Image_Transformation"] = np.nan
            if ("Z-score_normalized" in self.df["Image_Transformation"].values) or (
                    "N4BF_normalized" in self.df["Image_Transformation"].values):
                # drop samples were Image Transformation is nan
                self.df = self.df.loc[~self.df["Image_Transformation"].isna()]

            # remove Image transformations from normalied samples to makr them as ground truth
            if "z_score" in self.normalization_method:
                self.df.loc[self.df["Image_Transformation"] == "Z-score_normalized", "Image_Transformation"] = np.nan

            if "n4bf" in self.normalization_method:
                self.df.loc[self.df["Image_Transformation"] == "N4BF_normalized", "Image_Transformation"] = np.nan

        # 3. Apply Segmentation Filtering
        if not self.use_previous_output or not os.path.exists(self.out_file):
            print("### Starting Segmentation filtering ...")
            Seg_filter_processing(self).process_segmentation_filtering()
            self.df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')
        else:
            self.logger.info("Using previous output. No segmentation filtering necessary.")
            print("Using previous output. No segmentation filtering necessary.")

        print("Checking Segmentation filter completeness ...")
        self.df = SegProcessor.check_output_for_non_preprocessed_files(df = self.df.copy(), out_path = self.out_path, resampling=self.resampling)
        
        # Summarize results from segmentation perturbation
        self.df.to_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv", index=False)

        # 4. Apply Segmentation Perturbation
        if self.segmentation_perturbation:
            print("### Starting Segmentation perturbation ...")
            self.logger.info("### Starting Segmentation perturbation ...")
            Segmentation_perturbation_processing(self).process_segmentation_perturbation()
            self.df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')

        if self.peritumoral_seg:
            print("### Starting Surounding segmentations ...")
            self.logger.info("### Starting Surounding segmentations ...")
            Segmentation_perturbation_processing(self).process_surround_segmentation()
            self.df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')


        self.transform_exe = Executor(rptk_config_json=self.rptk_config_json,
                                 input_csv=self.path2csv,
                                 output_dir=self.out_path + "/transformed_images/",
                                 logger=self.logger,
                                 error=self.error,
                                 RunID=self.RunID)

        self.df = self.csv_image_transformation(input_csv=self.df,
                                                kernels_in_files=self.transform_exe.kernels_in_files,
                                                transformation_out_path=self.out_path + "/transformed_images/")

        self.df = self.df.drop_duplicates(subset=['Image', 'Mask'])

        # Summarize results from segmentation perturbation
        self.df.to_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv", index=False)

        # TODO: cropping Images and transfor corpped images as it might be faster

        # 5. Apply Image Transformation
        if self.image_transformation:
            self.image_transformer = Image_Transformation_processing(self) 
            self.image_transformer.process_image_transformation()

            # Image_Transformation_processing(self).process_image_transformation()
            self.df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')

            unnamed_cols = rptk.get_unnamed_cols(self.df)
            if len(unnamed_cols) > 0:
                self.logger.info("Unnamed columns found: " + str(unnamed_cols))
                self.df = self.df.drop(columns=unnamed_cols)

            # Check format of Image Transformation
            for ID in list(set(self.df["ID"].values)):
                # check each ID for missing kernel transformation:
                SamplesOI = self.df[self.df["ID"] == ID]

                # check if Image Kernel in Image Transformations contains the configuration
                for img in list(set(SamplesOI.loc[~SamplesOI["Image_Transformation"].isna(), "Image"].values)):
                    transform = Image_Transformation_processing(self).get_image_transformation_from_file(img)
                    self.df.loc[self.df["Image"] == img, "Image_Transformation"] = transform

            # drop not image-based parameters
            # for col in self.df.columns:
            #     if col not in self.needed_columns:
            #         self.df = self.df.drop(columns=col)
        # TODO: Check for transformations in ImageTransforamtions columns which are not including the kernel config

        self.df = self.df.drop_duplicates(subset=['Image', 'Mask'])

        self.check_image_transformation()

        self.df.to_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv", index=False)

        self.check_output_completeness()

        # self.df = SegProcessor.check_output_for_non_preprocessed_files(df = self.df.copy())
        print("Checking Segmentation filter processing ...")
        self.df = SegProcessor.check_output_for_non_preprocessed_files(df = self.df.copy(), out_path = self.out_path, resampling=self.resampling)

        self.df = self.df.drop_duplicates(subset=['Image', 'Mask'])

        self.df.to_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv", index=False)

        # if perturbations should be processed
        if len(self.perturbation_method) > 0:
            # get distribution of all accepted perturbations
            self.analyze_dice_distribution(self.df, self.out_path + "/plots/accepted_dice_distribution/", num_workers=self.n_cpu)

            # get distribution of all perturbations
            self.analyze_dice_distribution(self.df, self.out_path + "/plots/all_dice_distribution/", num_workers=self.n_cpu, perturbed_folder=self.out_path + "/perturbed_seg/")

        print("Preprocessing Done! Results in " + str(self.out_path + "/" + self.RunID + "_preprocessing_out.csv"))
        self.logger.info(
            "Preprocessing Done! Results in " + str(self.out_path + "/" + self.RunID + "_preprocessing_out.csv"))

        print("#### Finished RPTK Preprocessing ####")
        self.logger.info("#### Finished RPTK Preprocessing ####\n")

    def find_missing_samples_in_output(self, files: list, output: pd.DataFrame, column_to_search_for_data: str,
                                       input_target: str = ""):
        """
        Find files in columns of dataframe
        :param files: list of files to search for
        :param output: df which should contain the files
        :param column_to_add_data_to: the name of the column of the output to search for
        :param input_target: Name of input target to search for completeness (e.g. Image Transformation)
        :return list of not found files in output
        """

        missing_samples = []  # here samples are going to if they are completely absent

        for file in tqdm.tqdm(files, desc="Scanning for missing files in " + input_target):
            if file not in output[column_to_search_for_data].values:
                missing_samples.append(file)

        if len(missing_samples) > 0:
            print("Found {} missing samples in outfile!".format(str(len(missing_samples))))
            # print(missing_samples)
            return missing_samples
        else:
            return None

    def add_missing_samples_to_output(self, missing_files: list, output: pd.DataFrame, column_to_add_data_to: str,
                                      column_to_combine_with: str, input_target: str = "", transformation: str = ""):
        """
        Add missing samples to output in all available compinations.
        :param missing_files: list of file path to missing files in output
        :param output: df with data to add the missing files to
        :param column_to_add_data_to: column where the data needs to get added to
        :param column_to_combine_with: column where combinations of the new added samples should be considered
        :param input_target: Name of input target to complete (e.g. Image Transformation)
        :param transformation: if transformation needs to get applied add additional data
        :return output: output with added samples
        """

        # if output.index.name != "ID":
        #     IDs = list(set(output["ID"].values))
        #     output = output.set_index("ID")
        # else:
        #     IDs = list(set(output.index.values))
        if is_string_dtype(output["ID"]):
            string_format = True
        else:
            string_format = False

        IDs = list(set(output["ID"].values))
        print("Feature Space before Adding:", output.shape)
        print(transformation)

        for missing_sample in tqdm.tqdm(missing_files, desc="Adding missing files for completing " + input_target):
            # get ID
            file_name = os.path.basename(missing_sample)
            IDOI = ""
            file_ID = ""

            for ID in IDs:
                if file_name.startswith(str(ID)):
                    IDOI = ID
                    break

                if not string_format:
                    # if ID is not a string - it could be that the ID in the file contains 0 in front
                    for i in range(10):
                        id_0 = str(ID).zfill(len(str(ID)) + i)
                        if file_name.startswith(str(id_0)):
                            IDOI = ID
                            file_ID = str(id_0)
                            break
                    if IDOI != "":
                        if "ID" in output.columns:
                            output.loc[output["ID"] == IDOI, "ID"] = file_ID
                            IDOI = file_ID
                        elif output.index.name == "ID":
                            output["ID"] = output.index
                            output.loc[output["ID"] == IDOI, "ID"] = file_ID
                            IDOI = file_ID
                        break


            if IDOI != "":
                if transformation == "":
                    output_tmp = self.check_for_combinations_completeness(missing_sample=missing_sample,
                                                                          output=output.loc[output["ID"] == IDOI],
                                                                          column_to_add_data_to=column_to_add_data_to,
                                                                          column_to_combine_with=column_to_combine_with)
                    # if "ID" in output_tmp.columns:
                    #     output_tmp = output_tmp.set_index("ID")
                    # elif output_tmp.index.name != "ID":
                    #     self.error.error("No ID column found in data! Please check your Input CSV File.")
                    #     raise ValueError("No ID column found in data! Please check your Input CSV File.")

                    output = pd.concat([output, output_tmp], ignore_index=True)
                    output.drop_duplicates(subset=['Image', 'Mask'], keep='first', inplace=True)
                    # print(output.shape)

                elif transformation == "Mask_Transformation":
                    transform = Segmentation_perturbation_processing(self).get_mask_transformation_from_file(
                        missing_sample)
                    output_tmp = self.check_for_combinations_completeness(missing_sample=missing_sample,
                                                                          output=output.loc[output["ID"] == IDOI],
                                                                          column_to_add_data_to=column_to_add_data_to,
                                                                          column_to_combine_with=column_to_combine_with,
                                                                          transformation=(transformation, transform))
                    # if "ID" in output_tmp.columns:
                    #     output_tmp = output_tmp.set_index("ID")
                    # elif output_tmp.index.name != "ID":
                    #     self.error.error("No ID column found in data! Please check your Input CSV File.")
                    #     raise ValueError("No ID column found in data! Please check your Input CSV File.")

                    # print(str(output_tmp.columns) + " " + str(output.columns))
                    output = pd.concat([output, output_tmp], ignore_index=True)
                    output.drop_duplicates(subset=['Image', 'Mask'], keep='first', inplace=True)

                elif transformation == "Image_Transformation":
                    transform = Image_Transformation_processing(self).get_image_transformation_from_file(missing_sample)

                    if (output.index.name != "ID") and ("ID" in output.columns):
                        SampleIO = output.loc[output["ID"] == IDOI]
                    elif output.index.name == "ID":
                        SampleIO = output.loc[output.index == IDOI]
                    else:
                        self.error.error("ID is missing in columns of input CSV.")
                        raise ValueError("ID is missing in columns of input CSV.")

                    output_tmp = self.check_for_combinations_completeness(missing_sample=missing_sample,
                                                                          output=SampleIO.loc[SampleIO[
                                                                                                  "Mask_Transformation"] != "Peritumoral"],
                                                                          column_to_add_data_to=column_to_add_data_to,
                                                                          column_to_combine_with=column_to_combine_with,
                                                                          transformation=(transformation, transform))

                    # if "ID" in output_tmp.columns:
                    #     output_tmp = output_tmp.set_index("ID")
                    # elif output_tmp.index.name != "ID":
                    #     self.error.error("No ID column found in data! Please check your Input CSV File.")
                    #     raise ValueError("No ID column found in data! Please check your Input CSV File.")

                    output = pd.concat([output, output_tmp], ignore_index=True)
                    output.drop_duplicates(subset=['Image', 'Mask'], keep='first', inplace=True)
                    print(output.shape)


                else:
                    self.error.error(
                        "Transformation {} is unknown please set ot known ones (Mask_Transformation, Image_Transformation)".format(
                            trasformation))
                    raise ValueError(
                        "Transformation {} is unknown please set ot known ones (Mask_Transformation, Image_Transformation)".format(
                            trasformation))
            else:
                self.error.error(
                    "Could not find ID for target {}. Make sure it starts with an ID from the preprocessing CSV.".format(
                        missing_sample))
                raise ValueError(
                    "Could not find ID for target {}. Make sure it starts with an ID from the preprocessing CSV.".format(
                        missing_sample))

        print("Feature Space after Adding:", output.shape)
        return output

    def check_for_combinations_completeness(self, missing_sample: str, output: pd.DataFrame, column_to_add_data_to: str,
                                            column_to_combine_with: str, transformation: tuple = None):
        """
        Add missing samples to output in all available compinations with column_to_combine_with.
        :param missing_sample: samples to check if all combinations are in out and add if not
        :param output: df with data to add the missing files to
        :param column_to_add_data_to: column where the data needs to get added to
        :param column_to_combine_with: column where combinations of the new added samples should be considered
        :param transformation: tuple where Image_Transformation or Mask_Transformation and the performed transformation is included
        """

        added_combinations = output.copy()
        output_tmp = pd.DataFrame()

        # get all Samples with ID of interest
        for combine in list(set(output[column_to_combine_with].values)):

            # if missing_sample not in output.loc[output[column_to_combine_with] == combine, column_to_add_data_to].values:
            if transformation is not None:
                if transformation[0] == "Image_Transformation":
                    image_transformation = transformation[1]
                    mask_transformation = output.loc[output[column_to_combine_with] == combine, "Mask_Transformation"].values[0]
                    rater = output.loc[output[column_to_combine_with] == combine, "Rater"].values[0]
                elif transformation[0] == "Mask_Transformation":
                    image_transformation = output.loc[output[column_to_combine_with] == combine, "Image_Transformation"].values[0]
                    mask_transformation = transformation[1]
                    rater = transformation[1]
            else:
                image_transformation = output.loc[output[column_to_combine_with] == combine, "Image_Transformation"].values[0]
                mask_transformation = output.loc[output[column_to_combine_with] == combine, "Mask_Transformation"].values[0]
                rater = output.loc[output[column_to_combine_with] == combine, "Rater"].values[0]

            if "ID" in output.columns:
                ID = output.loc[output[column_to_combine_with] == combine, "ID"].values[0]
            else:
                ID = output.index[output[column_to_combine_with] == combine].values[0]

            if column_to_add_data_to == "Mask":
                output_tmp = self.add_new_csv_entry(ID=ID,
                                       img_path=
                                       output.loc[output[column_to_combine_with] == combine, "Image"].values[0],
                                       seg_path=missing_sample,
                                       modality=
                                       output.loc[output[column_to_combine_with] == combine, "Modality"].values[
                                           0],
                                       roi_label=
                                       output.loc[output[column_to_combine_with] == combine, "ROI_Label"].values[0],
                                       image_transformation=image_transformation,
                                       mask_transformation=mask_transformation,
                                       timepoint=
                                       output.loc[output[column_to_combine_with] == combine, "Timepoint"].values[0],
                                       rater=rater,
                                       prediction_label=
                                       output.loc[
                                           output[column_to_combine_with] == combine, "Prediction_Label"].values[
                                           0],
                                       df=output)


            elif column_to_add_data_to == "Image":
                output_tmp = self.add_new_csv_entry(ID=ID,
                                       img_path=missing_sample,
                                       seg_path=
                                       output.loc[output[column_to_combine_with] == combine, "Mask"].values[0],
                                       modality=
                                       output.loc[output[column_to_combine_with] == combine, "Modality"].values[
                                           0],
                                       roi_label=
                                       output.loc[output[column_to_combine_with] == combine, "ROI_Label"].values[0],
                                       image_transformation=image_transformation,
                                       mask_transformation=mask_transformation,
                                       timepoint=
                                       output.loc[output[column_to_combine_with] == combine, "Timepoint"].values[0],
                                       rater=rater,
                                       prediction_label=
                                       output.loc[
                                           output[column_to_combine_with] == combine, "Prediction_Label"].values[
                                           0],
                                       df=output)
            else:
                self.error.error("Error please define valid column to add data to (Image or Mask).")
                raise ValueError("Missing column_to_add_data_to variable. Please define valid column to add data to (Image or Mask).")

            added_combinations = pd.concat([added_combinations, output_tmp], ignore_index=True)

        added_combinations.drop_duplicates(subset=['Image', 'Mask'], keep='first', inplace=True)
        return added_combinations

    @staticmethod
    def check_output_for_non_preprocessed_files(df: pd.DataFrame, out_path: str, resampling: bool) -> pd.DataFrame:
        """
        Filters a DataFrame for non-preprocessed files and updates the "Mask" column 
        with appropriate file paths based on the presence of multilabel segmentations.

        This function iterates over the rows of the input DataFrame and checks if the 
        "Mask" column contains paths with '/input_reformated/'. If a matching path is 
        found, it updates the "Mask" column with the corresponding multilabel segmentation 
        file path from `multilabel_segs` when a match is identified by filename prefix.

        Args:
            df (pd.DataFrame): A DataFrame containing file information, with at least a "Mask" column 
                            specifying file paths.

        Returns:
            pd.DataFrame: The updated DataFrame with the "Mask" column modified for rows 
                        that match the criteria.
        """

         # check if all segmentations are 

        # 1. get all non transformed samples
        if "Mask_Transformation" in df.columns:
            non_transformed = df.copy()[df["Mask_Transformation"].isnull()]
        else:
            df["Mask_Transformation"] = np.nan
            non_transformed = df.copy()
            
        #if "Image_Transformation" in df.columns:
        #    non_transformed = non_transformed.copy()[non_transformed["Image_Transformation"].isnull()]
        #else:
        #    df["Image_Transformation"] = np.nan
        #    non_transformed["Image_Transformation"] = np.nan

        multilabel_segs = glob.glob(out_path + "/multilabel_seg/*.nii.gz")

        for msk_path in tqdm.tqdm(non_transformed["Mask"].to_list(), total=len(non_transformed["Mask"]), desc="Check for preprocessed data"):
            if "multilabel_seg/" not in msk_path:
                if resampling:
                    if "resampled" not in os.path.basename(msk_path):
                        id_ = str(non_transformed.loc[non_transformed["Mask"]==msk_path, "ID"])
                        print(f"Warning Segmentation not resampled for Sample {id_} with Segmentation {msk_path}!")
                        #self.error.error(f"Segmentation not resampled for Sample {id_} with Segmentation {msk_path}!")
                        raise ValueError(f"Segmentation not resampled for Sample {id_} with Segmentation {msk_path}!")
                    
                # Try to find filtered segmentation in folder matching segmentation name
                mask_file_pattern = os.path.basename(msk_path)[:-len(".nii.gz")]
                found = False

                for multilabel_seg_path in multilabel_segs:
                    multilabel_seg = os.path.basename(multilabel_seg_path) 
                    if multilabel_seg.startswith(mask_file_pattern):
                        df.loc[df["Mask"] == msk_path, "Mask"] = os.path.abspath(multilabel_seg_path)
                        found = True
                        break

                if not found:
                    id_ = str(non_transformed.loc[non_transformed["Mask"]==msk_path, "ID"])
                    print(f"Warning could not find filtered Segmentation for Sample {id_} with Segmentation {msk_path}!")
                    #self.error.warning(f"Warning could not find filtered Segmentation for Sample {id_} with Segmentation {msk_path}!")
                    # raise ValueError(f"Warning could not find filtered Segmentation for Sample {non_transformed.loc[non_transformed["Mask"]==msk_path, "ID"]} with Segmentation {msk_path}!")
        
        if "Mask_Transformation" in df.columns:
            # check for peritumoral seg
            peritumoral_segs = glob.glob(out_path + "/peritumor_seg/*.nii.gz")
            if len(peritumoral_segs) > 0:
                non_transformed = df.copy()[df["Mask_Transformation"] == "Peritumoral"]
                if len(non_transformed) > 0:
                    
                    for peri_msk_path in tqdm.tqdm(non_transformed["Mask"].to_list(), total=len(non_transformed["Mask"]), desc="Check for preprocessed peritumoral msks"):
                        if "peritumor_seg/"  in peri_msk_path:
                            if resampling:
                                if "resampled" not in os.path.basename(msk_path):
                                    id_ = str(non_transformed.loc[non_transformed["Mask"]==msk_path, "ID"])
                                    print(f"Warning Segmentation not resampled for Sample {id_} with Segmentation {msk_path}!")
                                    #self.error.error(f"Segmentation not resampled for Sample {id_} with Segmentation {msk_path}!")
                                    raise ValueError(f"Segmentation not resampled for Sample {id_} with Segmentation {msk_path}!")
                                
                            # Try to find filtered segmentation in folder matching segmentation name
                            mask_file_pattern = os.path.basename(peri_msk_path)[:-len("_3_peritumoral.nii.gz")]
                            found = False

                            for multilabel_seg_path in peritumoral_segs:
                                multilabel_seg = os.path.basename(multilabel_seg_path) 
                                if multilabel_seg.startswith(mask_file_pattern):
                                    df.loc[df["Mask"] == peri_msk_path, "Mask"] = os.path.abspath(multilabel_seg_path)
                                    found = True
                                    break

                            if not found:
                                id_ = str(non_transformed.loc[non_transformed["Mask"]==peri_msk_path, "ID"])
                                print(f"Warning could not find filtered Segmentation for Sample {id_} with Segmentation {peri_msk_path}!")
                                #self.error.warning(f"Warning could not find filtered Segmentation for Sample {id_} with Segmentation {peri_msk_path}!")

                else:
                    # add peritumoral samples to df if there are no peritumoral segs in the data
                    non_transformed = df.copy()[df["Mask_Transformation"].isnull()]
                    for msk_path in tqdm.tqdm(non_transformed["Mask"].to_list(), total=len(non_transformed["Mask"]), desc="Add preprocessed peritumoral msks"):
                        mask_file_pattern = os.path.basename(msk_path)[:-len("_3_peritumoral.nii.gz")]
                        found = False

                        for peritumoral_segs_path in peritumoral_segs:
                            peritumoral_seg = os.path.basename(peritumoral_segs_path) 
                            if peritumoral_seg.startswith(mask_file_pattern):
                                entry = df.copy().loc[df["Mask"] == msk_path]
                                entry["Mask"] = os.path.abspath(peritumoral_segs_path)
                                entry["Mask_Transformation"] = "Peritumoral"
                                df = pd.concat([df,entry], ignore_index=True)
                                found = True
                                break

                        if not found:
                            id_ = str(non_transformed.loc[non_transformed["Mask"]==msk_path, "ID"])
                            print(f"Warning could not find filtered Segmentation for Sample {id_} with Segmentation {msk_path}!")
                            # self.error.warning(f"Warning could not find filtered Segmentation for Sample {id_} with Segmentation {msk_path}!")


            else:
                print("No Peritumoral Segmentations found! Check output folder " + out_path + "/peritumor_seg/")
        #else:
            # add peritumoral samples to df

        # Filter for non-preprocessed files in the file
        #for i, row in tqdm.tqdm(df.copy().iterrows(), total=len(df), desc="Check for preprocessed data"):
        #    if '/input_reformated/' in row["Mask"]:
        #        mask_file = os.path.basename(str(row.copy()["Mask"]))[:-len(".nii.gz")]
        #        for multi_label_seg in multilabel_segs:
        #            multi_label_seg_file = os.path.basename(multi_label_seg)
        #            if multi_label_seg_file.startswith(mask_file):
        #                df.loc[i, "Mask"] = multi_label_seg
        #                break

        return df


    def check_output_completeness(self):
        """
        Check if all combinations of processed and accepted transformations are within the preprocessed csv file
        :return:
        """
        # 1. Get all processed data
        # get all filtered segmentations
        if not self.resampling and not self.normalization:
            multilabel_seg = glob.glob(self.out_path + "/multilabel_seg/*.nii.gz")
            missing_samples = self.find_missing_samples_in_output(files=multilabel_seg,
                                                                  output=self.df,
                                                                  column_to_search_for_data="Mask",
                                                                  input_target="Filtered Segmentations")
            if missing_samples is not None:
                self.df = self.add_missing_samples_to_output(missing_files=missing_samples,
                                                             output=self.df,
                                                             column_to_add_data_to="Mask",
                                                             column_to_combine_with="Image",
                                                             input_target="Filtered Segmentations")

        # get all resamples images/masks
        if self.resampling and (not self.normalization):
            # HERE
            resampled_img = glob.glob(self.out_path + "/resampled/img/*.nii.gz")
            missing_samples = self.find_missing_samples_in_output(files=resampled_img,
                                                                  output=self.df,
                                                                  column_to_search_for_data="Image",
                                                                  input_target="Resampled Images")
            if missing_samples is not None:
                self.df = self.add_missing_samples_to_output(missing_files=missing_samples,
                                                             output=self.df,
                                                             column_to_add_data_to="Image",
                                                             column_to_combine_with="Mask",
                                                             input_target="Resampled Images")

            # Filtered Segmentations should be resampled!
            #resampled_msk = glob.glob(self.out_path + "/resampled/seg/*.nii.gz")
            #missing_samples = self.find_missing_samples_in_output(files=resampled_msk,
            #                                                      output=self.df,
            #                                                      column_to_search_for_data="Mask",
            #                                                      input_target="Resampled Masks")
            #if missing_samples is not None:
            #    self.df = self.add_missing_samples_to_output(missing_files=missing_samples,
            #                                                 output=self.df,
            #                                                 column_to_add_data_to="Mask",
            #                                                 column_to_combine_with="Image",
            #                                                 input_target="Resampled Masks")

        # get all segmentation perturbations
        if self.segmentation_perturbation:
            accepted_perturbed_seg = glob.glob(self.out_path + "/accepted_perturbed_seg/*.nii.gz")
            missing_samples = self.find_missing_samples_in_output(files=accepted_perturbed_seg,
                                                                  output=self.df,
                                                                  column_to_search_for_data="Mask",
                                                                  input_target="Perturbed Masks")
            if missing_samples is not None:
                out = self.add_missing_samples_to_output(missing_files=missing_samples,
                                                         output=self.df,
                                                         column_to_add_data_to="Mask",
                                                         column_to_combine_with="Image",
                                                         input_target="Perturbed Masks",
                                                         transformation="Mask_Transformation")
                if len(out) > 0:
                    self.df = pd.concat([self.df, out])

            self.df.drop_duplicates(subset=["Image", "Mask"], inplace=True)

            # if resampled was done - check for correct samples
            self.logger.info("Check for correct resampling ...")
            print("Check for correct resampling ...")
            before = self.df.shape[0]
            # NEW Change 16.01.15
            # self.df = self.df[self.df["Mask"].apply(lambda x: os.path.basename(x).lower().count("resampled") ==1)]
            # self.df = self.df[self.df["Image"].apply(lambda x: os.path.basename(x).lower().count("resampled") ==1)]
            self.df = self.df[~self.df.Mask.str.contains("resampled_resampled")]
            self.df = self.df[~self.df.Image.str.contains("resampled_resampled")]

            after = self.df.shape[0]
            if before != after:
                print(f"Found wrong formatted resampling. Dropped {str(int(before) - int(after))} Samples.")
                self.logger.info(f"Found wrong formatted resampling. Dropped {str(int(before) - int(after))} Samples.")
            
            if len(accepted_perturbed_seg) > 0:
                # check for all processed Mask Transformations if there are all combinations with the Images of the samples
                for msk in tqdm.tqdm(accepted_perturbed_seg, desc="Checking for completeness of Mask Perturbations"):
                    mask_transformation = Segmentation_perturbation_processing(self).get_mask_transformation_from_file(msk)

                    # get ID
                    IDs = list(set(self.df["ID"].values))
                    file_name = os.path.basename(msk)
                    IDOI = ""

                    for ID in IDs:
                        if file_name.startswith(str(ID)):
                            IDOI = ID
                            break

                    SampleIO = self.df.loc[self.df["ID"] == IDOI]
                    out = pd.DataFrame()

                    # check if all combinations between target image and masks are included
                    for img in list(set(SampleIO["Image"].values)):
                        out = self.add_new_csv_entry(ID=SampleIO.loc[SampleIO["Image"] == img, "ID"].values[0],
                                            img_path=img,
                                            seg_path=msk,
                                            modality=SampleIO.loc[SampleIO["Image"] == img, "Modality"].values[0],
                                            roi_label=SampleIO.loc[SampleIO["Image"] == img, "ROI_Label"].values[0],
                                            image_transformation=
                                            SampleIO.loc[SampleIO["Image"] == img, "Image_Transformation"].values[0],
                                            mask_transformation=mask_transformation,
                                            timepoint=SampleIO.loc[SampleIO["Image"] == img, "Timepoint"].values[0],
                                            rater=mask_transformation,
                                            prediction_label=
                                            SampleIO.loc[SampleIO["Image"] == img, "Prediction_Label"].values[0],
                                            df=out)

                    self.df = pd.concat([self.df, out], ignore_index=True)

        # get all image transformations
        transformed_images = glob.glob(self.out_path + "/transformed_images/*.nii.gz")
        missing_samples = self.find_missing_samples_in_output(files=transformed_images,
                                                              output=self.df,
                                                              column_to_search_for_data="Image",
                                                              input_target="Transformed Images")
        if missing_samples is not None:
            out = self.add_missing_samples_to_output(missing_files=missing_samples,
                                                     output=self.df,
                                                     column_to_add_data_to="Image",
                                                     column_to_combine_with="Mask",
                                                     input_target="Transformed Images",
                                                     transformation="Image_Transformation")
            if len(out) > 0:
                self.df = pd.concat([self.df, out])

        self.df.drop_duplicates(subset=["Image", "Mask"], inplace=True)

        # check for all processed Image Transformations if there are all combinations with the masks of the samples
        for img in tqdm.tqdm(transformed_images, desc="Checking for completeness of Image Transformations"):

            if os.path.getsize(img) == 0:
                self.error.warning("Found empty file {}! Deleting it to repeat!".format(img))
                print("Found empty file {}! Deleting it to repeat!".format(img))
                os.remove(img)

            image_transformation = Image_Transformation_processing(self).get_image_transformation_from_file(img)

            # get ID
            IDs = list(set(self.df["ID"].values))
            file_name = os.path.basename(img)
            IDOI = ""

            for ID in IDs:
                if file_name.startswith(str(ID)):
                    IDOI = ID
                    break

                # if ID is a number it could be that 0s are in front of the ID which lead to mismatching
                if str(ID).isnumeric():
                    # adding up to 8 0s to the ID and check if file fits
                    for null in range(1, 9):
                        nulled_ID = str(ID).zfill(null)
                        if file_name.startswith(str(nulled_ID)):
                            IDOI = ID
                            break

            if IDOI == "":
                self.error.error("Could not find the ID for file " + file_name)
                raise ValueError("Could not find the ID for file " + file_name)

            SampleIO = self.df.loc[self.df["ID"] == IDOI]
            SampleIO = SampleIO.loc[SampleIO["Mask_Transformation"] != "Peritumoral"]
            out = pd.DataFrame()

            # check if all combinations between target image and masks are included
            for msk in list(set(SampleIO["Mask"].values)):
                out = self.add_new_csv_entry(
                    ID=SampleIO.loc[SampleIO["Mask"] == msk, "ID"].values[0],
                    img_path=img,
                    seg_path=SampleIO.loc[SampleIO["Mask"] == msk, "Mask"].values[0],
                    modality=SampleIO.loc[SampleIO["Mask"] == msk, "Modality"].values[0],
                    roi_label=SampleIO.loc[SampleIO["Mask"] == msk, "ROI_Label"].values[0],
                    image_transformation=image_transformation,
                    mask_transformation=SampleIO.loc[SampleIO["Mask"] == msk, "Mask_Transformation"].values[0],
                    timepoint=SampleIO.loc[SampleIO["Mask"] == msk, "Timepoint"].values[0],
                    rater=SampleIO.loc[SampleIO["Mask"] == msk, "Rater"].values[0],
                    prediction_label=SampleIO.loc[SampleIO["Mask"] == msk, "Prediction_Label"].values[0],
                    df=out)

            self.df = pd.concat([self.df, out], ignore_index=True)

        self.df.drop_duplicates(subset=["Image", "Mask"], inplace=True)

        # delete all samples with image transformation and peritumoral seg transformation
        to_drop = self.df.loc[
            (self.df["Mask_Transformation"] == "Peritumoral") & (self.df["Image_Transformation"].notna())]
        self.df = pd.concat([self.df, to_drop, to_drop]).drop_duplicates(keep=False)

        # check if also samples without image transformation fo have mask transformation
        raw_imgs = list(set(self.df.loc[(self.df["Mask_Transformation"].isna()) & (
            self.df["Image_Transformation"].isna()), "Image"].values))

        for img in tqdm.tqdm(raw_imgs, desc="Checking for completeness of Images"):

            # get ID
            IDs = list(set(self.df["ID"].values))
            file_name = os.path.basename(img)
            IDOI = ""

            for ID in IDs:
                if file_name.startswith(str(ID)):
                    IDOI = ID
                    break

            SampleIO = self.df.loc[self.df["ID"] == IDOI]
            out = pd.DataFrame()
            # check if all combinations between target image and masks are included
            for msk in list(set(SampleIO["Mask"].values)):
                out = self.add_new_csv_entry(ID=SampleIO.loc[SampleIO["Mask"] == msk, "ID"].values[0],
                                       img_path=img,
                                       seg_path=SampleIO.loc[SampleIO["Mask"] == msk, "ID"].values[0],
                                       modality=SampleIO.loc[SampleIO["Mask"] == msk, "Modality"].values[0],
                                       roi_label=SampleIO.loc[SampleIO["Mask"] == msk, "ROI_Label"].values[0],
                                       image_transformation=np.nan,
                                       mask_transformation=
                                       SampleIO.loc[SampleIO["Mask"] == msk, "Mask_Transformation"].values[0],
                                       timepoint=SampleIO.loc[SampleIO["Mask"] == msk, "Timepoint"].values[0],
                                       rater=SampleIO.loc[SampleIO["Mask"] == msk, "Rater"].values[0],
                                       prediction_label=
                                       SampleIO.loc[SampleIO["Mask"] == msk, "Prediction_Label"].values[0],
                                       df=out)

            self.df = pd.concat([self.df, out], ignore_index=True)

        self.df.drop_duplicates(subset=["Image", "Mask"], inplace=True)

        # msks = list(set(self.df.loc[self.df["Mask_Transformation"].notna(), "Mask"].values))
        #
        # for img in tqdm.tqdm(msks, desc="Checking for completeness of Masks"):
        #
        #     # get ID
        #     IDs = list(set(self.df["ID"].values))
        #     file_name = os.path.basename(msk)
        #     IDOI = ""
        #
        #     for ID in IDs:
        #         if file_name.startswith(ID):
        #             IDOI = ID
        #             break
        #
        #     SampleIO = self.df.loc[self.df["ID"] == IDOI]
        #
        #     # check if all combinations between target image and masks are included
        #     for img in list(set(SampleIO["Image"].values)):
        #         if not msk in list(set(SampleIO.loc[SampleIO["Image"] == img, "Mask"].values)):
        #             self.add_new_csv_entry(ID=SampleIO.loc[SampleIO["Image"] == img, "ID"].values[0],
        #                                    img_path=img,
        #                                    seg_path=SampleIO.loc[SampleIO["Image"] == img, "ID"].values[0],
        #                                    modality=SampleIO.loc[SampleIO["Image"] == img, "Modality"].values[0],
        #                                    roi_label=SampleIO.loc[SampleIO["Image"] == img, "ROI_Label"].values[0],
        #                                    image_transformation=np.nan,
        #                                    mask_transformation=
        #                                    SampleIO.loc[SampleIO["Image"] == img, "Mask_Transformation"].values[0],
        #                                    timepoint=SampleIO.loc[SampleIO["Image"] == img, "Timepoint"].values[0],
        #                                    rater=SampleIO.loc[SampleIO["Image"] == img, "Rater"].values[0],
        #                                    prediction_label=
        #                                    SampleIO.loc[SampleIO["Image"] == img, "Prediction_Label"].values[0],
        #                                    df=self.df)

        # out = self.check_for_combinations_completeness(missing_sample=img,
        #                                                output=SampleIO,
        #                                                column_to_add_data_to="Image",
        #                                                column_to_combine_with="Mask",
        #                                                transformation=("Image_Transformation", transform))

        # self.df = pd.concat([self.df, out])

        # get all peritumoral segmentations
        if self.peritumoral_seg:
            peritumoral_segs = glob.glob(self.out_path + "/peritumor_seg/*.nii.gz")

            # check for all processed Image Transformations if there are all combinations with the masks of the samples
            for msk in tqdm.tqdm(peritumoral_segs, desc="Checking for completeness of Peritumoral Mask"):
                mask_transformation = Segmentation_perturbation_processing(self).get_mask_transformation_from_file(msk)

                # get ID
                IDs = list(set(self.df["ID"].values))
                file_name = os.path.basename(msk)
                IDOI = ""

                for ID in IDs:
                    if file_name.startswith(str(ID)):
                        IDOI = ID
                        break

                SampleIO = self.df.loc[self.df["ID"] == IDOI]
                out = pd.DataFrame()

                # check if all combinations between target image and masks are included
                for img in list(set(SampleIO["Image"].values)):
                    out = self.add_new_csv_entry(ID=SampleIO.loc[SampleIO["Image"] == img, "ID"].values[0],
                                           img_path=img,
                                           seg_path=msk,
                                           modality=SampleIO.loc[SampleIO["Image"] == img, "Modality"].values[0],
                                           roi_label=SampleIO.loc[SampleIO["Image"] == img, "ROI_Label"].values[0],
                                           image_transformation=
                                           SampleIO.loc[SampleIO["Image"] == img, "Image_Transformation"].values[0],
                                           mask_transformation=mask_transformation,
                                           timepoint=SampleIO.loc[SampleIO["Image"] == img, "Timepoint"].values[0],
                                           rater=mask_transformation,
                                           prediction_label=
                                           SampleIO.loc[SampleIO["Image"] == img, "Prediction_Label"].values[0],
                                           df=out)

                self.df = pd.concat([self.df, out], ignore_index=True)

        to_drop = self.df.loc[
            (self.df["Mask_Transformation"] == "Peritumoral") & (self.df["Image_Transformation"].notna())]
        self.df = pd.concat([self.df, to_drop, to_drop]).drop_duplicates(keep=False)

        self.df.drop_duplicates(subset=["Image", "Mask"], inplace=True)

        if self.df[self.df['Mask'].isnull()].shape[0] > 0:
            self.error.warning("Got NaN in Mask for {} Samples".format(str(self.df[self.df['Mask'].isnull()].shape[0])))
            print("Got NaN in Mask for {} Samples".format(str(self.df[self.df['Mask'].isnull()].shape[0])))
            self.df = self.df[df['Mask'].notna()]

        if self.df[self.df['Image'].isnull()].shape[0] > 0:
            self.error.warning("Got NaN in Image for {} Samples".format(str(self.df[self.df['Image'].isnull()].shape[0])))
            print("Got NaN in Image for {} Samples".format(str(self.df[self.df['Image'].isnull()].shape[0])))
            self.df = self.df[df['Image'].notna()]

        # only valid files
        self.df = self.df[self.df["Image"].str.contains('.nii.gz', na=False)]
        self.df = self.df[self.df["Mask"].str.contains('.nii.gz', na=False)]

        # check if non transformed data includes duplicates
        raw = self.df.copy()[(self.df.copy()["Image_Transformation"].isnull()) & (self.df.copy()["Mask_Transformation"].isnull())]
        if len(raw[raw.index.duplicated()]) > 0:
            df = self.df.copy()
            df = df.reset_index()
            raw = df.copy()[(df.copy()["Image_Transformation"].isnull()) & (df.copy()["Mask_Transformation"].isnull())]
            # drop non preprocessed samples
            drop = raw.copy()[~raw.copy()["Mask"].str.contains('multilabel_seg', na=False)]
            self.logger.indo("Dropping {} samples without preprocessed masks.".format(str(len(drop))))
            df = df.copy().drop(drop.index)
            df = df.set_index("ID")
            self.df = df.copy()
            del df
            del raw
            del drop

        self.df = self.df.drop_duplicates(subset=['Image', 'Mask'])

        self.df.to_csv(self.out_path + "/" + self.RunID + "_preprocessing_out.csv", index=False)

        raw = self.df.copy()[(self.df.copy()["Image_Transformation"].isnull()) & (self.df.copy()["Mask_Transformation"].isnull())]
        if len(raw[raw.index.duplicated()]) > 0:
            self.error.error("Found duplicates in non transformed samples! Please check your file for duplications {}!".format(str(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")))
            raise ValueError("Found duplicates in non transformed samples! Please check your file for duplications {}!".format(str(self.out_path + "/" + self.RunID + "_preprocessing_out.csv")))

        # get all normalized images
        # if self.normalization:
        #     normalized = glob.glob(self.out_path + "/normalized/*.nii.gz")
        #     missing_samples = self.find_missing_samples_in_output(files=normalized,
        #                                                           output=self.df,
        #                                                           column_to_search_for_data="Images",
        #                                                           input_target="Normalized Images")
        #     if missing_samples is not None:
        #         self.df = self.add_missing_samples_to_output(missing_files=missing_samples,
        #                                                      output=self.df,
        #                                                      column_to_add_data_to="Images",
        #                                                      column_to_combine_with="Mask",
        #                                                      input_target="Normalized Images")

    @staticmethod
    def get_processed_files(input_files: list, out_folder: str, file_ending: str):
        """
        Returns a list of files which are not processed by removing processed files from input files
        :param input_files: list of input files which should be processed
        :param out_folder: folder where processed files are stored
        :param file_ending: files ending of processed files (attachment to the input file (including.nii.gz))
        :return: to_process: list of files which are not processed
        """
        to_process = input_files.copy()
        processed = []

        print("Searching for already processed files ...")

        processed_paths = glob.glob(out_folder + "/*" + file_ending)

        if len(processed_paths) != 0:
            for p_path in processed_paths:
                processed.append(os.path.basename(p_path)[:-len(file_ending)])

            for ipath in input_files:
                id_ = os.path.basename(ipath)[:-len(".nii.gz")]
                for process in processed:
                    if id_ in process:
                        to_process.remove(ipath)
                        break

            if len(processed) > 0:
                print("Found " + str(len(processed)) + " already processed file(s).")
        else:
            to_process = input_files

        return to_process

    def find_kernel_in_pattern(self, img_path: str, kernels_in_files=None):
        """
        Find kernel in image pattern and return kernel name
        :param img_path: Path to image
        :param kernels_in_files: List of kernels in file names
        :return: kernel name
        """
        img_pattern = os.path.basename(img_path)[:-(len("nii.gz"))]
        kernel = None

        if kernels_in_files is None:
            try:
                kernels_in_files = self.kernels_in_files
            except:
                print("Error: Please provide kernels_in_files")

        # search for kernel in pattern
        for f_kernel in kernels_in_files:
            if f_kernel in img_pattern:
                # transform kernel names to meaningful and understandable names
                if f_kernel == "wavelet-HHH":
                    kernel = "Wavelet-HHH"
                    break
                if f_kernel == "wavelet-HHL":
                    kernel = "Wavelet-HHL"
                    break
                if f_kernel == "wavelet-HLH":
                    kernel = "Wavelet-HLH"
                    break
                if f_kernel == "wavelet-HLL":
                    kernel = "Wavelet-HLL"
                    break
                if f_kernel == "wavelet-LHH":
                    kernel = "Wavelet-LHH"
                    break
                if f_kernel == "wavelet-LHL":
                    kernel = "Wavelet-LHL"
                    break
                if f_kernel == "wavelet-LLH":
                    kernel = "Wavelet-LLH"
                    break
                if f_kernel == "wavelet-LLL":
                    kernel = "Wavelet-LLL"
                    break
                if f_kernel == "_lbp-2D":
                    kernel = "LBP2D"
                    break
                if f_kernel == "_lbp-3D":
                    kernel = "LBP3D"
                    break
                if f_kernel == "log-":
                    kernel = "LoG"
                    break
                if f_kernel == "_logarithm":
                    kernel = "Logarithm"
                    break
                if f_kernel == "_exponential":
                    kernel = "Exponential"
                    break
                if f_kernel == "_gradient":
                    kernel = "Gradient"
                    break
                if f_kernel == "_laws_":
                    kernel = "laws"
                    break
                if f_kernel == "_wavelet_":
                    kernel = "nonseparable_wavelet"
                    break
                if f_kernel == "gauss":
                    kernel = "gaussian"
                    break
                if f_kernel == "square.":
                    kernel = "Square"
                    break
                if f_kernel == "squareroot":
                    kernel = "SquareRoot"
                    break
                if f_kernel == "_gabor_":
                    kernel = "gabor"
                    break
                if f_kernel == "_mean_":
                    kernel = "mean"
                    break
        return kernel

class Resample_processing(SegProcessor):
    """
    Class for resampling images and segmentations to a given resolution.
    """

    def __init__(self, segprocessor):
        self.segprocessor = segprocessor

    def exe_resampling(self):
        """
        Resampling Images with Resampler and saves them and replace values for imgs and segs in input df.
        """

        # Check previous image output
        self.segprocessor.resampling_img_out = self.segprocessor.out_path + "/resampled/img"
        Path(self.segprocessor.resampling_img_out).mkdir(parents=True, exist_ok=True)

        # Check previous segmentation output
        self.segprocessor.resampling_seg_out = self.segprocessor.out_path + "/resampled/seg"
        Path(self.segprocessor.resampling_seg_out).mkdir(parents=True, exist_ok=True)

        if len(self.segprocessor.df) == 0:
            self.segprocessor.error.error("Please provide a path to CSV input file as path2csv " +
                                          "which contains columns for Image:Path to Imag, Mask: Path to mask, and Modality: CT or MRI")

        # check if resampled samples are included in df
        input_imgs = self.segprocessor.df["Image"].values.tolist()
        input_msk = self.segprocessor.df["Mask"].values.tolist()

        resampled_files_seg = glob.glob(self.segprocessor.resampling_seg_out + "/*.nii.gz")
        resampled_files_img = glob.glob(self.segprocessor.resampling_img_out + "/*.nii.gz")

        print("Found {} resampled Images!".format(str(len(resampled_files_img))))
        print("Found {} resampled Masks!".format(str(len(resampled_files_seg))))

        if self.segprocessor.use_previous_output:

            print("Use previous resampled images and segmentations ...")

            if len(resampled_files_seg) == 0 or len(resampled_files_img) == 0:
                print("No previous resampled images and segmentations found. Resampling ...")
            else:
                print("Found previous resampled images and segmentations. Use them ...")
                print("Loading data from",self.segprocessor.resampling_seg_out,"...")

                # make common file patterns
                resampled_files_seg_files = [os.path.basename(file_path)[:-len("_resampled.nii.gz")] for file_path in
                                             resampled_files_seg]
                resampled_files_img_files = [os.path.basename(file_path)[:-len("_resampled.nii.gz")] for file_path in
                                             resampled_files_img]

                input_msk_files = [os.path.basename(file_path)[:-len(".nii.gz")] for file_path in input_msk]
                input_imgs_files = [os.path.basename(file_path)[:-len(".nii.gz")] for file_path in input_imgs]

                # count resmpaled files
                resampled_imgs = []
                resampled_segs = []

                # check file existence and remove if file pattern exist in out folder
                for img_path in tqdm.tqdm(resampled_files_img_files, desc="Check for previous resampled images"):
                    if img_path in input_imgs_files:
                        # input_imgs_files.remove(img_path)
                        resampled_imgs.append(img_path)

                for seg_path in tqdm.tqdm(resampled_files_seg_files, desc="Check for previous resampled segmentations"):
                    if seg_path in input_msk_files:
                        # input_msk_files.remove(seg_path)
                        resampled_segs.append(seg_path)

                if (len(resampled_segs) > 1) or (len(resampled_imgs) > 1):
                    self.segprocessor.logger.info(
                        "Found {} resampled Images and {} resampled Masks".format(str(len(resampled_imgs)),
                                                                                  str(len(resampled_segs))))
                # get the entire path of the files for processing
                for img_file in resampled_imgs:
                    for img_path in input_imgs:
                        if img_file in img_path:
                            input_imgs.remove(img_path)
                            break

                for msk_file in resampled_segs:
                    for msk_path in input_msk:
                        if msk_file in msk_path:
                            input_msk.remove(msk_path)
                            break

                print("Need to resample " + str(len(input_imgs)) + " images and " + str(
                    len(input_msk)) + " segmentations.")

        if (len(input_msk) > 0) or (len(input_imgs) > 0):
            if self.segprocessor.self_optimize:
                if int(self.segprocessor.mean_slice_thickness) > int(self.segprocessor.resample_slice_thickness_threshold):
                    self.segprocessor.logger.info("Resample images as slice thickness of this dataset exceeds the threshold of " + str(int(self.segprocessor.resample_slice_thickness_threshold)) + ": "+ str(self.segprocessor.mean_slice_thickness))
                    sampling = [1.0]
                else:
                    self.segprocessor.logger.info("Resampling not necessary as slice thickness of this dataset is: " + str(self.segprocessor.mean_slice_thickness))
                    sampling = [self.segprocessor.mean_slice_thickness]
            else:
                # default
                sampling = [1.0]
            
            if len(sampling) >0:
                if len(input_msk) > 0:
                    self.segprocessor.logger.info("Resampling Segmentations ...")
                    # 1. resample msk
                    Resampler(input_files=input_msk,
                              output_folder=self.segprocessor.resampling_seg_out,
                              segmentation=True,
                              desired_sampling=sampling,
                              n_cpus=self.segprocessor.n_cpu,
                              logger=self.segprocessor.logger,
                              target="Masks").exe()

                    resampled_files_seg = glob.glob(self.segprocessor.resampling_seg_out + "/*.nii.gz")

                if len(input_imgs) > 0:
                    self.segprocessor.logger.info("Resampling Images ...")
                    # 2. resample image
                    Resampler(input_files=input_imgs,
                              output_folder=self.segprocessor.resampling_img_out,
                              segmentation=False,
                              desired_sampling=sampling,
                              n_cpus=self.segprocessor.n_cpu,
                              logger=self.segprocessor.logger,
                              target="Images").exe()

                    resampled_files_img = glob.glob(self.segprocessor.resampling_img_out + "/*.nii.gz")

        if self.segprocessor.df["Image_Transformation"].isnull().all() and self.segprocessor.df[
            "Mask_Transformation"].isnull().all():
            idx_to_drop = []
            # sort resampled images to the resampled mask as there might be maks that do share the same image
            for i, row in tqdm.tqdm(self.segprocessor.df.iterrows(), total=len(self.segprocessor.df),
                                    desc='Saving Resampled Files'):

                img_pattern = os.path.basename(row["Image"])[:-len(".nii.gz")]
                msk_pattern = os.path.basename(row["Mask"])[:-len(".nii.gz")]

                # get all resampled segmentations
                for resampled_seg in resampled_files_seg:

                    resampled_seg_file_pattern = os.path.basename(resampled_seg)[:-len("_resampled.nii.gz")]
                    # search for the mask pattern in df
                    if resampled_seg_file_pattern == msk_pattern:

                        # get all resampled images
                        for resampled_img in resampled_files_img:

                            resampled_img_file_pattern = os.path.basename(resampled_img)[:-len("_resampled.nii.gz")]
                            # search here resampled img and resampled mask are fitting to the nun resampled file pattern as they do not match to each other
                            if (resampled_seg_file_pattern == msk_pattern) and (resampled_img_file_pattern == img_pattern):
                                # check for transformation and different scanner protocols - if so, add transformation to ID
                                # if len(self.scanner_protocols) > 0:
                                #     transformation = row["Image_Transformation"].values[0] + "_resampled_" + str(self.isotropic_scale) + "_mm"
                                # else:
                                #     transformation = "resampled_" + str(self.isotropic_scale) + "_mm"

                                self.segprocessor.df = super().add_new_csv_entry(ID=row["ID"],
                                                                                 img_path=os.path.abspath(
                                                                                     resampled_img),
                                                                                 seg_path=os.path.abspath(
                                                                                     resampled_seg),
                                                                                 modality=row["Modality"],
                                                                                 roi_label=row["ROI_Label"],
                                                                                 image_transformation=row[
                                                                                     "Image_Transformation"],
                                                                                 # transformation,
                                                                                 mask_transformation=row[
                                                                                     "Mask_Transformation"],
                                                                                 timepoint=row["Timepoint"],
                                                                                 rater=row["Rater"],
                                                                                 prediction_label=row[
                                                                                     "Prediction_Label"],
                                                                                 df=self.segprocessor.df)

                                idx_to_drop.append(i)
                                break

                        break

            # drop all the non resampled samples
            self.segprocessor.df = self.segprocessor.df.drop(list(set(idx_to_drop)))

        else:
            self.segprocessor.logger.info(
                "Image and Mask Transformation already exist in the input data. Skipping resampling...")

        self.segprocessor.logger.info("### Resampling Done!")
        print("Resampling Successful!")

    def resample_image_with_isotropic_spacing(self, segs):
        """
        Read image and create a downsampled version according to DownsamplingFactor for Segmentation.
        """

        resampled = []

        if len(segs) == 1:
            des = 'Resampling Seg ' + str(os.path.basename(self.segprocessor.segPath))
        else:
            des = 'Resampling Seg Files'

        for seg in tqdm.tqdm(segs, desc=des):
            seg_ = sitk.GetImageFromArray(seg)
            # get image properties and set new size and (isotropic) spacing
            input_spacing = np.array(seg_.GetSpacing())
            image_size = np.array(seg_.GetSize())

            desired_sampling = tuple(np.array(3 * [self.segprocessor.isotropic_scale]))
            output_image_size = np.floor((input_spacing / desired_sampling) * image_size)
            output_image_size = np.array(np.round(output_image_size).astype(np.uint32)).tolist()

            # Resample filter to isotropic scaling
            itk_resampler = sitk.ResampleImageFilter()
            itk_resampler.SetInterpolator(sitk.sitkNearestNeighbor)

            itk_resampler.SetDefaultPixelValue(0)
            itk_resampler.SetOutputOrigin(seg_.GetOrigin())
            itk_resampler.SetOutputDirection(seg_.GetDirection())
            itk_resampler.SetOutputSpacing(desired_sampling)
            itk_resampler.SetSize(output_image_size)

            resampled_img = itk_resampler.Execute(seg_)

            resampled.append(resampled_img)
        return resampled

    def check_resampled_images(self):
        """
        Check if the resampled images and segmentations are integrated into the dataframe.
        :return:
        """

        # check integration of resampled images and segmentations into the dataframe which replaces the original images and segmentations
        resampled_img = glob.glob(self.segprocessor.out_path + "/resampled/img/*.nii.gz")
        resampled_msk = glob.glob(self.segprocessor.out_path + "/resampled/seg/*.nii.gz")

        resampled_files_msk_files = [os.path.basename(file_path)[:-len("_resampled.nii.gz")] for file_path in
                                     resampled_msk]
        resampled_files_img_files = [os.path.basename(file_path)[:-len("_resampled.nii.gz")] for file_path in
                                     resampled_img]

        resample_img_mapper = pd.DataFrame({"Path": resampled_img, "File": resampled_files_img_files})
        resample_msk_mapper = pd.DataFrame({"Path": resampled_msk, "File": resampled_files_msk_files})

        if len(resampled_files_img_files) > 0 and len(resampled_files_msk_files) > 0:
            for i, row in self.segprocessor.df.iterrows():
                if os.path.basename(row["Image"])[:-len(".nii.gz")] in resample_img_mapper["File"].values:
                    if os.path.basename(row["Mask"])[:-len(".nii.gz")] in resample_msk_mapper["File"].values:
                        # replace normal images with resampled images
                        self.segprocessor.df.at[i, "Image"] = resample_img_mapper.loc[
                            resample_img_mapper["File"] == os.path.basename(row["Image"])[
                                                            :-len(".nii.gz")], "Path"].values[0]
                        self.segprocessor.df.at[i, "Mask"] = resample_msk_mapper.loc[
                            resample_msk_mapper["File"] == os.path.basename(row["Mask"])[
                                                            :-len(".nii.gz")], "Path"].values[0]

    def process_resampling(self):
        """
        Execute resampling of MRI Images
        :return:
        """
        # 1. Step in main function - resampling
        self.exe_resampling()
        self.check_resampled_images()


class Normalization_processing(SegProcessor):
    """
    Class to execute normalization of MRI Images
    """

    def __init__(self, segprocessor):
        self.segprocessor = segprocessor

    def process_normalization(self):
        """
        Execute normalization of MRI Images
        :parameter
        """

        # 2. Step in main function - normalization
        self.segprocessor.normalized_out = self.segprocessor.out_path + "/normalized"
        Path(self.segprocessor.normalized_out).mkdir(parents=True, exist_ok=True)

        if "Image_Transformation" in self.segprocessor.df.columns:
            df = self.segprocessor.df.loc[self.segprocessor.df["Image_Transformation"].isna()]
            mri_images_list = df.loc[df["Modality"] == "MR", "Image"].values.tolist()
        else:
            mri_images_list = self.segprocessor.df.loc[
                self.segprocessor.df["Modality"] == "MR", "Image"].values.tolist()

        # if MRIs are in the dataset
        if len(mri_images_list) > 0:
            self.segprocessor.logger.info("### Start MRI Normalization ###")
            self.segprocessor.logger.info(
                "Applying Normalization Methods: {}".format(self.segprocessor.normalization_method))
            print("Performing {} Normalisation of MR Images ...".format(self.segprocessor.normalization_method))

            if "z_score" in self.segprocessor.normalization_method:
                # Check for already normalized images
                mri_images_z_list = super().get_processed_files(input_files=mri_images_list,
                                                                out_folder=self.segprocessor.normalized_out,
                                                                file_ending="_z_score_normalized.nii.gz")

                print("Need to perform z normalization for {} MR Images ...".format(len(mri_images_z_list)))

                # Perform Z-Score Normalization
                MR_Normalizer(img_paths=mri_images_z_list,
                              logger=self.segprocessor.logger,
                              outpath=self.segprocessor.normalized_out,
                              n_cpu=self.segprocessor.n_cpu).z_normalize_imge_exe()

            if "n4bf" in self.segprocessor.normalization_method:
                # Check for already normalized images
                mri_images_n4bf_list = super().get_processed_files(input_files=mri_images_list,
                                                                   out_folder=self.segprocessor.normalized_out,
                                                                   file_ending="_N4BF_normalized.nii.gz")

                print("Need to perform n4bf normalization for {} MR Images ...".format(len(mri_images_n4bf_list)))

                # Perform N4 Bias Field Correction
                MR_Normalizer(img_paths=mri_images_n4bf_list,
                              logger=self.segprocessor.logger,
                              outpath=self.segprocessor.normalized_out,
                              n_cpu=self.segprocessor.n_cpu).N4BF_exe()

            self.segprocessor.logger.info(str(len(mri_images_list)) + " MR Images got normalized!")

            for mri in mri_images_list:

                if "z_score" in self.segprocessor.normalization_method:
                    # Add Z-Score Normalized Images to CSV
                    z_score_norm_img = os.path.abspath(self.segprocessor.normalized_out) + \
                                       "/" + os.path.basename(mri)[
                                             :-(len(".nii.gz"))] + "_z_score_normalized.nii.gz"
                    transform = "Z-score_normalized"

                    # check for transformation and different scanner protocols
                    if len(self.segprocessor.scanner_protocols) > 0:
                        for i in range(len(self.segprocessor.df.loc[
                                               self.segprocessor.df["Image"] == mri, "Image_Transformation"].values)):
                            if not pd.isna(self.segprocessor.df.loc[
                                               self.segprocessor.df["Image"] == mri, "Image_Transformation"].values[i]):
                                # Add transformation on existing transformation
                                if len(self.segprocessor.df.loc[
                                           self.segprocessor.df["Image"] == mri, "Image_Transformation"].values[0]) > 0:
                                    transform = self.segprocessor.df.loc[self.segprocessor.df[
                                                                             "Image"] == mri, "Image_Transformation"].values[
                                                    0] + \
                                                "_" + "Z-score_normalized"

                    self.segprocessor.df = super().add_new_csv_entry(
                        ID=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "ID"].values[0],
                        img_path=z_score_norm_img,
                        seg_path=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Mask"].values[0],
                        modality=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Modality"].values[0],
                        roi_label=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "ROI_Label"].values[0],
                        image_transformation=transform,
                        mask_transformation=self.segprocessor.df.loc[
                            self.segprocessor.df["Image"] == mri, "Mask_Transformation"].values[0],
                        timepoint=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Timepoint"].values[0],
                        rater=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Rater"].values[0],
                        prediction_label=self.segprocessor.df.loc[
                            self.segprocessor.df["Image"] == mri, "Prediction_Label"].values[0],
                        df=self.segprocessor.df)

                if "n4bf" in self.segprocessor.normalization_method:
                    # Add N4 Bias Field Corrected Images to CSV
                    mri_file_pattern = os.path.basename(mri)[:-(len(".nii.gz"))]
                    n4bc_img = os.path.abspath(
                        self.segprocessor.normalized_out) + "/" + mri_file_pattern + "_N4BF_normalized.nii.gz"

                    transform = "N4BF_normalized"

                    # check for transformation and different scanner protocols - if so, add transformation to ID
                    if len(self.segprocessor.scanner_protocols) > 0:
                        for i in range(len(self.segprocessor.df.loc[
                                               self.segprocessor.df["Image"] == mri, "Image_Transformation"].values)):
                            if not pd.isna(self.segprocessor.df.loc[
                                               self.segprocessor.df["Image"] == mri, "Image_Transformation"].values[i]):
                                # Add transformation on existing transformation
                                if len(self.segprocessor.df.loc[
                                           self.segprocessor.df["Image"] == mri, "Image_Transformation"].values[0]) > 0:
                                    transform = self.segprocessor.df.loc[self.segprocessor.df[
                                                                             "Image"] == mri, "Image_Transformation"].values[
                                                    0] + \
                                                "_" + "N4BF_normalized"

                    self.segprocessor.df = super().add_new_csv_entry(
                        ID=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "ID"].values[0],
                        img_path=n4bc_img,
                        seg_path=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Mask"].values[0],
                        modality=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Modality"].values[0],
                        roi_label=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "ROI_Label"].values[0],
                        image_transformation=transform,
                        mask_transformation=self.segprocessor.df.loc[
                            self.segprocessor.df["Image"] == mri, "Mask_Transformation"].values[0],
                        timepoint=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Timepoint"].values[0],
                        rater=self.segprocessor.df.loc[self.segprocessor.df["Image"] == mri, "Rater"].values[0],
                        prediction_label=self.segprocessor.df.loc[
                            self.segprocessor.df["Image"] == mri, "Prediction_Label"].values[0],
                        df=self.segprocessor.df)

            self.segprocessor.logger.info("#### MRI Normalization Done!")
            print("MRI Normalization Successful!")
        else:
            if "MR" not in self.segprocessor.df["Modality"].values:
                self.segprocessor.logger.info(
                    "No MRI found for normalization! Check modality or disable normalization. Skipping ...")
            else:
                self.segprocessor.error.warning("MRI found but normalization not possible!")

        if "MR" in self.segprocessor.df["Modality"].values:
            self.segprocessor.logger.info("Replace original Images with normalized Images!")
            # drop all non-normalized images
            # check for non-normalized images
            non_transformed_img = self.segprocessor.df.copy().loc[~self.segprocessor.df["Image"].str.contains("normalized"),"Image"].unique() # Get unique 'Image' values from non normalized
            # drop non normalized images
            self.segprocessor.df = self.segprocessor.df.copy().loc[~self.segprocessor.df['Image'].isin(non_transformed_img)]
            del non_transformed_img

            if "N4BF_normalized" in self.segprocessor.df["Image_Transformation"].values:
                self.segprocessor.df[self.segprocessor.df["Image_Transformation"] == "N4BF_normalized"]['Image_Transformation'] = np.nan

            elif "Z-score_normalized" in self.segprocessor.df["Image_Transformation"].values:
                self.segprocessor.df[self.segprocessor.df["Image_Transformation"] == "Z-score_normalized"]['Image_Transformation'] = np.nan

            else:
                self.segprocessor.error.error("No Normalized Images found in CSV! Normalization Failed!")
                raise ValueError("No Normalized Images found in CSV! Normalization Failed!")

            # Check if there are raw samples with only Normalization and no Mask Transformation
            raw_samples = self.segprocessor.df.copy().loc[self.segprocessor.df["Image"].isnull() & self.segprocessor.df["Mask"].isnull()]
            
            # check in folder multi_label_seg for the correct segmentations
            preprocessed_masks = glob.glob(self.segprocessor.out_path + "/multilabel_seg/*.nii.gz")
            # check in normalize folder for correct images (if resampling is enabled it has been applied to the images already)
            normalized_images =  glob.glob(self.segprocessor.out_path + "/normalized/*.nii.gz")

            normalized_img_dict = {
                    os.path.basename(image).replace(".nii.gz", ""): image
                    for image in normalized_images
                }
            
            preprocessed_mask_dict = {
                    os.path.basename(mask).replace(".nii.gz", ""): mask
                    for mask in preprocessed_masks
                }

            non_normalized_images_exist = self.segprocessor.df.copy()['Image'].str.contains('normalized').any() == False

            # Check if all images are normalized - replace for non transfomred samples - not normalized transformations need to get redone
            if non_normalized_images_exist:
                print("Detected Images that are not normalized. Trying to replace ... ")
                self.segprocessor.error.warning("Detected Images that are not normalized. Trying to replace ... ")

                for i, row in tqdm.tqdm(self.segprocessor.df.copy().iterrows(), total=len(self.segprocessor.df), desc="Integrate missing normalized images"):
                    if "normalized" not in row["Image"]:
                        if row["Image_Transformation"].isnull():
                            # Extract the non-normalized base name (without the extension)
                            non_normalized_base = os.path.basename(row['Image']).replace(".nii.gz", "")

                            # Find the corresponding normalized image path by matching the prefix
                            matched_normalized = next(
                                (norm_path for norm_prefix, norm_path in normalized_img_dict.items() if norm_prefix.startswith(non_normalized_base)),
                                None
                            )

                            # Replace the full path in the original DataFrame if a match is found
                            if not matched_normalized is None:
                                self.segprocessor.df.at[i, 'Image'] = matched_normalized
                            else:
                                print("Could not find normalized Images for", non_normalized_base, ". Check file naming or errors during normalization.")
                                self.segprocessor.error.error("Could not find normalized Images for", non_normalized_base, ". Check file naming or errors during normalization.")
                        else:
                            print("Not normalized transformed Image detected! need to redo transformation and remove image ...")
                            self.segprocessor.error.error("Not normalized transformed Image detected! need to redo transformation and remove image ...")
                            raise ValueError("Not normalized transformed Image detected! need to redo transformation and remove image ...")
                            # Drop rows where the 'Image' column matches the current image path
                            #self.segprocessor.df = self.segprocessor.df[self.segprocessor.df["Image"] != row["Image"]]
                            # Delete the image file if it exists
                            #if os.path.exists(row["Image"]):
                            #    os.remove(row["Image"])
                            #    print(f"Deleted image file: {row["Image"]}")
                            #    self.error.warning(f"Deleted image file: {row["Image"]}")

            else:
                print("All images are normalized.")
                self.segprocessor.logger.info("All images are normalized.")

            # Iterate through non-normalized images and replace them if a match exists
            if len(raw_samples) == 0:
                print("Could not find any non-transformed sample in the dataset. Trying to add them ...")

                updated_rows = []
                for i,row in  tqdm.tqdm(self.segprocessor.df.copy().iterrows(),total=len(self.segprocessor.df), desc="Add missing non-transformed samples"):
                    if row["Mask_Transformation"] == "Peritumoral":
                        mask_name = os.path.basename(mask_path)[:-len("_3_peritumoral.nii.gz")]
                         # Find the corresponding normalized image path by matching the prefix
                        matched_preprocessed_mask = next(
                            (preprocessed_path for preprocessed_prefix, preprocessed_path in preprocessed_mask_dict.items() 
                            if preprocessed_prefix.startswith(mask_name)),
                            None
                            )
                        if not matched_preprocessed_mask is None:
                            new_row = row.copy()
                            new_row["Mask"] = matched_preprocessed_mask
                            new_row["Mask_Transformation"] = np.nan

                            if not new_row["Image_Transformation"].isnull():
                                print("Detected Peritumoral Sample with Image Transformation ...")
                                self.error.warning("Detected Peritumoral Sample with Image Transformation ...")
                                new_row["Image_Transformation"] = np.nan
                                non_normalized_base = os.path.basename(new_row['Image']).replace(".nii.gz", "")

                                # Find the corresponding normalized image path by matching the prefix
                                matched_normalized = next(
                                    (norm_path for norm_prefix, norm_path in normalized_img_dict.items() if norm_prefix.startswith(non_normalized_base)),
                                    None
                                )
                                new_row['Image'] = matched_normalized
                            updated_rows.append((i, new_row))
                
                # Create a DataFrame from the updated rows, keeping the same index
                non_transformed_df = pd.DataFrame([row for _, row in updated_rows])
                non_transformed_df.index = [index for index, _ in updated_rows]

                # self.segprocessor.df.index.name = index_name

                # Concatenate the updated DataFrame with the original DataFrame
                self.segprocessor.df = pd.concat([self.segprocessor.df, non_transformed_df], ignore_index=False)
                del non_transformed_df


                # Remove duplicate rows based on 'Image' and 'Mask' columns
                self.segprocessor.df = self.segprocessor.df.drop_duplicates(subset=['Image', 'Mask'], keep='last')  
                
                for i,row in self.segprocessor.df.copy().iterrows():
                    # check for non preprocessed masks and replace them if possible
                    if pd.isnull(row["Mask_Transformation"]):
                        # Check if the mask path contains 'resampled'
                        if 'resampled' in os.path.basename(row['Mask']):
                            mask_path =  row['Mask']
                            # Extract the base name of the resampled mask (without the extension)
                            resampled_base = os.path.basename(mask_path).replace(".nii.gz", "")

                            # Find the corresponding preprocessed mask by matching the prefix
                            matched_preprocessed_mask = next(
                                (preprocessed_path for preprocessed_prefix, preprocessed_path in preprocessed_mask_dict.items() 
                                if preprocessed_prefix.startswith(resampled_base)),
                                None
                            )

                            # Replace the mask path if a match is found
                            if matched_preprocessed_mask:
                                self.segprocessor.df.at[i, 'Mask'] = matched_preprocessed_mask
            else:
                for i,row in  tqdm.tqdm(self.segprocessor.df.copy().iterrows(),total=len(self.segprocessor.df), desc="check for preprocessed masks"):
                    if  pd.isnull(row["Mask_Transformation"]):
                        # Check if the mask path contains '/resampled/'
                        if 'resampled' in os.path.basename(row['Mask']):
                            mask_path =  row['Mask']
                            # Extract the base name of the resampled mask (without the extension)
                            resampled_base = os.path.basename(mask_path).replace(".nii.gz", "")

                            # Find the corresponding preprocessed mask by matching the prefix
                            matched_preprocessed_mask = next(
                                (preprocessed_path for preprocessed_prefix, preprocessed_path in preprocessed_mask_dict.items() 
                                if preprocessed_prefix.startswith(resampled_base)),
                                None
                            )

                            # Replace the mask path if a match is found
                            if matched_preprocessed_mask:
                                self.segprocessor.df.at[i, 'Mask'] = matched_preprocessed_mask

                # for i,r in non_normalized_images.iterrows():

                # add samples without image transformation to these data and get prediction label from data for new entry


            # else:
            #    self.logger.info("No MRI found for normalization! Check modality or disable normalization. Skipping ...")


class Seg_filter_processing(SegProcessor):
    """
        This class is used to filter the segmentation content based on the following criteria:
    """

    def __init__(self,segprocessor):

        self.segprocessor = segprocessor
        self.fast_mode = self.segprocessor.fast_mode

    def filter_segmentations(self, segmentation_paths: list = None):
        """
        Filter segmentations that are either empty or have less than roi_threshold foreground pixels.
        :param segmentation_paths: List of file paths to segmentation files.
        :return: List of file paths to valid segmentations.
        """

        print("#### Start Segmentation preprocessing ... ")
        # if no segmentation paths are given, use the ones from the dataframe
        if segmentation_paths is None:
            segmentation_paths = self.segprocessor.df["Mask"].tolist()

        # it has already been filtered on the previous output

        removed = 0
        # TODO: currently not checking as it might be not correct
        # Process files using multiple CPUs
        #with Pool(processes=self.segprocessor.n_cpu) as pool:
        #    # Use tqdm with imap to show progress
        #    for result in tqdm.tqdm(pool.imap(self.is_valid_segmentation, segmentation_paths),
        #                            total=len(segmentation_paths), desc="Segmentation Quality Check"):
        #        if not result[0]:
        #            self.segprocessor.error.warning(f"Segmentation {result[1]} is too small or empty. Removing from preprocessing file ...")
        #            print(f"Segmentation {result[1]} is too small or empty. Removing from preprocessing file ...")
        #            self.segprocessor.df = self.segprocessor.df[self.segprocessor.df["Mask"] != result[1]]
        #            removed += 1

        if self.segprocessor.df.empty:
            self.segprocessor.error.error("All segmentations are too small or empty. Please check you input data.")
            raise ValueError("All segmentations are too small or empty. Please check you input data.")
        if removed > 0:
            print(removed, "Segmentations removed after filtering!")
            self.segprocessor.logger.info(f"{removed} Segmentations removed after filtering!")

    def is_valid_segmentation(self, segmentation: str):
        """
        Check if a segmentation is valid.
        :param segmentation: Path to the segmentation file.
        :return: bool for fitted criteria, segmentation path
        """

        seg_object = self.load_segmentation(segmentation_path=segmentation)

        # check if the segmentation could be loaded
        if seg_object is None:
            return False, segmentation

        # check if the segmentation is empty or has less than roi_threshold foreground pixels
        if np.sum(seg_object == 1) >= self.segprocessor.roi_threshold:

            return True, segmentation
        else:
            self.segprocessor.error.error(f"Segmentation {os.path.basename(segmentation)} has size of {str(np.sum(seg_object == 1))} which is smaller than min size threshold {str(self.segprocessor.roi_threshold)}.")
            return False, segmentation

    def load_segmentation(self, segmentation_path: str):
        """
        Load a segmentation from a NIfTI file.
        :param segmentation_path (str): Path to the segmentation file.
        :return: ndarray or None: Segmentation data, or None if loading failed.
        """

        try:
            segmentation_data = nib.load(segmentation_path).get_fdata()
            return segmentation_data
        except Exception as e:
            self.segprocessor.error.error(f"Failed loading segmentation from {segmentation_path}: {e}")
            print(f"Failed loading segmentation from {segmentation_path}: {e}")
            return None

    def process_segmentation_filtering(self):
        """
        This function filters the segmentation content based on the following criteria:
        :parameter
        """
        # 3. Step in main function - segmentation filtering

        # Check if segmentations are empty
        self.segprocessor.logger.info("### Start Segmentation Filtering ###")

        self.segprocessor.logger.info("Config of Segmentation Filtering:\n" +
                                      "\t\t\t\t\t\t\t\t\t\t\t\t\t\tROI Threshold: {}\n".format(
                                          self.segprocessor.roi_threshold) +
                                      "\t\t\t\t\t\t\t\t\t\t\t\t\t\tMax Number of ROIs: {}\n".format(
                                          self.segprocessor.max_num_rois) +
                                      "\t\t\t\t\t\t\t\t\t\t\t\t\t\tResampling: {}\n".format(
                                          self.segprocessor.resampling))

        self.segprocessor.multilabel_seg = self.segprocessor.out_path + "/multilabel_seg"
        Path(self.segprocessor.multilabel_seg).mkdir(parents=True, exist_ok=True)

        # 1. get all files in outfolder
        filtered_segmentations = glob.glob(self.segprocessor.multilabel_seg + "/*.nii.gz")

        # 2. get prefix of input files
        input_msks_prefix = []

        for msk in self.segprocessor.df["Mask"].values.tolist():
            input_msks_prefix.append(os.path.basename(msk)[:-(len(".nii.gz"))])

        # 3. check if prefix is in multilabel_seg folder
        to_process = self.segprocessor.df.copy()
        for msk in tqdm.tqdm(to_process.copy()["Mask"].values.tolist(), desc="Filtering Segmentations"):
            msk_pattern = os.path.basename(msk)[:-(len(".nii.gz"))]
            for done_msk in filtered_segmentations:
                # done_msks_pattern = os.path.basename(done_msk)[:-(len("_roiL_0_roiT_3_roiN_1.nii.gz"))]
                done_msk_file = os.path.basename(done_msk)
                if done_msk_file.startswith(msk_pattern): # msk_pattern == done_msks_pattern:
                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Mask"] = os.path.abspath(done_msk)
                    to_process.drop(to_process[to_process["Mask"] == msk].index, inplace=True)
                    break

        if len(to_process) > 0:
            self.segprocessor.logger.info(
                "Need to filter {} segmentations".format(len(to_process["Mask"].values.tolist())))
            print("Need to filter {} segmentations".format(len(to_process["Mask"].values.tolist())))

            filtered = SegmentationFilter(df=to_process,
                                          multilabel_seg_path=self.segprocessor.multilabel_seg,
                                          out_path=self.segprocessor.out_path,
                                          n_cpus=self.segprocessor.n_cpu,
                                          logger=self.segprocessor.logger,
                                          error=self.segprocessor.error,
                                          runid=self.segprocessor.RunID,
                                          roi_threshold=self.segprocessor.roi_threshold,
                                          max_num_rois=self.segprocessor.max_num_rois,
                                          resampling=self.segprocessor.resampling,
                                          scanner_protocols=self.segprocessor.scanner_protocols,
                                          consider_multiple_labels=self.segprocessor.consider_multiple_labels,
                                          use_previous_output=self.segprocessor.use_previous_output,
                                          fast_mode=self.fast_mode).process()

            if len(filtered) != 0:
                self.segprocessor.df = filtered

        self.segprocessor.logger.info("#### Segmentation Filtering Done!")
        print("Segmentation Filtering Successful!")


class Segmentation_perturbation_processing(SegProcessor):
    """
        This class is used to perturb the segmentation content based on the following criteria:
    """

    def __init__(self, segprocessor):
        self.segprocessor = segprocessor
        self.logger = self.segprocessor.logger
        self.error = self.segprocessor.error
        self.pertubration_closing_correction = self.segprocessor.pertubration_closing_correction
        self.pertubration_convex_correction = self.segprocessor.pertubration_convex_correction

    def calc_Dice_4_seg_pertu(self, pert_path: str):
        """
        Calculates a 3D Dice for the given segmentation and the perturbed segmentation
        :param pert_path: path to file which got perturbated
        :return acc_pert_num, non_acc_pert_num: number of accepted perturbations and non-accepted perturbations
        """

        acc_pert_num = 0
        non_acc_pert_num = 0
        if "_dice_" not in pert_path:

            # Cosed Expansion
            if pert_path.endswith("_closed.nii.gz"):
                # Calc Dice for postprocessed closed segmentation
                accepted, not_accepted = self.calc_dice4control(perturbed_seg=pert_path,
                                                                # calculated_dice_pert=pert_with_dice,
                                                                file_ending="_closed.nii.gz")
                acc_pert_num += accepted
                non_acc_pert_num += not_accepted
                
            # Convex Hull Expansion
            elif pert_path.endswith("_convex.nii.gz"):
                # Calc Dice for postprocessed convex segmentation
                accepted, not_accepted = self.calc_dice4control(perturbed_seg=pert_path,
                                                                # calculated_dice_pert=pert_with_dice,
                                                                file_ending="_convex.nii.gz")
                acc_pert_num += accepted
                non_acc_pert_num += not_accepted

            # Connected Component Expansion
            elif pert_path.endswith("_expanded.nii.gz"):
                # Calc Dice for extended Masks
                accepted, not_accepted = self.calc_dice4control(perturbed_seg=pert_path,
                                                                # calculated_dice_pert=pert_with_dice,
                                                                file_ending="_expanded.nii.gz")
                acc_pert_num += accepted
                non_acc_pert_num += not_accepted

            # Random Supervoxel Expansion
            elif pert_path.endswith("_morph.nii.gz"):
                # Calc Dice for extended Masks
                accepted, not_accepted = self.calc_dice4control(perturbed_seg=pert_path,
                                                                # calculated_dice_pert=pert_with_dice,
                                                                file_ending="_morph.nii.gz")
                acc_pert_num += accepted
                non_acc_pert_num += not_accepted
            else:
                # get file ending for random_walker         
                for i in range(1, self.segprocessor.roi_pert_repetition):
                    file_ending = "random_walker_" + str(i) + ".nii.gz"
                    # Calc Dice for extended Masks
                    accepted, not_accepted = self.calc_dice4control(perturbed_seg=pert_path,
                                                                    # calculated_dice_pert=pert_with_dice,
                                                                    file_ending=file_ending)
                    acc_pert_num += accepted
                    non_acc_pert_num += not_accepted
        else:
            # get dice from file name
            match = re.search(r"_dice_(\d+\.\d+)", os.path.basename(pert_path))

            if match:
                try:
                    dice_value = float(match.group(1))  # Convert extracted string to float
                    if self.dice_threshold <= dice_value < 0.99:
                        self.logger.info(f"Dice score {dice_value} is within the accepted range. {os.path.basename(pert_path)}")
                        acc_pert_num += 1
                    else:
                        self.logger.info(f"Dice score {dice_value} is OUT of the accepted range. {os.path.basename(pert_path)}")
                        non_acc_pert_num += 1
                except:
                    self.error.warning(f"Could not find Dice score in file name! {os.path.basename(pert_path)}")
            else:
                self.error.warning("No Dice score found in the filename.")

        return acc_pert_num, non_acc_pert_num


    def process_connected_component_expansion(self):
        """
        This function expands the connected components of the segmentations
        :return:
        """
        self.segprocessor.logger.info("### Connected Component Expansion")
        print("### Starting Connected Component Expansion ...")

        if "Image_Transformation" in self.segprocessor.df.columns:
            # input_files = list(set(self.df.loc[self.df["Mask_Transformation"].isnull(),"Mask"].to_list()))
            tmp = self.segprocessor.df.loc[self.segprocessor.df["Mask_Transformation"].isna(), :].copy()
            tmp = tmp.loc[tmp["Image_Transformation"].isna(), :]
            input_files = list(set(tmp["Mask"].to_list()))
        else:
            input_files = list(set(self.segprocessor.df["Mask"].to_list()))

        # Check for already expanded segmentations
        maks4extended = self.segprocessor.get_processed_files(input_files=input_files,
                                                              out_folder=self.segprocessor.out_path + "/perturbed_seg",
                                                              file_ending="_" + str(
                                                                  self.segprocessor.expand_seg_dist[
                                                                      0]) + "_expanded.nii.gz")

        if len(maks4extended) > 0:
            self.segprocessor.logger.info("Need to perform " + str(len(maks4extended)) + " perturbations!")
            print("Need to perform " + str(len(maks4extended)) + " perturbations!")

            CCExtender(seg_paths=maks4extended,
                       output=self.segprocessor.out_path + "/perturbed_seg",
                       expand_distance=self.segprocessor.expand_seg_dist,
                       logger=self.segprocessor.logger,
                       n_cpu=self.segprocessor.n_cpu,
                       chunksize=self.segprocessor.chunksize,
                       error=self.segprocessor.error,
                       ).exe()

        self.segprocessor.logger.info("### Connected Component Expansion Done!")
        print("Connected Component Expansion Successful!")

    def process_surround_segmentation(self):
        """
        This function expands the segmentations to the surrounding area
        :return:
        """

        self.segprocessor.peritumor_seg_dir = self.segprocessor.out_path + "/peritumor_seg"
        Path(self.segprocessor.peritumor_seg_dir).mkdir(parents=True, exist_ok=True)  # saving peritumoral segmentations

        self.segprocessor.logger.info("### Surrounding Expansion")
        print("### Starting Segmentation Surrounding Expansion ...")
        # Check for already expanded segmentations

        peritumoral_files = glob.glob(self.segprocessor.peritumor_seg_dir + "/*_peritumoral.nii.gz")
        if len(peritumoral_files) > 0:
            print("Found " + str(len(peritumoral_files)) + " peritumoral segmentations")

        # check for non changed segmentations as input from the previous run
        list_of_to_perform_surronding_segmentations = []
        for msk in set(self.segprocessor.df["Mask"].values.tolist()):
            if "Image_Transformation" in self.segprocessor.df.columns:
                msk_df = self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, :]
                msk_df = msk_df.loc[msk_df["Image_Transformation"].isna(), :]
                msk_df = msk_df.loc[msk_df["Mask_Transformation"].isna(), :]

                if len(msk_df) > 0:
                    list_of_to_perform_surronding_segmentations.append(msk)
            else:
                msk_df = self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, :]
                msk_df = msk_df.loc[msk_df["Mask_Transformation"].isna(), :]
                if len(msk_df) > 0:
                    list_of_to_perform_surronding_segmentations.append(msk)

        maks4peri = self.performed_perturbations(list_of_out_files=peritumoral_files,
                                                 list_of_input_files=list_of_to_perform_surronding_segmentations,
                                                 file_ending="*_peritumoral.nii.gz")

        if len(maks4peri) > 0:
            print("Need to perform " + str(len(maks4peri)) + " calculations!")

            # Generate peritumoral/surrounding segmentations
            PeriTumoralSegmentator(seg_paths=maks4peri,
                                   logger=self.segprocessor.logger,
                                   n_cpu=self.segprocessor.n_cpu,
                                   chunksize=self.segprocessor.chunksize,
                                   output=self.segprocessor.peritumor_seg_dir + "/",
                                   expand_seg_dist=self.segprocessor.peri_dist,
                                   error=self.segprocessor.error,
                                   fast_mode=self.segprocessor.fast_mode,
                                   ).exe()
        else:
            print("All surrounding expansions have been perturbated already!")

        peritumoral_segs = glob.glob(self.segprocessor.peritumor_seg_dir + "/*_peritumoral.nii.gz")

        for msk in tqdm.tqdm(self.segprocessor.df["Mask"].values, total=len(self.segprocessor.df["Mask"]), desc="Loading Surrounding segmentations"):
            for pseg in peritumoral_segs:
                dist = re.findall(r'_([0-9]*[.]*[0-9]*)_peritumoral.nii.gz', os.path.basename(pseg))
                psegid = os.path.basename(pseg)[:-(len("_" + str(dist[0]) + "_peritumoral.nii.gz"))]
                if os.path.basename(msk)[:-len(".nii.gz")] == psegid:
                    #  img_id = os.path.basename(self.df.loc[self.df["Mask"] == msk, "Image"].values[0])[
                    #         :-len(".nii.gz")]

                    self.segprocessor.df = super().add_new_csv_entry(
                        ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                        img_path=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0],
                        seg_path=os.path.abspath(pseg),
                        modality=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[0],
                        roi_label=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0],
                        image_transformation=self.segprocessor.df.loc[
                            self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[0],
                        mask_transformation="Peritumoral",
                        timepoint=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0],
                        rater=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[0],
                        prediction_label=self.segprocessor.df.loc[
                            self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                        df=self.segprocessor.df)

        self.segprocessor.logger.info("### Segmentation Surrounding Expansion Done!")
        print("Segmentation Surrounding Expansion Successful!")

    def process_supervoxel_perturbation(self):
        """
        This function perturbs the supervoxels of the segmentations
        :return:
        """

        self.segprocessor.logger.info("### Random Supervoxel Perturbation")
        print("### Starting Random Supervoxel Perturbation ...")

        # Check for already processed files
        performed_perturbation = glob.glob(self.segprocessor.out_path + "/perturbed_seg/*_morph.nii.gz")
        if len(performed_perturbation) > 0:
            print("Found " + str(len(performed_perturbation)) + " perturbed segmentations")

        performing_perturbation = self.performed_supervoxel_calc(list_of_out_files=performed_perturbation,
                                                                 file_endings=["_morph.nii.gz"])

        # performing_perturbation = performing_perturbation.loc[performing_perturbation["Mask"]]
        print("Need to perform " + str(
            len(performing_perturbation) * self.segprocessor.roi_pert_repetition) + " perturbations!")

        if len(performing_perturbation) > 0:

            SupervoxelPerturbator(img_paths=performing_perturbation["Image"].values.tolist(),
                                  seg_paths=performing_perturbation["Mask"].values.tolist(),
                                  modality=performing_perturbation["Modality"].values.tolist(),
                                  output=self.segprocessor.out_path + "/perturbed_seg/",
                                  logger=self.segprocessor.logger,
                                  n_cpu=self.segprocessor.n_cpu,
                                  chunksize=self.segprocessor.chunksize,
                                  repetition=self.segprocessor.roi_pert_repetition,
                                  distance=self.segprocessor.perturbation_roi_adapt_size,
                                  perturbation_roi_adapt_type=self.segprocessor.perturbation_roi_adapt_type,
                                  error=self.segprocessor.error,
                                  timeout=self.segprocessor.timeout).exe()

            # Removing non used files
            supervoxel_pert_samples = os.listdir(self.segprocessor.out_path + "/perturbed_seg/")

            for item in supervoxel_pert_samples:
                mod = performing_perturbation["Modality"].values.tolist()[0]
                if item.endswith("int.nii.gz"):
                    os.remove(os.path.join(self.segprocessor.out_path + "/perturbed_seg/", item))
                if item.endswith(mod + "_.nii.gz") or item.endswith(mod + "_" + mod + ".nii.gz"):
                    os.remove(os.path.join(self.segprocessor.out_path + "/perturbed_seg/", item))

        self.segprocessor.logger.info("### Random Supervoxel Perturbation Done!")
        print("Random Supervoxel Perturbation Successful!")

    def process_random_walker_perturbation(self):
        """
        This function perturbs the segmentations using the random walker algorithm
        :return:
        """
        self.segprocessor.logger.info("### Random Walker Perturbation")
        print("### Starting Random Walker Perturbation ...")

        # Check for already processed files
        performed_perturbation = glob.glob(self.segprocessor.out_path + "/perturbed_seg/*_random_walker_[0-9]*.nii.gz")
        file_endings = ["_random_walker_" + str(i) + ".nii.gz" for i in range(0, self.segprocessor.roi_pert_repetition)]
        performing_perturbation = self.performed_transformations(list_of_out_files=performed_perturbation,
                                                                 file_endings=file_endings)

        print("Need to perform " + str(
            len(performing_perturbation) * self.segprocessor.roi_pert_repetition) + " perturbations!")

        if len(performing_perturbation) > 0:
            RandomWalker(segs=performing_perturbation["Mask"].values,
                         output=self.segprocessor.out_path + "/perturbed_seg/",
                         logger=self.segprocessor.logger,
                         error=self.segprocessor.error,
                         seed=self.segprocessor.seed,
                         RunID=self.segprocessor.RunID,
                         n_cpu=self.segprocessor.n_cpu,
                         chunksize=self.segprocessor.chunksize,
                         random_walker_iterations=self.segprocessor.roi_pert_repetition,
                         perturbation_factor=self.segprocessor.perturbation_factor,
                         fast_mode=self.segprocessor.fast_mode).exe()

        self.segprocessor.logger.info("### Random Walker Perturbation Done!")
        print("Random Walker Perturbation Successful!")

    def process_segmentation_postprocessing(self):
        """
        This function performs the postprocessing of the perturbed segmentations
        :return:
        """

        # 5. Closing and Convex Hull - Postprocessing
        print("### Starting Postprocessing of Segmentation Perturbations ...")
        self.segprocessor.logger.info("### Segmentation Perturbation Postprocessing")

        if self.pertubration_closing_correction:
            # Correction for Random Walker and Supervoxel based Perturbation
            print("Performing Correction Closing ...")

            # Random Walker based Pertubation - correction
            random_performed_correction_closing = glob.glob(
                self.segprocessor.out_path + "/perturbed_seg/*_random_walker_[0-9]_closed.nii.gz")
            random_input_files = glob.glob(self.segprocessor.out_path + "/perturbed_seg/*_random_walker_[0-9]*.nii.gz")

            # Morphological Supervoxel based Randomization Pertubation - correction
            morph_performed_correction_closing = glob.glob(
                self.segprocessor.out_path + "/perturbed_seg/*_morph_closed.nii.gz")
            morph_input_files = glob.glob(self.segprocessor.out_path + "/perturbed_seg/*_morph.nii.gz")

            input_files = random_input_files + morph_input_files
            performed_correction_closing = random_performed_correction_closing + morph_performed_correction_closing

            performing_correction_closing = self.performed_perturbations(list_of_out_files=performed_correction_closing,
                                                                        list_of_input_files=input_files,
                                                                        file_ending="_closed.nii.gz")

            if len(performing_correction_closing) > 0:
                print("Need to perform correction closing on:", len(performing_correction_closing), "files")

                # Create a multiprocessing pool
                pool = Pool(self.segprocessor.n_cpu)

                # Apply the process function to each item in the list
                with tqdm.tqdm(total=len(performing_correction_closing), desc="Performing Correction Closing") as pbar:
                    for result_ in pool.imap_unordered(self.segmentation_closing, performing_correction_closing):
                        pbar.update(1)

                # Close the pool
                pool.close()
                pool.join()

        if self.pertubration_convex_correction:
            print("Performing Convex Hull Correction ...")

            random_performed_correction_convex_hull = glob.glob(
                self.segprocessor.out_path + "/perturbed_seg/*_random_walker_[0-9]_convex*.nii.gz")
            morph_performed_correction_convex_hull = glob.glob(
                self.segprocessor.out_path + "/perturbed_seg/*_morph_convex.nii.gz")

            performed_correction_convex_hull = random_performed_correction_convex_hull + morph_performed_correction_convex_hull

            performing_correction_convex_hull = self.performed_perturbations(
                                                                        list_of_out_files=performed_correction_convex_hull,
                                                                        list_of_input_files=input_files,
                                                                        file_ending="_convex.nii.gz")
            
            if len(performing_correction_convex_hull) > 0:
                print("Need to perform correction convex hull on:", len(performing_correction_convex_hull), "files")

                # Create a multiprocessing pool
                pool = Pool(self.segprocessor.n_cpu)

                # Apply the process function to each item in the list
                with tqdm.tqdm(total=len(performing_correction_convex_hull),
                            desc="Performing Convex Hull Correction") as pbar:
                    for _ in pool.imap_unordered(self.segmentation_convex_hull, performing_correction_convex_hull):
                        pbar.update(1)

                # Close the pool
                pool.close()
                pool.join()

        print("Calculate Dice for Segmentation Perturbations ...")
        self.segprocessor.logger.info("### Calculate Dice for Segmentation Perturbations")

        all_pert_list, all_pert_num = self.get_all_perturbed_segs()
        acc_pert_list, acc_pert_num = self.get_accepted_perturbed_segs()
        non_acc_pert_list, non_acc_pert_num = self.get_not_accepted_perturbed_segs()

        pert_with_dice = acc_pert_list + non_acc_pert_list

        # Create a list of string patterns from file names of the perturbated segmentations
        pert_file_pattern_df = pd.DataFrame({"Perturbated_File_Path": [], "Perturbated_File_Pattern": []})
        for pert_path in all_pert_list:
            pert_file_pattern = os.path.basename(pert_path)[:-len(".nii.gz")]
            pert_file_pattern_df = pd.concat(
                [pert_file_pattern_df, pd.DataFrame({"Perturbated_File_Path": [pert_path],
                                                     "Perturbated_File_Pattern": [pert_file_pattern]})])

        # check how many perturbed segmentations do not have a dice and need to get calculated
        if len(pert_with_dice) > 0:
            print("Already calculated Dice for", str(len(pert_with_dice)), "perturbations")
            self.segprocessor.logger.info("Already calculated Dice for " + str(len(pert_with_dice)) + " perturbations")

            for pert_with_dice_path in pert_with_dice:
                # get the dice score
                pert_with_dice_file = os.path.basename(pert_with_dice_path)
                dice_score = re.findall(r"_dice_([0-9]*.[0-9]*).nii.gz", pert_with_dice_file)
                pert_with_dice_file_pattern = pert_with_dice_file[:-len("_dice_" + dice_score[0] + ".nii.gz")]

                if pert_with_dice_file_pattern in pert_file_pattern_df["Perturbated_File_Pattern"].values:
                    # print(pert_with_dice_file_pattern in pert_file_pattern_df["Perturbated_File_Pattern"].values)
                    pert_file_pattern_df = pert_file_pattern_df[
                        pert_file_pattern_df["Perturbated_File_Pattern"] != pert_with_dice_file_pattern]

        if len(pert_file_pattern_df) > 0:
            print("Need to calculate Dice for", str(len(pert_file_pattern_df)), "perturbations")
            self.segprocessor.logger.info(
                "Need to calculate Dice for " + str(len(pert_file_pattern_df)) + " perturbations")

        # not all perturbed segmentations do have a dice
        if (acc_pert_num + non_acc_pert_num) != all_pert_num:

            # Use functools.partial to pass constant_value to my_function
            # partial_function = partial(self.calc_Dice_4_seg_pertu, pert_with_dice=pert_with_dice)

            # Process files using multiple CPUs
            with Pool(processes=self.segprocessor.n_cpu) as pool:
                # Use tqdm with imap to show progress
                for result in tqdm.tqdm(pool.imap(self.calc_Dice_4_seg_pertu,
                                                  list(set(pert_file_pattern_df["Perturbated_File_Path"].values))),
                                        total=len(list(set(pert_file_pattern_df["Perturbated_File_Path"].values))),
                                        desc="Calc Dice for Segmentation Perturbations"):
                    if result[0] != 0 and result[1] != 0:
                        acc_pert_num += result[0]
                        non_acc_pert_num += result[1]
        else:
            self.segprocessor.logger.info(
                "All perturbated segmentations have been controlled already. Skipping Dice calculation...")

        # remove images from segmentation folder
        img_in_perturbated_seg_folder = glob.glob(
            self.segprocessor.out_path + "/perturbed_seg/*_" + self.segprocessor.modality + ".nii.gz")

        if len(img_in_perturbated_seg_folder) > 0:
            for file in tqdm.tqdm(img_in_perturbated_seg_folder, desc="Cleaning perturbated segmentations ..."):
                os.remove(file)

        if acc_pert_num == all_pert_num:
            self.segprocessor.error.warning("All perturbated segmentations have been accepted. " +
                                            "You may want to increase the threshold for the Dice calculation or the perturbation parameter.")

        self.segprocessor.logger.info("Dice Filtering Result - Accepted: " + str(acc_pert_num) +
                                      " - Not Accepted: " + str(non_acc_pert_num))

        print("Gathering accepted perturbations ...")

        # Add accepted pertubated Masks to the csv and out_path the csv
        accepted_pertubations = glob.glob(self.segprocessor.out_path + "/accepted_perturbed_seg/*.nii.gz")

        # check if all perturbations are already in the dataframe
        accepted_perturbations_in_data = self.segprocessor.df.loc[
            ~(self.segprocessor.df["Mask_Transformation"].isnull())].copy()
        accepted_perturbations_in_data = accepted_perturbations_in_data.loc[
            ~(accepted_perturbations_in_data["Mask_Transformation"] == "Peritumoral")]

        print("Found Perturbations in folder: ", len(set(accepted_pertubations)))
        print("Found Perturbations in dataframe: ", len(accepted_perturbations_in_data["Mask"].unique().tolist()))

        if len(accepted_perturbations_in_data["Mask"].unique().tolist()) == len(accepted_pertubations) and len(
                accepted_pertubations) > 0:
            print("All accepted perturbations are already in the data:", len(accepted_perturbations_in_data))
            self.segprocessor.logger.info(
                "All accepted perturbations are already in the data:" + str(len(accepted_perturbations_in_data)))
        else:
            # check if the total number of perturbed segmentations is the same between the csv file and the number of files in the folder
            for msk in tqdm.tqdm(
                    self.segprocessor.df.loc[self.segprocessor.df["Mask_Transformation"].isnull(), "Mask"].values,
                    desc="Gathering accepted perturbations"):
                for acc in acc_pert_list:
                    if os.path.basename(msk)[:-len(".nii.gz")] in os.path.basename(acc):
                        # Super_Voxel_Randomization Perturbation
                        if "_morph_" in os.path.basename(acc):
                            if pd.isnull(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"]).all():
                                self.segprocessor.df = super().add_new_csv_entry(
                                    ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                                    img_path=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[
                                        0],
                                    seg_path=os.path.abspath(acc),
                                    modality=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                        0],
                                    roi_label=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[
                                        0],
                                    image_transformation=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[
                                        0],
                                    mask_transformation="Super_Voxel_Randomization",
                                    timepoint=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[
                                        0],
                                    rater="Super_Voxel_Randomization",
                                    prediction_label=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                                    df=self.segprocessor.df)

                            else:
                                self.segprocessor.df = super().add_new_csv_entry(
                                    ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                                    img_path=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[
                                        0],
                                    seg_path=os.path.abspath(acc),
                                    modality=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[0],
                                    roi_label=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[
                                        0],
                                    image_transformation=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[
                                        0],
                                    mask_transformation="Super_Voxel_Randomization",
                                    timepoint=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[
                                        0],
                                    rater=str(
                                        self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[
                                            0]) + "_Super_Voxel_Randomization",
                                    prediction_label=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                                    df=self.segprocessor.df)

                        # Connected_Component_Expansion Perturbation
                        elif "_expanded_" in os.path.basename(acc):
                            if pd.isnull(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"]).all():
                                self.segprocessor.df = super().add_new_csv_entry(
                                    ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                                    img_path=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[
                                        0],
                                    seg_path=os.path.abspath(acc),
                                    modality=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[0],
                                    roi_label=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[
                                        0],
                                    image_transformation=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[
                                        0],
                                    mask_transformation="Connected_Component_Expansion",
                                    timepoint=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[
                                        0],
                                    rater="Connected_Component_Expansion",
                                    prediction_label=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                                    df=self.segprocessor.df)

                            else:
                                self.segprocessor.df = super().add_new_csv_entry(
                                    ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                                    img_path=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[
                                        0],
                                    seg_path=os.path.abspath(acc),
                                    modality=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[0],
                                    roi_label=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[
                                        0],
                                    image_transformation=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[
                                        0],
                                    mask_transformation="Connected_Component_Expansion",
                                    timepoint=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[
                                        0],
                                    rater=str(
                                        self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[
                                            0]) + "_Connected_Component_Expansion",
                                    prediction_label=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                                    df=self.segprocessor.df)

                        # Random_Walker Perturbation
                        elif "_random_walker_" in os.path.basename(acc):
                            if pd.isnull(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"]).all():
                                self.segprocessor.df = super().add_new_csv_entry(
                                    ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                                    img_path=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[
                                        0],
                                    seg_path=os.path.abspath(acc),
                                    modality=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[0],
                                    roi_label=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[
                                        0],
                                    image_transformation=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[
                                        0],
                                    mask_transformation="Random_Walker_Mask_Change",
                                    timepoint=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[
                                        0],
                                    rater="Random_Walker_Mask_Change",
                                    prediction_label=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                                    df=self.segprocessor.df)


                            else:
                                self.segprocessor.df = super().add_new_csv_entry(
                                    ID=self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0],
                                    img_path=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[
                                        0],
                                    seg_path=os.path.abspath(acc),
                                    modality=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[0],
                                    roi_label=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[
                                        0],
                                    image_transformation=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Image_Transformation"].values[
                                        0],
                                    mask_transformation="Random_Walker_Mask_Change",
                                    timepoint=
                                    self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[
                                        0],
                                    rater=str(
                                        self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[
                                            0]) + "_Random_Walker_Mask_Change",
                                    prediction_label=self.segprocessor.df.loc[
                                        self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0],
                                    df=self.segprocessor.df)

        # check if all accepted perturbations are in dataframe
        for accepted_pert in acc_pert_list:
            if accepted_pert not in self.segprocessor.df["Mask"].values:
                self.segprocessor.error.warning("Accepted perturbation " + accepted_pert + " not in dataframe!")
                # raise ValueError("Accepted perturbation " + accepted_pert + " not in dataframe!")

        self.segprocessor.df.to_csv(
            self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv", index=False)
        self.segprocessor.logger.info("### Segmentation Perturbation Postprocessing Done!")
        print("Segmentation Postprocessing Successful!")

    def gather_accepted_perturbations(self, acc_pert_list: list, msk: str):
        """
        Gather all accepted perturbations from the perturbation methods and put it into a df
        :param acc_pert_list: list of accepted perturbations
        :param msk: mask to which the perturbations belong
        :return: True if it worked and False if could not find perturbations
        """

        entry = pd.DataFrame()
        for acc in acc_pert_list:
            if os.path.basename(msk)[:-len(".nii.gz")] in os.path.basename(acc):
                # Super_Voxel_Randomization Perturbation
                if "_morph_" in os.path.basename(acc):
                    if pd.isnull(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"]).all():
                        entry = pd.DataFrame(
                            {"ID": [self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0]],
                             "Image": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0]],
                             "Mask": [os.path.abspath(acc)],
                             "Modality": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                     0]],
                             "ROI_Label": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0]],
                             "Image_Transformation": [self.segprocessor.df.loc[
                                                          self.segprocessor.df[
                                                              "Mask"] == msk, "Image_Transformation"].values[0]],
                             "Mask_Transformation": ["Super_Voxel_Randomization"],
                             "Timepoint": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0]],
                             "Rater": ["Super_Voxel_Randomization"],
                             "Prediction_Label": [self.segprocessor.df.loc[
                                                      self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[
                                                      0]]
                             })

                    else:
                        entry = pd.DataFrame(
                            {"ID": [self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0]],
                             "Image": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0]],
                             "Mask": [os.path.abspath(acc)],
                             "Modality": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                     0]],
                             "ROI_Label": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0]],
                             "Image_Transformation": [self.segprocessor.df.loc[
                                                          self.segprocessor.df[
                                                              "Mask"] == msk, "Image_Transformation"].values[
                                                          0]],
                             "Mask_Transformation": ["Super_Voxel_Randomization"],
                             "Timepoint": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0]],
                             "Rater": [
                                 str(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[0])
                                 + "_Super_Voxel_Randomization"],
                             "Prediction_Label": [
                                 self.segprocessor.df.loc[
                                     self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0]]
                             })

                # Connected_Component_Expansion Perturbation
                elif "_expanded_" in os.path.basename(acc):
                    if pd.isnull(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"]).all():
                        entry = pd.DataFrame(
                            {"ID": [self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0]],
                             "Image": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0]],
                             "Mask": [os.path.abspath(acc)],
                             "Modality": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                     0]],
                             "ROI_Label": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0]],
                             "Image_Transformation": [self.segprocessor.df.loc[
                                                          self.segprocessor.df[
                                                              "Mask"] == msk, "Image_Transformation"].values[
                                                          0]],
                             "Mask_Transformation": ["Connected_Component_Expansion"],
                             "Timepoint": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0]],
                             "Rater": ["Connected_Component_Expansion"],
                             "Prediction_Label": [
                                 self.segprocessor.df.loc[
                                     self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0]]
                             })

                    else:
                        entry = pd.DataFrame(
                            {"ID": [self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0]],
                             "Image": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0]],
                             "Mask": [os.path.abspath(acc)],
                             "Modality": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                     0]],
                             "ROI_Label": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0]],
                             "Image_Transformation": [self.segprocessor.df.loc[
                                                          self.segprocessor.df[
                                                              "Mask"] == msk, "Image_Transformation"].values[
                                                          0]],
                             "Mask_Transformation": ["Connected_Component_Expansion"],
                             "Timepoint": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0]],
                             "Rater": [
                                 str(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[0])
                                 + "_Connected_Component_Expansion"],
                             "Prediction_Label": [
                                 self.segprocessor.df.loc[
                                     self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0]]
                             })

                # Random_Walker Perturbation
                elif "_random_walker_" in os.path.basename(acc):
                    if pd.isnull(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"]).all():
                        entry = pd.DataFrame(
                            {"ID": [self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0]],
                             "Image": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0]],
                             "Mask": [os.path.abspath(acc)],
                             "Modality": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                     0]],
                             "ROI_Label": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0]],
                             "Image_Transformation": [self.segprocessor.df.loc[
                                                          self.segprocessor.df[
                                                              "Mask"] == msk, "Image_Transformation"].values[
                                                          0]],
                             "Mask_Transformation": ["Random_Walker_Mask_Change"],
                             "Timepoint": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0]],
                             "Rater": ["Random_Walker_Mask_Change"],
                             "Prediction_Label": [
                                 self.segprocessor.df.loc[
                                     self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0]]
                             })


                    else:
                        entry = pd.DataFrame(
                            {"ID": [self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ID"].values[0]],
                             "Image": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Image"].values[0]],
                             "Mask": [os.path.abspath(acc)],
                             "Modality": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Modality"].values[
                                     0]],
                             "ROI_Label": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "ROI_Label"].values[0]],
                             "Image_Transformation": [self.segprocessor.df.loc[
                                                          self.segprocessor.df[
                                                              "Mask"] == msk, "Image_Transformation"].values[
                                                          0]],
                             "Mask_Transformation": ["Random_Walker_Mask_Change"],
                             "Timepoint": [
                                 self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Timepoint"].values[0]],
                             "Rater": [tr(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == msk, "Rater"].values[
                                              0]) + "_Random_Walker_Mask_Change"],
                             "Prediction_Label": [
                                 self.segprocessor.df.loc[
                                     self.segprocessor.df["Mask"] == msk, "Prediction_Label"].values[0]]
                             })
            else:
                return False, entry
        return True, entry

    def get_mask_transformation_from_file(self, file):
        """
        Get Mask Transformation from file name
        :param file: file to extract mask transformation from.
        :return mask transformation: Type of mask transformations extracted from file name
        """
        transformation = None

        if "_random_walker_" in os.path.basename(file):
            transformation = "Random_Walker_Mask_Change"
        elif "_expanded_" in os.path.basename(file):
            transformation = "Connected_Component_Expansion"
        elif "_morph_" in os.path.basename(file):
            transformation = "Super_Voxel_Randomization"
        elif "peritumoral" in os.path.basename(file):
            transformation = "Peritumoral"
        else:
            self.segprocessor.error.warning(
                "Could not find the mask transformation for {}.".format(os.path.basename(file)))

        return transformation

    def get_all_perturbed_segs(self):
        """
        Screen the pertibations out_folder and return a list of all perturbed segmentations and the number
        :parameter
        """

        all_perturbed_segs = []
        all_num = 0
        out_pert_seg = self.segprocessor.out_path + "/perturbed_seg"

        # if the folder does not exist there are no perturbed segmentations
        if not os.path.isdir(out_pert_seg):
            os.mkdir(out_pert_seg)
            return all_perturbed_segs, all_num

        all_perturbed = glob.glob(out_pert_seg + "/*.nii.gz")
        for pert_seg in all_perturbed:
            if "_closed" in os.path.basename(pert_seg):
                all_perturbed_segs.append(pert_seg)
            elif "_convex" in os.path.basename(pert_seg):
                all_perturbed_segs.append(pert_seg)
            elif pert_seg.endswith("_expanded.nii.gz"):
                all_perturbed_segs.append(pert_seg)
            elif "random_walker" in os.path.basename(pert_seg):
                all_perturbed_segs.append(pert_seg)
            elif "_morph_" in os.path.basename(pert_seg):
                all_perturbed_segs.append(pert_seg)

        all_num = len(all_perturbed_segs)

        return all_perturbed_segs, all_num

    def get_accepted_perturbed_segs(self):
        """
        Returns a list of accepted perturbed segmentations
        :return:
        """

        accepted_perturbed_segs = []
        accepted_num = 0
        out_acc_seg = self.segprocessor.out_path + "/accepted_perturbed_seg"

        # if the folder does not exist there are no accepted segmentations
        if not os.path.isdir(out_acc_seg):
            os.mkdir(out_acc_seg)
            return accepted_perturbed_segs, accepted_num

        accepted_perturbed_segs = glob.glob(out_acc_seg + "/*.nii.gz")
        accepted_num = len(accepted_perturbed_segs)


        return accepted_perturbed_segs, accepted_num

    def get_not_accepted_perturbed_segs(self):
        """
        Returns a list of not accepted perturbed segmentations
        :return:
        """

        not_accepted_perturbed_segs = []
        not_accepted_num = 0
        out_not_acc_seg = self.segprocessor.out_path + "/not_accepted_perturbed_seg"

        # if the folder does not exist there are no accepted segmentations
        if not os.path.isdir(out_not_acc_seg):
            os.mkdir(out_not_acc_seg)
            return not_accepted_perturbed_segs, not_accepted_num

        not_accepted_perturbed_segs = glob.glob(out_not_acc_seg + "/*_dice_*.nii.gz")
        not_accepted_num = len(not_accepted_perturbed_segs)

        return not_accepted_perturbed_segs, not_accepted_num

    def performed_transformations(self, list_of_out_files, file_endings=None):
        """
        Function to filter already performed perturbations for supervoxel and random walker
        :param list_of_out_files: list of already performed perturbations
        :param file_endings: file endings of the perturbations to identify the segmentation
        :return:
        """

        # default file endings
        if file_endings is None:
            file_endings = [".nii.gz"]

        performing = self.segprocessor.df.loc[self.segprocessor.df["Mask_Transformation"].isnull(), :].copy()
        performing = performing.loc[performing["Image_Transformation"].isnull(), :]

        skipping = 0
        skipped_files = []

        # Filtering for already performed analysis
        if len(list_of_out_files) >= 0:
            for pert in list_of_out_files:
                for file_ending in file_endings:
                    # search for common ID in file pattern
                    pert_file_pattern = os.path.basename(pert)[:-(len(file_ending))]
                    for seg in self.segprocessor.df["Mask"].values:
                        mask_file_pattern = os.path.basename(seg)[:-(len(".nii.gz"))]
                        if mask_file_pattern in pert_file_pattern:
                            # if segmentation is still in input files
                            if seg in performing["Mask"].values:
                                # delete the processed files from the input
                                for id_ in set(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == seg, "ID"]):
                                    performing = performing[performing["ID"] != id_]
                                    skipping += 1
                                    skipped_files.append(seg)
                                # self.logger.info("Skipping " + str(os.path.basename(seg)))
                            break
        else:
            self.segprocessor.logger.info("No already performed perturbations found!")

        # filtering for surrounding segmentations
        for seg in performing["Mask"].values.tolist():
            if "peritumoral" in os.path.basename(seg):
                performing = performing[performing["Mask"] != seg]

        if skipping > 0:
            self.segprocessor.logger.info("Skipped " + str(skipping) + " already performed perturbations!")
            print("Skipped " + str(skipping) + " already performed perturbations!")

        return performing

    def calc_dice4control(self, perturbed_seg: str, file_ending: str):  # calculated_dice_pert: list, seg_path: str, file_ending: str):
        """
        Calculated Dice and copies file if dice is above self.dice_threshold to accepted dice folder if not to non accepted dice folder
        :param perturbed_seg: path to perturbed segmentation
        :param file_ending: ending of the perturbed files of interest to get the original segmentation
        """
        #:param calculated_dice_pert: list of perturbed segmentations which have a calculates dice
        #:param seg_path: path to segmentation
        #

        # counter for accepted and not accepted segmentations
        accept = 0
        not_accept = 0

        # get the original segmentation path
        original_seg_path = ""

        # get perturbed and non perturbed segmentation array
        seg_pert = ""
        seg_orig = ""

        # get file path of the true segmentation
        pert_seg_file_pattern = os.path.basename(perturbed_seg)[:-len(file_ending)]
        msks = self.segprocessor.df.loc[self.segprocessor.df["Mask_Transformation"].isnull() & self.segprocessor.df[
            "Image_Transformation"].isnull()].copy()
        file_pattern = pd.DataFrame(
            {"File_pattern": [os.path.basename(f)[:-len(file_ending)] for f in msks["Mask"].values]},
            index=msks.index.copy())
        msks["File_pattern"] = file_pattern["File_pattern"]

        # get the original segmentation path with the file pattern
        for i, row in msks.iterrows():
            if row["File_pattern"] in pert_seg_file_pattern:
                original_seg_path = row["Mask"]
                break

        # check if the perturbed segmentation is readable
        if os.path.isfile(perturbed_seg):
            try:
                seg_pert = sitk.ReadImage(perturbed_seg)
            except:
                self.segprocessor.error.warning(
                    "Segmentation perturbation {} corrupted! Deleting ...".format(os.path.basename(perturbed_seg)))
                os.remove(perturbed_seg)
        else:
            self.segprocessor.error.warning("Perturbation {} does not exist!".format(os.path.basename(perturbed_seg)))

        if original_seg_path != "":
            # check if the original segmentation is readable
            if os.path.isfile(original_seg_path):
                try:
                    seg_orig = sitk.ReadImage(original_seg_path)
                except:
                    self.segprocessor.error.error(
                        "Original segmentation {} not readable!".format(os.path.basename(original_seg_path)))
                    raise ValueError(
                        "Original segmentation {} not readable!".format(os.path.basename(original_seg_path)))
            else:
                self.segprocessor.error.warning(
                    "Original segmentation {} does not exist!".format(os.path.basename(original_seg_path)))
        else:
            self.segprocessor.error.warning(
                "Could not find original segmentation for sample {}!".format(pert_seg_file_pattern))

        if seg_pert != "" and seg_orig != "":

            # get output pth to accepted and non accepted perturbations
            out_acc_seg = self.segprocessor.out_path + "/accepted_perturbed_seg"
            out_not_acc_seg = self.segprocessor.out_path + "/not_accepted_perturbed_seg"

            y_true = sitk.GetArrayFromImage(seg_orig)
            y_pred = sitk.GetArrayFromImage(seg_pert)

            # calculte dice between original and perturbed segmentation
            ddice = self.dice_coeff(y_true, y_pred)
            # print("Dice", ddice, "Threshold", self.dice_threshold)

            # it should not be perfect but it should be a good estimation of the ground truth (interrater)
            if (round(ddice, 2) >= self.segprocessor.dice_threshold) and (round(ddice, 2) < 1.0):
                new_file_name = os.path.basename(perturbed_seg)[:-(len(".nii.gz"))] + "_dice_" + str(
                    round(ddice, 2)) + ".nii.gz"
                shutil.copy2(perturbed_seg, out_acc_seg + "/" + new_file_name)
                accept += 1
            else:
                new_file_name = os.path.basename(perturbed_seg)[:-(len(".nii.gz"))] + "_dice_" + str(
                    round(ddice, 2)) + ".nii.gz"
                shutil.copy2(perturbed_seg, out_not_acc_seg + "/" + new_file_name)
                not_accept += 1

        return accept, not_accept

    @staticmethod
    def dice_coeff(y_true, y_pred):
        """
        Computes Soerensen-dice coefficient.

        compute the Soerensen-dice coefficient between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred`.

        Args:
         y_true: 3-dim Numpy array of type bool. The ground truth mask.
         y_pred: 3-dim Numpy array of type bool. The predicted mask.

        Returns:
         the dice coefficient as float. If both masks are empty, the result is NaN.
        """

        area_sum = np.sum(y_pred == 1) + np.sum(y_true == 1)

        if area_sum > 0:
            return np.sum(y_pred[y_true == 1]) * 2.0 / area_sum
        else:
            return 0.0

    def segmentation_closing(self, seg_path):
        """
        Closing of the segmentation
        :param seg_path: Path to the segmentation
        :return: corrected segmentation
        """

        radius = self.segprocessor.seg_closing_radius

        seg_ = sitk.ReadImage(seg_path)
        nda = sitk.GetArrayFromImage(seg_)

        closed_nda = np.ndarray(nda.shape)
        for i in range(nda.shape[0]):
            # bnda = skimage.morphology.isotropic_closing(image=nda[i], radius=radius)
            bnda = skimage.morphology.binary_closing(image=nda[i])
            closed_nda[i] = bnda

        sitk_img_out = sitk.GetImageFromArray(closed_nda.astype(np.uint16))
        sitk_img_out.CopyInformation(seg_)
        sitk.WriteImage(sitk_img_out,
                        seg_path[:-len(".nii.gz")] + "_closed.nii.gz",
                        useCompression=True)

        return closed_nda

    def segmentation_convex_hull(self, seg_path):
        """
        Correction of the segmentation by convex hull
        :param seg_path: Path to the segmentation
        :return: corrected segmentation
        """
        # self.logger.info("Segmentation convex hull correction of " + str(os.path.basename(seg_path)))

        seg_ = sitk.ReadImage(seg_path)
        nda = sitk.GetArrayFromImage(seg_)

        closed_nda = np.ndarray(nda.shape)
        for i in range(nda.shape[0]):
            bnda = skimage.morphology.convex_hull_object(nda[i])
            closed_nda[i] = bnda

        sitk_img_out = sitk.GetImageFromArray(closed_nda.astype(np.uint16))
        sitk_img_out.CopyInformation(seg_)
        sitk.WriteImage(sitk_img_out,
                        seg_path[:-len(".nii.gz")] + "_convex.nii.gz",
                        useCompression=True)

        return closed_nda

    def performed_supervoxel_calc(self, list_of_out_files, file_endings=None):
        """
        Function to filter already performed perturbations for supervoxel and random walker
        :param list_of_out_files: list of already performed perturbations
        :param file_endings: file endings of the perturbations to identify the segmentation
        :return:
        """

        # default file endings
        if file_endings is None:
            file_endings = [".nii.gz"]

        # get samples which are not having any mask transformation
        performing = self.segprocessor.df.loc[self.segprocessor.df["Mask_Transformation"].isnull(), :].copy()
        performing = performing.loc[performing["Image_Transformation"].isnull(), :]
        skipping = 0
        skipped_files = []
        print("Found " + str(len(performing)) + " Samples to process!")

        # Filtering for already performed analysis
        if len(list_of_out_files) > 0:
            for pert in list_of_out_files:
                for file_ending in file_endings:
                    # search for common ID in file pattern
                    pert_file_pattern = os.path.basename(pert)[:-(len(file_ending))]
                    for seg in set(self.segprocessor.df.loc[
                                       self.segprocessor.df["Mask_Transformation"].isnull(), "Mask"].to_list()):
                        mask_file_pattern = os.path.basename(seg)[:-(len(".nii.gz"))]
                        if mask_file_pattern in pert_file_pattern:
                            # if segmentation is still in input files
                            if seg in set(performing.loc[:, "Mask"].to_list()):
                                # delete the processed files from the input
                                for id_ in set(self.segprocessor.df.loc[self.segprocessor.df["Mask"] == seg, "ID"]):
                                    performing = performing[performing["ID"] != id_]
                                    skipping += 1
                                    skipped_files.append(seg)
                                # self.logger.info("Skipping " + str(os.path.basename(seg)))
                            break
        else:
            self.segprocessor.logger.info("No already performed perturbations found!")

        # filtering for surrounding segmentations
        for seg in performing["Mask"].values.tolist():
            if "peritumoral" in os.path.basename(seg):
                performing = performing[performing["Mask"] != seg]

        if skipping > 0:
            self.segprocessor.logger.info("Skipped " + str(skipping) + " already performed perturbations!")
            print("Skipped " + str(skipping) + " already performed perturbations!")

        return performing

    def performed_perturbations(self, list_of_out_files: list, list_of_input_files: list, file_ending=".nii.gz"):

        performing = list_of_input_files
        skipping = 0

        # Filtering for already performed analysis
        if len(list_of_out_files) >= 0:
            for out in list_of_out_files:
                out_file_pattern = os.path.basename(out)[:-(len(file_ending))]
                for input_f in list_of_input_files:
                    input_file_pattern = os.path.basename(input_f)[:-(len(".nii.gz"))]
                    if input_file_pattern in out_file_pattern:
                        if input_f in performing:
                            performing.remove(input_f)
                            skipping += 1
                            # self.logger.info("Skipping " + str(os.path.basename(input_f)))
                        break

        if skipping > 0:
            self.segprocessor.logger.info("Skipped " + str(skipping) + " already performed calculations!")
            print("Skipped " + str(skipping) + " already performed calculations!")
        return performing

    def process_segmentation_perturbation(self):
        """
        Execute segmentation perturbation methods
        """

        self.segprocessor.logger.info("### Start Segmentation Perturbation ###")

        # 1. expand segmentations
        if "connected_component" in self.segprocessor.perturbation_method:
            self.process_connected_component_expansion()

        # 2. surround segmentations - > handled extra
        #if self.segprocessor.peritumoral_seg:
        #    self.process_surround_segmentation()

        # 3. SuperVoxel based perturbation
        if "supervoxel" in self.segprocessor.perturbation_method:
            self.process_supervoxel_perturbation()

        # 4. Random walker based perturbation
        if "random_walker" in self.segprocessor.perturbation_method:
            self.process_random_walker_perturbation()

        # remove images from segmentation folder
        img_in_perturbated_seg_folder = glob.glob(
            self.segprocessor.out_path + "/perturbed_seg/*_" + self.segprocessor.modality + ".nii.gz")

        if len(img_in_perturbated_seg_folder) > 0:
            for file in tqdm.tqdm(img_in_perturbated_seg_folder, desc="Cleaning perturbated segmentations ..."):
                os.remove(file)

        # 5. Segmentation Postprocessing
        self.process_segmentation_postprocessing()


class Image_Transformation_processing(SegProcessor):
    """
    Class for image transformation processing
    """

    def __init__(self, segprocessor):
        # super().__init__()
        self.segprocessor = segprocessor

    def add_transformed_images_to_output(self, img: str, ext_df: pd.DataFrame):
        """
        Add transformed images to output csv
        :param ext_df: transformed img path
        :param img: path to transformed image
        :return: pd.DataFrame, updated output csv
        """

        trans_img_pattern = os.path.basename(img)[:-len(".nii.gz")]
        kernel = super().find_kernel_in_pattern(img_path=img, kernels_in_files=self.segprocessor.kernels_in_files)
        added = 0

        if kernel is not None:

            id_df = self.segprocessor.df.loc[self.segprocessor.df["Image_Transformation"].isna()].copy()
            for i, r in id_df.iterrows():
                file_pattern = os.path.basename(r["Image"])[:-len(".nii.gz")]

                if file_pattern in trans_img_pattern:
                    entry = pd.DataFrame({"ID": [r["ID"]],
                                          "Image": [os.path.abspath(img)],
                                          "Mask": [r["Mask"]],
                                          "Modality": [r["Modality"]],
                                          "ROI_Label": [r["ROI_Label"]],
                                          "Image_Transformation": [kernel],
                                          "Mask_Transformation": [r["Mask_Transformation"]],
                                          "Timepoint": [r["Timepoint"]],
                                          "Rater": [r["Rater"]],
                                          "Prediction_Label": [r["Prediction_Label"]]
                                          })





                    added += 1
                    ext_df = pd.concat([ext_df, entry], ignore_index=True)
                    # Add all mask transformations from this image to the DataFrame
                    for mi, mr in id_df.loc[~id_df["Mask_Transformation"].isna()].iterrows():
                        # Do not extract Peritumoral process from transformed images
                        if mr["Mask_Transformation"] != "Peritumoral":
                            if mr["Image"] == r["Image"]:
                                # if r["ID"] in mr["Mask"]:
                                entry = pd.DataFrame({"ID": [mr["ID"]],
                                                      "Image": [os.path.abspath(img)],
                                                      "Mask": [mr["Mask"]],
                                                      "Modality": [mr["Modality"]],
                                                      "ROI_Label": [mr["ROI_Label"]],
                                                      "Image_Transformation": [kernel],
                                                      "Mask_Transformation": [mr["Mask_Transformation"]],
                                                      "Timepoint": [mr["Timepoint"]],
                                                      "Rater": [mr["Rater"]],
                                                      "Prediction_Label": [mr["Prediction_Label"]]
                                                      })

                        ext_df = pd.concat([ext_df, entry], ignore_index=True)
                    break
        else:
            self.segprocessor.error.warning("No kernel found for transformed image " + os.path.basename(img))
            return None, None

        # free memory
        del id_df

        return ext_df, added

    def get_transformer(self, trans_to_process:dict):
        """
        Get Transformation object
        :param trans_to_process (dict): tranformation kernel: [original_img_paths]
        :return:
        """

        transformer = Executor(rptk_config_json=self.segprocessor.rptk_config_json,
                               input_csv=self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv",
                               kernels=self.segprocessor.kernels,
                               output_dir=self.segprocessor.out_path + "/transformed_images/",
                               n_cpu=self.segprocessor.n_cpu,
                               modality=self.segprocessor.df["Modality"].values[0],
                               logger=self.segprocessor.logger,
                               error=self.segprocessor.error,
                               to_process=trans_to_process,
                               use_previous_output=self.segprocessor.use_previous_output,
                               RunID=self.segprocessor.RunID)
        return transformer

    def get_missing_kernel_images(self, kernel_categories):
        """
        Identifies missing kernel categories for each ID and maps them to image paths where they are missing.

        Parameters:
            kernel_categories (list): List of required kernel categories (partial matches allowed).

        Returns:
            dict: A dictionary where keys are missing kernel categories and values are lists of corresponding image paths.
        """
        
        df = pd.read_csv(self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv", index_col=0)

        if "ID" != df.index.name:
            if "ID" not in df.columns:
                self.segprocessor.error.error("ID column not found in the input CSV file!")
                raise ValueError("ID column not found in the input CSV file!")
            else:
                df = df.set_index("ID")

        df = df.drop_duplicates(subset=['Image', 'Mask'])

        tmp = df.copy().loc[df["Image_Transformation"].isnull()]
        raw_samples_count = len(tmp.loc[tmp["Mask_Transformation"].isnull()])
        raw_samples = tmp.loc[tmp["Mask_Transformation"].isnull()]

        transformed_samples_count = {}
        for i in df['Image_Transformation'].value_counts().keys():
            transformed_samples_count[i] = df['Image_Transformation'].value_counts()[i]
            
        transformed_samples = {}
        for i, r in df.copy().loc[~df["Image_Transformation"].isnull()].iterrows():
            if r['Image_Transformation'] in transformed_samples:
                transformed_samples[r['Image_Transformation']].append(i)
            else:
                transformed_samples[r['Image_Transformation']] = [i]

        found_real_kernel = ""
        need_to_process_transformed_samples = {}
        for kernel in transformed_samples:
            for i, sample in raw_samples.iterrows():
                if i not in transformed_samples[kernel]:
                    found_real_kernel = ""
                    for real_kernel in kernel_categories:
                        if kernel.startswith(real_kernel):
                            found_real_kernel = real_kernel
                            break
                    if found_real_kernel != "":
                        if found_real_kernel not in need_to_process_transformed_samples:
                            need_to_process_transformed_samples[found_real_kernel] = [sample["Image"]]
                        else:
                            if sample["Image"] not in need_to_process_transformed_samples[found_real_kernel]:
                                need_to_process_transformed_samples[found_real_kernel].append(sample["Image"])
                    else:
                        print(f"Could not find any matching kernel for image transformations {kernel}!")

        return need_to_process_transformed_samples
    
    def extract_kernel_info(self, filename: str, kernel_dict: dict) -> str:
        """
        Extract the kernel name and configuration from the filename based on predefined patterns.
        
        Parameters:
        filename (str): Name of the transformed image file.
        kernel_dict (dict): Dictionary mapping kernel names to their patterns.
        
        Returns:
        str: Formatted kernel name with configuration if present, otherwise just the kernel name.
        """
        for kernel, pattern in kernel_dict.items():
            if pattern in filename:
                # Extract configuration after the pattern (if present)
                match = re.search(pattern + r"([\w\d_.-]+)?(?=\.nii\.gz)", filename)
                config = match.group(1) if match and match.group(1) else ""
                return f"{kernel}{config}".strip()

        return "Unknown"

    def process_transformed_image(self, args: tuple) -> list:
        """
        Process a transformed image and check if it exists with all allowed mask transformations.
        
        Parameters:
        args (tuple): Contains transformed image filename, transformed images folder path, and the dataframe.
        
        Returns:
        list: New rows to be added to the dataframe.
        """
        transformed_image, transformed_images_folder, df = args
        image_id = transformed_image.split("_")[0]  # Extract ID from filename
        
        # Find all existing mask transformations for this ID (excluding Peritumoral masks)
        existing_transformations = df[(df['ID'] == image_id) & (~df['Mask_Transformation'].str.contains("Peritumoral", na=False))]
        
        new_rows = []
        for _, row in existing_transformations.iterrows():
            transformed_exists = ((df['ID'] == image_id) & 
                                (df['Image'] == os.path.join(transformed_images_folder, transformed_image)) & 
                                (df['Mask_Transformation'] == row['Mask_Transformation'])).any()
            
            if not transformed_exists:
                new_row = row.copy()
                new_row['Image'] = os.path.join(transformed_images_folder, transformed_image)
                new_row['Image_Transformation'] = self.extract_kernel_info(transformed_image, self.segprocessor.kernels_in_files_dict)
                new_rows.append(new_row)

        return new_rows

    def update_csv_with_transformed_images(self, csv_file: str, transformed_images_folder: str) -> None:
        """
        Update the CSV file by adding transformed images and their corresponding mask transformations.
        
        Parameters:
        csv_file (str): Path to the existing CSV file.
        transformed_images_folder (str): Path to the folder containing transformed images.
        
        Returns:
        None
        """
        # Load existing CSV file
        df = pd.read_csv(csv_file)
        
        # List all transformed images in the folder
        transformed_images = [f for f in os.listdir(transformed_images_folder) if f.endswith(".nii.gz")]
        
        # Use multiprocessing to speed up processing
        pool = Pool(self.segprocessor.n_cpu)
        results = []
        
        with tqdm.tqdm(total=len(transformed_images), desc="Transformed Image Quality Check") as pbar:
            for result in pool.imap_unordered(self.process_transformed_image, [(img, transformed_images_folder, df) for img in transformed_images]):
                results.extend(result)
                pbar.update(1)
        
        pool.close()
        pool.join()
        
        # Convert results to DataFrame and concatenate
        new_df = pd.DataFrame(results)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Remove duplicate rows based on Mask and Image columns
        df = df.drop_duplicates(subset=["Mask", "Image"], keep="first")
        
        # Save the updated CSV
        # df.to_csv("updated_data.csv", index=False)
        return df


    # 5. Image Transformations
    def process_image_transformation(self):
        """
        Execute Image Transformation
        """
        self.segprocessor.logger.info("#### Start Image Transformation ####")
        print("#### Starting Image Transformation ...")

        transformer = self.get_transformer(trans_to_process=self.segprocessor.trans_to_process)
        transformer.run()

        df = pd.read_csv(self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv", index_col=0)

        if "ID" != df.index.name:
            if "ID" in df.columns:
                df.set_index("ID", inplace=True)
            else:
                self.segprocessor.error.error("No ID column in the DataFrame!")
                raise ValueError("No ID column in the DataFrame!")

        df = df.drop_duplicates(subset=['Image', 'Mask'])

        tmp = df.copy().loc[df["Image_Transformation"].isnull()]
        raw_samples_count = len(tmp.loc[tmp["Mask_Transformation"].isnull()])
        raw_samples = tmp.loc[tmp["Mask_Transformation"].isnull()]

        transformed_samples_count = {}
        for i in df['Image_Transformation'].value_counts().keys():
            transformed_samples_count[i] = df['Image_Transformation'].value_counts()[i]
            
        transformed_samples = {}
        for i, r in df.copy().loc[~df["Image_Transformation"].isnull()].iterrows():
            if r['Image_Transformation'] in transformed_samples:
                transformed_samples[r['Image_Transformation']].append(i)
            else:
                transformed_samples[r['Image_Transformation']] = [i]

        need_to_process_transformed_samples_count = {}
        for kernel in transformed_samples_count:
            if transformed_samples_count[kernel] != raw_samples_count:
                print(f"Need to transform {raw_samples_count-transformed_samples_count[kernel]} samples with kernel {kernel}.")
                need_to_process_transformed_samples_count[kernel] = raw_samples_count-transformed_samples_count[kernel]  

        # double check if all images are transformed
        trans_to_process_dict = self.get_missing_kernel_images(self.segprocessor.transform_exe.kernels_in_files.keys())
        
        if len(trans_to_process_dict.values()) > 0:
            transformer = self.get_transformer(trans_to_process=trans_to_process_dict)
            transformer.run()

        # remove garbage transformations from MIRP
        delete_transformed_imgs = glob.glob(self.segprocessor.out_path + "/transformed_images/*_int.nii.gz")
        delete_transformed_imgs += glob.glob(self.segprocessor.out_path + "/transformed_images/*_morph.nii.gz")
        delete_transformed_imgs += glob.glob(self.segprocessor.out_path + "/transformed_images/*_.nii.gz")

        if len(delete_transformed_imgs) > 0:
            for img in tqdm.tqdm(delete_transformed_imgs, desc="Clean the transformed images"):
                os.remove(img)
            # free memory
            del delete_transformed_imgs

        # screening transformed images for the right ID and add them to the csv
        transformed_imgs = glob.glob(self.segprocessor.out_path + "/transformed_images/*.nii.gz")

        ext_df = self.segprocessor.df.copy()
        t_df = ext_df.loc[~(ext_df["Image_Transformation"].isna()) & ext_df["Mask_Transformation"].isna(), :]
        t_df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')
        t_img_in_csv = [os.path.basename(file_path) for file_path in list(set(t_df["Image"].values))]

        need_to_add = []
        for transformed_img_path in transformed_imgs:
            transformed_img = os.path.basename(transformed_img_path)
            if transformed_img not in t_img_in_csv:
                need_to_add.append(transformed_img_path)

        added = 0
        results = pd.DataFrame()

        if len(need_to_add) > 0:
            # Process files using multiple CPUs
            with Pool(processes=self.segprocessor.n_cpu) as pool:
                # Use tqdm with imap to show progress
                partial_function = partial(self.add_transformed_images_to_output, ext_df=self.segprocessor.df.copy())
                for result in tqdm.tqdm(pool.imap(partial_function, need_to_add),
                                        total=len(need_to_add),
                                        desc="Add transformed Images to output csv"):

                    if result[0] is not None and result[1] is not None:
                        added += result[1]
                        results = pd.concat([results, result[0]], ignore_index=True)

            self.segprocessor.df = results.copy()
            self.segprocessor.df.drop_duplicates(subset=['Image', 'Mask'], inplace=True, keep='first')
            self.segprocessor.df.to_csv(
                self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv",
                index=False)
            del result

        results = self.update_csv_with_transformed_images(self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv", 
                                                self.segprocessor.out_path + "/transformed_images/")

        self.segprocessor.df = results.copy()
        self.segprocessor.df.to_csv(self.segprocessor.out_path + "/" + self.segprocessor.RunID + "_preprocessing_out.csv", index=False)

        self.segprocessor.logger.info("#### Image Transformation Done!")
        print("Image Transformation Done!")

    def get_image_transformation_from_file(self, file: str):
        """
        :param file: name of file to get transformation from
        :return transform_config: transformation with config
        """

        transform_config = ""

        for transformation in self.segprocessor.kernels_in_files_dict:
            if transformation in os.path.basename(file):
                try:
                    config = re.search(transformation + "([-0-9A-Za-z-_.]*).nii.gz", os.path.basename(file)).group(1)
                    transform_config = self.segprocessor.kernels_in_files_dict[transformation] + config
                except AttributeError:
                    self.segprocessor.error.warning("Could not find {} Transformation configuration of {}".format(transformation, os.path.basename(file)))
                    transform_config = self.segprocessor.kernels_in_files_dict[transformation]

        return transform_config
