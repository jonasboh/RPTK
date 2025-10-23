import os
import pytest

import tqdm

import SimpleITK as sitk

import numpy as np
import pandas as pd

from typing import Union, List

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass,\
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass,\
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.mirp.imageRead import load_image
from rptk.mirp import imageClass
from rptk.mirp import *

# from radiomics import imageoperations
#from mirp_pipeline.Pipeline import *
#from mirp_pipeline.Transformation_config import *
#from mirp_pipeline.Experiment_config import *

import glob
import logging

class Experiment_Generator:

    def __init__(self,
                 MIRP_transformations: list,
                 modality: str,
                 Feature_extraction_settings: FeatureExtractionSettingsClass,
                 write_path: str,
                 df: pd.DataFrame,
                 feature_calc_conf: list = ["raw_peri", "default_intra"],
                 logger: logging.Logger = logging.getLogger("Mirp Pipeline"),
                 log_file_name: str = "mirp_extraction.log"): # raw_intra, raw_peri, default_intra, default_peri, predict_intra, predict_peri, reseg (only intra)

        """
        Attributes for Experiment_Generator:
        MIRP_transformations: list of MIRP transformations
        modality: modality of the images
        Feature_extraction_settings: Feature extraction settings
        write_path: path to write the results
        df: dataframe containing the image paths
        feature_calc_conf: list of feature calculation configurations
        logger: logger
        log_file_name: name of the log file
        """


        self.logger = logger

        # create file handler which logs info messages
        fh = logging.FileHandler(log_file_name, 'a', 'utf-8')
        fh.setLevel(logging.INFO)

        # create console handler with a debug log level
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.ERROR)

        # feature calculation settings: raw_intra, raw_peri, default_intra, default_peri, predict_intra, predict_peri, reseg (only intra)
        self.feature_calc_conf = feature_calc_conf

        accepted_feature_calc_conf = ["raw_intra", "raw_peri", "default_intra", "default_peri", "predict_intra",
                                      "predict_peri", "reseg"]

        # create folder for each feature_calc_conf if not existent
        for conf in feature_calc_conf:
            if conf not in accepted_feature_calc_conf:
                raise ValueError("Feature calculation configuration not accepted. Please choose from: " + str(
                    accepted_feature_calc_conf))
            else:
                if not os.path.exists(os.path.join(write_path, conf)):
                    os.makedirs(os.path.join(write_path, conf))

        # creating a formatter
        formatter = logging.Formatter('%(name)s - %(levelname)-8s: %(message)s')

        # setting handler format
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        # logger.addHandler(ch)

        self.logger.info('Setting up MIRP experiments.')
        self.modality = modality
        self.MIRP_transformations = MIRP_transformations
        self.Feature_extraction_settings = Feature_extraction_settings
        self.write_path = write_path
        self.df = df



    def generate_experiments(self):

        """ Create a list of experiments for all image transformations """

        # Check if results are already in the writing path:
        for conf in self.feature_calc_conf:
            if self.write_path[-1] != "/":
                self.write_path = self.write_path + "/"

            finished_samples = glob.glob(self.write_path + conf + "/*.csv")


        finished_basenames = []

        for sample in finished_samples:
            finished_basenames.append(os.path.basename(sample))

        self.logger.info("Detected files in out folder: " + str(len(finished_basenames)))

        self.finished_basenames = finished_basenames
        self.finished_samples = finished_samples

        experiments = self.experiment_setting_config()

        return experiments


    def experiment_setting_config(self):

        """ Configure experiments for all image transformations """

        experiments = []

        # Several image transformations should be applied
        if len(self.MIRP_transformations) != 0:

            filter_ = Transformation_config(MIRP_transformations=self.MIRP_transformations, Feature_extraction_settings=self.Feature_extraction_settings)

            mirp_trans_settings = filter_.get_mirp_transformation_settings()

            # create experiments for all image transformations
            for index, row in mirp_trans_settings.iterrows():
                tmp_experiments = self.create_exp_list(image_transformation_settings=row["Settings"])
                experiments.extend(tmp_experiments)

        # Default: Extraction on the raw Image: No Image transformation should be applied
        transformation = ImageTransformationSettingsClass(by_slice=False,
                                                          response_map_feature_settings=self.Feature_extraction_settings)

        tmp_experiments = self.create_exp_list(image_transformation_settings=transformation)
        experiments.extend(tmp_experiments)

        return experiments

    def create_exp_list(self,
                        image_transformation_settings: ImageTransformationSettingsClass):
        """ Create a list of experiments for a given image transformation setting """
        # self.logger.info("Create existing file list... ")
        experiments = []
        num = 0
        # get different configurations for resegmentation and feature extraction
        for conf in self.feature_calc_conf:
            # Get Settings for raw, default and predicted experiments
            settings_intra, settings_peri = Experiment_config(experiment_config_string=conf,
                                                              Feature_extraction_settings=self.Feature_extraction_settings,
                                                              modality=self.modality,
                                                              image_transformation_settings=image_transformation_settings).exp_config()
            num += 1
            for idx, row_ in self.df.iterrows():
                
                # Get Settings for intratumoral or peritumoral experiments
                if "intra" in conf:
                    # Intratumoral - processed sample?
                    image_id = os.path.basename(row_["Image"])[:-len(".nii.gz")]
                    seg_id = os.path.basename(row_["Mask_intra"])[:-len(".nii.gz")]
                    sample_id = image_id + "_" + seg_id
                    
                    completed = self.check_existing_exp(spl=sample_id,
                                                         setting=settings_intra,
                                                         finished_filenames=self.finished_basenames)

                    # Intratumoral - not processed sample
                    if (sample_id + "_" + str(settings_intra.general.config_str) in completed) == False:
                        experiment = set_exp_settings(sample_id, row_, settings_intra, self.modality, self.write_path + conf + "/")
                        experiments.append(experiment)

                elif "peri" in conf:
                    # Peritumoral - processed sample?
                    image_id = os.path.basename(row_["Image"])[:-len(".nii.gz")]
                    seg_id = os.path.basename(row_["Mask_peri"])[:-len(".nii.gz")]
                    sample_id = image_id + "_" + seg_id
                    
                    completed = self.check_existing_exp(spl=sample_id,
                                                         setting=settings_peri,
                                                         finished_filenames=self.finished_basenames)

                    # Peritumoral - not processed sample
                    if (sample_id + "_" + str(settings_peri.general.config_str) in completed) == False:
                        experiment = set_exp_settings(sample_id, row_, settings_peri, self.modality, self.write_path + conf + "/")
                        experiments.append(experiment)
                else:
                    sample_id = os.path.basename(row_["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row_["Mask_intra"])[:-len(".nii.gz")]

        return experiments


    def check_existing_exp(self, spl: str,
                           setting: SettingsClass,
                           finished_filenames: list):

        """ Check if the sample has already been processed and if the results are already there. """

        completed = [] # Check if the sample have been process with this configuration
        for finished_sample in finished_filenames:
            if finished_sample.startswith(spl): # Sample already processed
                if finished_sample.endswith(str(setting.general.config_str) + "_features.csv"):
                    self.logger.info("Skipping:" + spl + " " + str(setting.general.config_str))
                    completed.append(spl + "_" + str(setting.general.config_str))

        return completed

def set_exp_settings(Sample_ID, row_, settings, modality, write_path):

    """ Set the experiment settings for the given sample. """

    if "predict_intra" in settings.general.config_str:
        experiment = ExperimentClass(
                    modality=modality,
                    subject=Sample_ID,
                    cohort=None,
                    write_path=write_path,
                    image_folder=os.path.dirname(row_["Image"]),
                    roi_folder=os.path.dirname(row_["Mask_intra"]),
                    roi_reg_img_folder=None,
                    image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                    registration_image_file_name_pattern=None,
                    roi_names=[os.path.basename(row_["Mask_intra"])[:-len(".nii.gz")]],
                    data_str=None,
                    provide_diagnostics=True,
                    settings=settings,
                    compute_features=True,
                    extract_images=False,
                    plot_images=False,
                    keep_images_in_memory=False,
                    # img_obj =None,
                    # roi_list=None
                )

    elif "intra" in settings.general.config_str:
        experiment = ExperimentClass(
                    modality=modality,
                    subject=Sample_ID,
                    cohort=None,
                    write_path=write_path,
                    image_folder=os.path.dirname(row_["Image"]),
                    roi_folder=os.path.dirname(row_["Mask_intra"]),
                    roi_reg_img_folder=None,
                    image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                    registration_image_file_name_pattern=None,
                    roi_names=[os.path.basename(row_["Mask_intra"])[:-len(".nii.gz")]],
                    data_str=None,
                    provide_diagnostics=True,
                    settings=settings,
                    compute_features=True,
                    extract_images=False,
                    plot_images=False,
                    keep_images_in_memory=False,
                    # img_obj =None,
                    # roi_list=None
                )

    elif "predict_peri" in settings.general.config_str:
        experiment = ExperimentClass(
                    modality=modality,
                    subject=Sample_ID,
                    cohort=None,
                    write_path=write_path,
                    image_folder=os.path.dirname(row_["Image"]),
                    roi_folder=os.path.dirname(row_["Mask_peri"]),
                    roi_reg_img_folder=None,
                    image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                    registration_image_file_name_pattern=None,
                    roi_names=[os.path.basename(row_["Mask_peri"])[:-len(".nii.gz")]],
                    data_str=None,
                    provide_diagnostics=False,
                    settings=settings,
                    compute_features=True,
                    extract_images=False,
                    plot_images=False,
                    keep_images_in_memory=False,
                    # img_obj =None,
                    # roi_list=None
                )

    elif "peri" in settings.general.config_str:
        experiment = ExperimentClass(
                    modality=modality,
                    subject=Sample_ID,
                    cohort=None,
                    write_path=write_path,
                    image_folder=os.path.dirname(row_["Image"]),
                    roi_folder=os.path.dirname(row_["Mask_peri"]),
                    roi_reg_img_folder=None,
                    image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                    registration_image_file_name_pattern=None,
                    roi_names=[os.path.basename(row_["Mask_peri"])[:-len(".nii.gz")]],
                    data_str=None,
                    provide_diagnostics=False,
                    settings=settings,
                    compute_features=True,
                    extract_images=False,
                    plot_images=False,
                    keep_images_in_memory=False,
                    # img_obj =None,
                    # roi_list=None
                )

    elif "reseg" in settings.general.config_str:
        experiment = ExperimentClass(
                    modality=modality,
                    subject=Sample_ID,
                    cohort=None,
                    write_path=write_path,
                    image_folder=os.path.dirname(row_["Image"]),
                    roi_folder=os.path.dirname(row_["Mask_peri"]),
                    roi_reg_img_folder=None,
                    image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                    registration_image_file_name_pattern=None,
                    roi_names=[os.path.basename(row_["Mask_peri"])[:-len(".nii.gz")]],
                    data_str=None,
                    provide_diagnostics=False,
                    settings=settings,
                    compute_features=False,
                    extract_images=False,
                    plot_images=False,
                    keep_images_in_memory=False,
                    # img_obj =None,
                    # roi_list=None
                )

    else:
        raise ValueError("Unknown experiment config string! "
                         "\nPlease choose one of the following: "
                         "\nperi, intra, predict_peri, predict_intra, reseg, raw_intra, raw_peri, default_intra, default_peri")

    return experiment
