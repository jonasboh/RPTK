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

import glob
import logging

# TODO generate configurtation for MIRP and Pyradiomics for each Modality to comapre them --> also include meaningful transfomred images (Pyradiomics Wavelet, (LBP), LoG, Garbor, laws --> furthermore make segmentation changes on the mask

class Experiment_config:

    def __init__(self,
                 experiment_config_string:str,
                 Feature_extraction_settings: importSettings.FeatureExtractionSettingsClass,
                 modality: str,
                 image_transformation_settings: importSettings.ImageTransformationSettingsClass):

        self.Feature_extraction_settings = Feature_extraction_settings
        self.modality = modality
        self.experiment_config_string = experiment_config_string
        self.image_transformation_settings = image_transformation_settings
        self.logger = logging.getLogger("mirp_pipeline")

    """ 
    Attributes for Experiment_config class
    experiment_config_string: str (String to identify the experiment configuration.)
    Feature_extraction_settings: importSettings.FeatureExtractionSettingsClass (Settings for feature extraction.)
    modality: str (Modality of the image.)
    image_transformation_settings: importSettings.ImageTransformationSettingsClass (Settings for image transformation.)
    """

    def exp_config(self):

        # get different configurations for resegmentation and feature extraction
        if ("raw" in self.experiment_config_string) or ("all" in self.experiment_config_string):

            raw = PipelineSetting(conf_str="raw",
                                  modality=self.modality,
                                  image_transformation_settings=self.image_transformation_settings,
                                  feature_computation_settings=self.Feature_extraction_settings)

            settings_raw_intra, settings_raw_peri = raw.process()

            # Configure Settings to identify transformation
            settings_raw_intra = self.set_iD_for_setting(settings=settings_raw_intra)
            settings_raw_peri = self.set_iD_for_setting(settings=settings_raw_peri)

            return settings_raw_intra, settings_raw_peri

        elif ("default" in self.experiment_config_string) or ("all" in self.experiment_config_string):

            default = PipelineSetting(conf_str="default",
                                      modality=self.modality,
                                      image_transformation_settings=self.image_transformation_settings,
                                      feature_computation_settings=self.Feature_extraction_settings)

            settings_default_intra, settings_default_peri = default.process()

            settings_default_intra = self.set_iD_for_setting(settings=settings_default_intra)
            settings_default_peri = self.set_iD_for_setting(settings=settings_default_peri)

            return settings_default_intra, settings_default_peri

        elif ("predict" in self.experiment_config_string) or ("all" in self.experiment_config_string):

            predict = PipelineSetting(conf_str="predict",
                                      modality=self.modality,
                                      image_transformation_settings=self.image_transformation_settings,
                                      feature_computation_settings=self.Feature_extraction_settings)

            settings_intra, settings_peri = predict.process()

            settings_intra = self.set_iD_for_setting(settings=settings_intra)
            settings_peri = self.set_iD_for_setting(settings=settings_peri)

            return settings_intra, settings_peri

        else:
            raise self.logger.debug("No valid experiment configuration string")


    def set_iD_for_setting(self, settings: SettingsClass):
        ''' Change the config string accordingly with the transformation filter string '''

        if settings.img_transform.spatial_filters != None:
            if len(settings.img_transform.spatial_filters) != 0:
                filter_ = ""
                if len(settings.img_transform.spatial_filters) > 1:
                    for _filter_ in settings.img_transform.spatial_filters:
                        filter_ = filter_ + "_" + _filter_

                    ID = settings.general.config_str
                    new_ID = ID + filter_
                    settings.general.config_str = new_ID

                elif(len(settings.img_transform.spatial_filters) == 1):
                    new_ID = settings.general.config_str + "_" + settings.img_transform.spatial_filters[0]
                    settings.general.config_str = new_ID

        return settings
