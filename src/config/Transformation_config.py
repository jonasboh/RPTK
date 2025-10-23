import os

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
from radiomics import imageoperations

import glob


class Transformation_config:
    """
    Configuration for MIRP image transformation.
    :param rptk_config_json: Path to configuration file. Get configuration for transformation kernels.
    """

    def __init__(self,
                 rptk_config_json:str = None,
                 MIRP_transformations: List[str] = None,
                 Feature_extraction_settings: importSettings.FeatureExtractionSettingsClass = None,
                 response_map_discretisation_method: str = "fixed_bin_number",  # fixed_bin_size for bin with
                 response_map_discretisation_n_bins: int = 16,
                 response_map_discretisation_bin_width: int =25,
                 response_map_feature_settings: importSettings.FeatureExtractionSettingsClass = None,
                 response_map_feature_families: List[str] = None,
                 gabor_sigma: float = 3.0,
                 gabor_lambda: float = 0.9,
                 laws_kernel: List[str] = ["L5S5E5", "E5E5E5"],
                 laws_delta: int = 7,
                 laws_compute_energy: bool = True,
                 laws_rotation_invariance: bool = True,
                 laws_pooling_method: str = "max",
                 laplacian_of_gaussian_sigma: List[float] = [2.0],
                 gaussian_sigma: List[float] = [2.0],
                 nonseparable_wavelet_families: List[str] = ["shannon", "simoncelli"],
                 nonseparable_wavelet_responses: str = "real",
                 separable_wavelet_families: List[str] = ["haar"],
                 separable_wavelet_response: str = "real",
                 mean_filter_kernel_size: List[int] = [15]):

        self.rptk_config_json = rptk_config_json
        self.MIRP_transformations = MIRP_transformations
        self.Feature_extraction_settings = Feature_extraction_settings
        self.response_map_discretisation_method = response_map_discretisation_method
        self.response_map_discretisation_n_bins = response_map_discretisation_n_bins
        self.response_map_discretisation_bin_width = response_map_discretisation_bin_width
        self.response_map_feature_settings = response_map_feature_settings
        self.response_map_feature_families = response_map_feature_families
        self.gabor_sigma = gabor_sigma
        self.gabor_lambda = gabor_lambda
        self.laws_kernel = laws_kernel
        self.laws_delta = laws_delta
        self.laws_compute_energy = laws_compute_energy
        self.laws_rotation_invariance = laws_rotation_invariance
        self.laws_pooling_method = laws_pooling_method
        self.laplacian_of_gaussian_sigma = laplacian_of_gaussian_sigma
        self.gaussian_sigma = gaussian_sigma
        self.nonseparable_wavelet_families = nonseparable_wavelet_families
        self.nonseparable_wavelet_responses = nonseparable_wavelet_responses
        self.separable_wavelet_families = separable_wavelet_families
        self.separable_wavelet_response = separable_wavelet_response
        self.mean_filter_kernel_size = mean_filter_kernel_size

        if self.rptk_config_json != None:
            # read conifg json file
            with open(self.rptk_config_json, 'r') as f:
                self.config = json.load(f)

            self.response_map_discretisation_method = self.config["Preprocessing_config"]["transformation_kernel_config"]["response_map_discretisation_method"]
            self.response_map_discretisation_n_bins = self.config["Preprocessing_config"]["transformation_kernel_config"]["response_map_discretisation_n_bins"]
            self.response_map_discretisation_bin_width = self.config["Preprocessing_config"]["transformation_kernel_config"]["response_map_discretisation_bin_width"]
            self.gabor_sigma = self.config["Preprocessing_config"]["transformation_kernel_config"]["gabor"]["gabor_sigma"]
            self.gabor_lambda = self.config["Preprocessing_config"]["transformation_kernel_config"]["gabor"]["gabor_lambda"]
            self.laws_kernel = self.config["Preprocessing_config"]["transformation_kernel_config"]["laws"]["laws_kernel"]
            self.laws_delta = self.config["Preprocessing_config"]["transformation_kernel_config"]["laws"]["laws_delta"]
            self.laws_compute_energy = self.config["Preprocessing_config"]["transformation_kernel_config"]["laws"]["laws_compute_energy"]
            self.laws_rotation_invariance = self.config["Preprocessing_config"]["transformation_kernel_config"]["laws"]["laws_rotation_invariance"]
            self.laws_pooling_method = self.config["Preprocessing_config"]["transformation_kernel_config"]["laws"]["laws_pooling_method"]
            self.laplacian_of_gaussian_sigma = self.config["Preprocessing_config"]["transformation_kernel_config"]["laplacian_of_gaussian"]["laplacian_of_gaussian_sigma"]
            self.gaussian_sigma = self.config["Preprocessing_config"]["transformation_kernel_config"]["gaussian"]["gaussian_sigma"]
            self.nonseparable_wavelet_families = self.config["Preprocessing_config"]["transformation_kernel_config"]["nonseparable_wavelet"]["nonseparable_wavelet_families"]
            self.nonseparable_wavelet_responses = self.config["Preprocessing_config"]["transformation_kernel_config"]["nonseparable_wavelet"]["nonseparable_wavelet_responses"]
            self.mean_filter_kernel_size = self.config["Preprocessing_config"]["transformation_kernel_config"]["mean_filter"]["mean_filter_kernel_size"]

    def get_mirp_transformation_settings(self):
        MIRP_transformation_settings = pd.DataFrame(columns=["Settings"], index=self.MIRP_transformations)
        
        # if no kernel is given
        if (self.MIRP_transformations != None) and (self.MIRP_transformations != [None]) :
            for i in self.MIRP_transformations:

                #Image_transformation_setting = ImageTransformationSettingsClass()

                # Garbor Filter
                if (i == "gabor") or (i == "garbor"):
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width = self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        filter_kernels="gabor",
                        gabor_sigma=self.gabor_sigma,
                        gabor_lambda=self.gabor_lambda,
                        # gabor_gamma = 1.0, # default
                        # gabor_theta = 0.0, # default
                        # gabor_theta_step = None, # default
                        # gabor_response = "modulus", # default
                        # gabor_rotation_invariance = False, # default
                        # gabor_pooling_method = "max", # default
                        # gabor_boundary_condition = ,
                    )

                # Laws Filter
                if i == "laws":
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width=self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        laws_kernel=self.laws_kernel,
                        laws_delta=self.laws_delta,  # default
                        laws_compute_energy=self.laws_compute_energy,  # default
                        laws_rotation_invariance=self.laws_rotation_invariance,  # default
                        laws_pooling_method=self.laws_pooling_method,  # default
                        filter_kernels="laws",
                    )

                # LoG Filter
                if i == "laplacian_of_gaussian":
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width = self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        laplacian_of_gaussian_sigma=self.laplacian_of_gaussian_sigma,
                        filter_kernels="laplacian_of_gaussian",
                    )

                # Gaussian Filter
                if i == "gaussian":
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width = self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        gaussian_sigma=self.gaussian_sigma,
                        filter_kernels="gaussian",
                    )

                # nonseparable_wavelet Filter
                if i == "nonseparable_wavelet":
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width = self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        nonseparable_wavelet_families=self.nonseparable_wavelet_families,
                        # nonseparable_wavelet_decomposition_level = 1, # default?
                        nonseparable_wavelet_response=self.nonseparable_wavelet_responses,  # default
                        # nonseparable_wavelet_boundary_condition = ,
                        filter_kernels="nonseparable_wavelet",
                    )

                # separable_wavelet Filter
                if i == "separable_wavelet":
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width = self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        separable_wavelet_families=self.separable_wavelet_families,
                        # separable_wavelet_decomposition_level = 1, # default?
                        separable_wavelet_response=self.separable_wavelet_response,  # default
                        # separable_wavelet_boundary_condition = ,
                        # separable_wavelet_families=["shannon","simoncelli"],
                        filter_kernels="separable_wavelet",
                    )

                # mean Filter
                if i == "mean":
                    Image_transformation_setting = ImageTransformationSettingsClass(
                        by_slice=False,  # use 3D calculations
                        response_map_discretisation_method=self.response_map_discretisation_method,
                        response_map_discretisation_bin_width = self.response_map_discretisation_bin_width,
                        response_map_discretisation_n_bins=self.response_map_discretisation_n_bins,
                        response_map_feature_settings=self.Feature_extraction_settings,
                        response_map_feature_families=self.response_map_feature_families,
                        filter_kernels="mean",
                        mean_filter_kernel_size=self.mean_filter_kernel_size,
                        # mean_filter_boundary_condition="constant"
                    )

                MIRP_transformation_settings.loc[i] = Image_transformation_setting
        else:
            Image_transformation_setting = ImageTransformationSettingsClass(by_slice=False,
                                                 response_map_feature_settings=None
                                                )
                
            MIRP_transformation_settings.loc[1] = Image_transformation_setting
                
        return MIRP_transformation_settings
