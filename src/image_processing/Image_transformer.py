from __future__ import print_function

import os
import psutil
from threading import Semaphore
import logging
import json

import numpy as np
import pandas as pd
from typing import Union, List

from rptk.src.config.Transformation_config import *
from rptk.src.segmentation_processing.Segmentation_perturbator import *

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass

import radiomics
from radiomics import *


class Pyradiomics_image_transformer:
    """
    Class to extract transformed images from Pyradiomics for processing with different feature extraction tools.
    :parameter: kernels: Union[List, DataFrame, Series, str] = None (List of kernels to be used for image transformation.)
    :parameter: yaml_file: str = "" (Path to yaml file.)
    :parameter: input_csv: str = "" (Path to input csv file.)
    :parameter: output_dir: str = "" (Path to output directory.)
    """

    def __init__(self,
                 rptk_config_json: str = None,  # path to config file for kernels
                 kernels: Union[List, str] = None,  # kernels to be used for image transformation --> run from list
                 yaml_file: str = None,  # path to yaml file  --> run from yaml file
                 img_path: str = "",  # path to image
                 seg_path: str = "",  # path to segmentation
                 output_dir: str = "",  # output directory
                 tidy_output: bool = False,  # If True, the output will be in separated folders and csv files with all the
                 logger=logging.getLogger("Pyradiomics Image Transformer"),
                 LoG_sigma=None,
                 Gradient_gradientUseSpacing=True,
                 LBP2D_lbp2DRadius=None,
                 LBP2D_lbp2DSamples=None,
                 LBP2D_lbp2DMethod=None,
                 LBP3D_lbp3DLevels=None,
                 LBP3D_lbp3DIcosphereRadius=None,
                 LBP3D_lbp3DIcosphereSubdivision=None,
                 ):

        # default settings
        if LBP3D_lbp3DIcosphereSubdivision is None:
            LBP3D_lbp3DIcosphereSubdivision = [1]

        if LBP3D_lbp3DIcosphereRadius is None:
            LBP3D_lbp3DIcosphereRadius = [1]

        if LBP3D_lbp3DLevels is None:
            LBP3D_lbp3DLevels = [2]

        if LBP2D_lbp2DMethod is None:
            LBP2D_lbp2DMethod = ["uniform"]

        if LBP2D_lbp2DSamples is None:
            LBP2D_lbp2DSamples = [9]

        if LBP2D_lbp2DRadius is None:
            LBP2D_lbp2DRadius = [1]

        if LoG_sigma is None:
            LoG_sigma = [1]

        radiomics.setVerbosity(40)
        logger = radiomics.logger
        logger.setLevel(logging.ERROR)

        self.logger = logger
        self.kernels = kernels
        self.yaml_file = yaml_file
        self.img_path = img_path
        self.seg_path = seg_path
        self.output_dir = output_dir
        self.tidy_output = tidy_output
        self.rptk_config_json = rptk_config_json

        self.LoG_sigma = LoG_sigma
        self.Gradient_gradientUseSpacing = Gradient_gradientUseSpacing
        self.LBP2D_lbp2DRadius = LBP2D_lbp2DRadius
        self.LBP2D_lbp2DSamples = LBP2D_lbp2DSamples
        self.LBP2D_lbp2DMethod = LBP2D_lbp2DMethod
        self.LBP3D_lbp3DLevels = LBP3D_lbp3DLevels
        self.LBP3D_lbp3DIcosphereRadius = LBP3D_lbp3DIcosphereRadius
        self.LBP3D_lbp3DIcosphereSubdivision = LBP3D_lbp3DIcosphereSubdivision


        # read config file
        if self.rptk_config_json != None:
            # read conifg json file
            with open(self.rptk_config_json, 'r') as f:
                self.config = json.load(f)

            self.LoG_sigma = self.config["Preprocessing_config"]["transformation_kernel_config"]["LoG"]["LoG_sigma"]
            self.Gradient_gradientUseSpacing = self.config["Preprocessing_config"]["transformation_kernel_config"]["Gradient"]["Gradient_gradientUseSpacing"]
            self.LBP2D_lbp2DRadius = self.config["Preprocessing_config"]["transformation_kernel_config"]["LBP2D"]["LBP2D_lbp2DRadius"]
            self.LBP2D_lbp2DSamples = self.config["Preprocessing_config"]["transformation_kernel_config"]["LBP2D"]["LBP2D_lbp2DSamples"]
            self.LBP2D_lbp2DMethod = self.config["Preprocessing_config"]["transformation_kernel_config"]["LBP2D"]["LBP2D_lbp2DMethod"]
            self.LBP3D_lbp3DLevels = self.config["Preprocessing_config"]["transformation_kernel_config"]["LBP3D"]["LBP3D_lbp3DLevels"]
            self.LBP3D_lbp3DIcosphereRadius = self.config["Preprocessing_config"]["transformation_kernel_config"]["LBP3D"]["LBP3D_lbp3DIcosphereRadius"]
            self.LBP3D_lbp3DIcosphereSubdivisionLevel = self.config["Preprocessing_config"]["transformation_kernel_config"]["LBP3D"]["LBP3D_lbp3DIcosphereSubdivisionLevel"]


        self.roi_names = [os.path.basename(self.seg_path)[:-len(".nii.gz")]]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # self.df = pd.read_csv(self.input_csv)

        self.options_of_filters = ["Wavelet",
                                   "Square",
                                   "LoG",
                                   "SquareRoot",
                                   "Logarithm",
                                   "Exponential",
                                   "Gradient",
                                   "LBP2D",
                                   "LBP3D"]

        self.Wavelet_img_list = []
        self.Square_img_list = []
        self.LoG_img_list = []
        self.SquareRoot_img_list = []
        self.Logarithm_img_list = []
        self.Exponential_img_list = []
        self.Gradient_img_list = []
        self.LBP2D_img_list = []
        self.LBP3D_img_list = []

        self.transformed_img_paths_csv = pd.DataFrame(columns=["Image", "Mask"])

        self.transformed_img_paths = []
        self.t_img_paths_list = []

    def extract_2_list(self, generator):
        list_ = []

        for img in generator:
            list_.append(img)

        return list_

    def save_transformed_img(self, out_folder_name, img_list):
        """
        write image to out_folder
        """
        
        file_name = os.path.basename(self.img_path)
        out_files = []

        if self.tidy_output:
            if not os.path.exists(self.output_dir + "/" + out_folder_name):
                os.makedirs(self.output_dir + "/" + out_folder_name)

        # Save files to an appropriate location
        for img in img_list:
            if self.tidy_output:
                src_filepath = os.path.join(self.output_dir, out_folder_name, file_name[:-len(".nii.gz")] + "_" + img[1] + ".nii.gz")
                if not os.path.exists(src_filepath):
                    sitk.WriteImage(img[0], src_filepath, useCompression=True)
                out_files.append(src_filepath)
            else:
                src_filepath = os.path.join(self.output_dir, file_name[:-len(".nii.gz")] + "_" + img[1] + ".nii.gz")
                if not os.path.exists(src_filepath):
                    sitk.WriteImage(img[0], src_filepath, useCompression=True)
                out_files.append(src_filepath)
                
        # free memory
        del img_list
        
        return out_files

    def run_from_yaml_file(self):
        # reading Pyradiomics yaml settings file and extract settings for filter application
        with open(self.yaml_file) as file:
            documents = yaml.full_load(file)

            image = sitk.ReadImage(self.img_path)
            label = sitk.ReadImage(self.seg_path)
            self.transformed_img_paths_csv = self.transformed_img_paths_csv.append(pd.DataFrame([self.img_path],
                                                                                                columns=['Image']),
                                                                                   ignore_index=True)

            self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(self.seg_path)

            for item, doc in documents.items():

                for kernel in doc:
                    if kernel in self.options_of_filters:

                        if kernel == "LoG":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                log_generator = imageoperations.getLoGImage(image,
                                                                            label,
                                                                            sigma=[1])
                            else:
                                # only sigma is possible
                                log_generator = imageoperations.getLoGImage(image,
                                                                            label,
                                                                            sigma=doc[i]["sigma"])

                            LoG_img_list = self.extract_2_list(log_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "LoG",
                                LoG_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del LoG_img_list

                        if kernel == "Wavelet":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings: LLH, LHL, LHH, HLL, HLH, HHL, HHH, LLL
                                Wavelet_generator = imageoperations.getWaveletImage(image,
                                                                                    label)

                            Wavelet_img_list = self.extract_2_listextract_2_list(Wavelet_generator)
                            t_img_paths_list = self.save_transformed_img(
                                                                        "Wavelet",
                                                                        Wavelet_img_list)
                            del Wavelet_img_list
                            del Wavelet_generator
                            
                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            
                            del t_img_paths_list

                        if kernel == "Square":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                Square_generator = imageoperations.getSquareImage(image,
                                                                                  label)

                            Square_img_list = self.extract_2_list(Square_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "Square",
                                Square_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del Square_img_list
                            del t_img_paths_list

                        if kernel == "SquareRoot":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                SquareRoot_generator = imageoperations.getSquareRootImage(image,
                                                                                          label)

                            SquareRoot_img_list = self.extract_2_list(SquareRoot_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "SquareRoot",
                                SquareRoot_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del SquareRoot_img_list
                            del t_img_paths_list

                        if kernel == "Logarithm":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                Logarithm_generator = imageoperations.getLogarithmImage(image,
                                                                                        label)

                            Logarithm_img_list = self.extract_2_list(Logarithm_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "Logarithm",
                                Logarithm_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del Logarithm_img_list
                            del t_img_paths_list

                        if kernel == "Exponential":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                Exponential_generator = imageoperations.getExponentialImage(image,
                                                                                            label)

                            Exponential_img_list = self.extract_2_list(Exponential_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "Exponential",
                                Exponential_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del Exponential_img_list
                            del t_img_paths_list

                        if kernel == "Gradient":
                            # no further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                Gradient_generator = imageoperations.getGradientImage(image,
                                                                                      label)

                            Gradient_img_list = self.extract_2_list(Gradient_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "Gradient",
                                Gradient_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del Gradient_img_list

                        if kernel == "LBP2D":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                LBP2D_generator = imageoperations.getLBP2DImage(image,
                                                                                label)

                            LBP2D_img_list = self.extract_2_list(LBP2D_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "LBP2D",
                                LBP2D_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del LBP2D_img_list
                            del t_img_paths_list

                        if kernel == "LBP3D":
                            # No further parameter set
                            if len(doc[kernel]) == 0:
                                # default settings
                                LBP3D_generator = imageoperations.getLBP3DImage(image,
                                                                                label)

                            LBP3D_img_list = self.extract_2_list(LBP3D_generator)
                            t_img_paths_list = self.save_transformed_img(
                                "LBP3D",
                                LBP3D_img_list)

                            for t_img in t_img_paths_list:
                                self.transformed_img_paths.append(t_img)

                            # free memory
                            del LBP3D_img_list
                            del t_img_paths_list

            self.transformed_img_paths_csv = pd.concat([self.transformed_img_paths_csv,
                                                        pd.DataFrame(self.transformed_img_paths,
                                                                     columns=['Image'])],
                                                       ignore_index=True)

            self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(self.seg_path)

    def run_from_list(self):

        # TODO add for all filters to get the transformed image and save it from the input list
        #print("self.img_path", self.img_path)
        #print("self.seg_path", self.seg_path)
        image = sitk.ReadImage(self.img_path)
        label = sitk.ReadImage(self.seg_path)

        self.transformed_img_paths_csv = pd.concat([self.transformed_img_paths_csv,
                                                    pd.DataFrame([self.img_path],
                                                                 columns=['Image'])],
                                                   ignore_index=True)

        self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(self.seg_path)

        for kernel in self.kernels:
            if kernel in self.options_of_filters:

                if kernel == "LoG":
                    # Default configuration
                    log_generator = imageoperations.getLoGImage(image,
                                                                label,
                                                                sigma=self.LoG_sigma)

                    LoG_img_list = self.extract_2_list(log_generator)
                    t_img_paths_list = self.save_transformed_img("LoG", LoG_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del LoG_img_list
                    del t_img_paths_list

                if kernel == "Wavelet":
                    Wavelet_generator = imageoperations.getWaveletImage(image, label)

                    Wavelet_img_list = self.extract_2_list(Wavelet_generator)
                    
                    t_img_paths_list = self.save_transformed_img(
                        "Wavelet",
                        Wavelet_img_list)
                    
                    del Wavelet_img_list
                    del Wavelet_generator
                    
                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    
                    del t_img_paths_list

                if kernel == "Square":
                    Square_generator = imageoperations.getSquareImage(image, label)

                    Square_img_list = self.extract_2_list(Square_generator)

                    t_img_paths_list = self.save_transformed_img(
                        "Square",
                        Square_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del Square_img_list
                    del t_img_paths_list

                if kernel == "SquareRoot":
                    SquareRoot_generator = imageoperations.getSquareRootImage(image, label)

                    SquareRoot_img_list = self.extract_2_list(SquareRoot_generator)

                    t_img_paths_list = self.save_transformed_img(
                        "SquareRoot",
                        SquareRoot_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del SquareRoot_img_list
                    del t_img_paths_list

                if kernel == "Logarithm":
                    Logarithm_generator = imageoperations.getLogarithmImage(image, label)

                    Logarithm_img_list = self.extract_2_list(Logarithm_generator)
                    t_img_paths_list = self.save_transformed_img(
                        "Logarithm",
                        Logarithm_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del Logarithm_img_list
                    del t_img_paths_list

                if kernel == "Exponential":
                    Exponential_generator = imageoperations.getExponentialImage(image, label)

                    Exponential_img_list = self.extract_2_list(Exponential_generator)

                    t_img_paths_list = self.save_transformed_img(
                        "Exponential",
                        Exponential_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del Exponential_img_list
                    del t_img_paths_list

                if kernel == "Gradient":
                    Gradient_generator = imageoperations.getGradientImage(image, label,
                                                                          gradientUseSpacing=self.Gradient_gradientUseSpacing)

                    Gradient_img_list = self.extract_2_list(Gradient_generator)

                    t_img_paths_list = self.save_transformed_img(
                        "Gradient",
                        Gradient_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del Gradient_img_list
                    del t_img_paths_list

                if kernel == "LBP2D":
                    LBP2D_generator = imageoperations.getLBP2DImage(image, label,
                                                                    lbp2DRadius=self.LBP2D_lbp2DRadius,
                                                                    lbp2DMethod=self.LBP2D_lbp2DMethod,
                                                                    lbp2DSamples=self.LBP2D_lbp2DSamples,)

                    LBP2D_img_list = self.extract_2_list(LBP2D_generator)
                    t_img_paths_list = self.save_transformed_img(
                        "LBP2D",
                        LBP2D_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del LBP2D_img_list
                    del t_img_paths_list

                if kernel == "LBP3D":
                    LBP3D_generator = imageoperations.getLBP3DImage(image, label,
                                                                    lbp3DLevels=self.LBP3D_lbp3DLevels,
                                                                    lbp3DIcosphereRadius=self.LBP3D_lbp3DIcosphereRadius,
                                                                    lbp3DIcosphereSubdivisionLevel=self.LBP3D_lbp3DIcosphereSubdivisionLevel,)

                    LBP3D_img_list = self.extract_2_list(LBP3D_generator)
                    t_img_paths_list = self.save_transformed_img(
                        "LBP3D",
                        LBP3D_img_list)

                    for t_img in t_img_paths_list:
                        self.transformed_img_paths.append(t_img)

                    # free memory
                    del LBP3D_img_list
                    del t_img_paths_list

                self.transformed_img_paths_csv = pd.concat([self.transformed_img_paths_csv,
                                                            pd.DataFrame(self.transformed_img_paths,
                                                                         columns=['Image'])],
                                                           ignore_index=True)

                self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(self.seg_path)

            else:
                self.logger.error("Kernel not available!")
                break

        # remove for better memory efficientcy
        del image
        del label

    def process(self):

        if self.yaml_file != None:
            self.run_from_yaml_file()

        elif self.kernels != None:
            self.run_from_list()
        else:
            self.logger.error("No configuration file or kernels provided!")

        if self.tidy_output:
            if not os.path.isfile(self.output_dir + '/pyradiomics_transformed_img.csv'):
                self.transformed_img_paths_csv.to_csv(self.output_dir + '/pyradiomics_transformed_img.csv')
            else:
                self.transformed_img_paths_csv.to_csv(self.output_dir + '/pyradiomics_transformed_img.csv', mode='a',
                                                      header=False)


class MIRP_image_and_seg_pertubator:
    """ Class for image transformation and segmentation variation from MIRP.
    Requirements:
    :input_format Image naming <SampleID>_<modality>_<timepoint>_<setting>.nii.gz.
    :input_format Segmentation naming <SampleID>_<modality>_<timepoint>_<rator>.nii.gz.

    Attributes for Image_transformer class
    :parameter img_path: Path to the image.
    :parameter kernels: List of transformations to apply to the image.
    :parameter modality: Modality of the image.
    :parameter output_dir: Output directory for the transformed images.
    :parameter distance: Distance in mm to grow or shrink the ROI.
    :parameter crop_around_roi: Crop the image around the ROI.
    :parameter perturbation_roi_adapt_type: Type of perturbation to apply to the ROI.
    :parameter repetition: Number of repetitions for the perturbation.
    :parameter img_path: Path to the image.
    :parameter seg_path: Path to the segmentation.

    :return transformed_image: Transformed image with replaced postfix string (<setting>) in the image name.
    :return transformed_seg: Transformed segmentation with replaced postfix string (<setting>) in the image name.
    """

    def __init__(self,
                 img_path: str,
                 seg_path: str,
                 modality: str,
                 rptk_config_json: str = None,  # path to config file for kernels
                 kernels: str = None,
                 seg_pertubation: bool = False,
                 output_dir: str = None,
                 repetition: int = 3,
                 distance: Union[None, List[float], float] = 0.0,
                 crop_around_roi: bool = False,
                 perturbation_roi_adapt_type: str = "distance",
                 logger=logging.getLogger("MIRP transformation and Segmentation pertubation")
                 ):

        self.rptk_config_json = rptk_config_json
        self.img_path = img_path
        self.seg_path = seg_path
        self.kernels = kernels
        self.modality = modality
        self.seg_pertubation = seg_pertubation
        self.output_dir = output_dir
        self.repetition = repetition
        self.distance = distance
        self.crop_around_roi = crop_around_roi
        self.perturbation_roi_adapt_type = perturbation_roi_adapt_type
        self.logger = logger

        # generate needed variables
        if self.output_dir is None:
            self.output_dir = os.path.dirname(self.img_path)

        # take the last part of the filename (<setting>) as the new string for the transformed image
        # self.setting = os.path.basename(self.img_path).split("_")[-1]

        # take the timepoint of the image
        # self.timepoint = os.path.basename(self.img_path).split("_")[-2]

        # file name configuration: <SampleID>_<modality>_<timepoint>_<setting>.nii.gz
        # self.sample_id = os.path.basename(self.img_path).split("_")[0]

        self.options_of_filters = ["laws",
                                   "gabor",
                                   "gaussian",
                                   "nonseparable_wavelet",
                                   "separable_wavelet",
                                   "mean",
                                   "laplacian_of_gaussian"]

        # self.img_name = self.timepoint + "_" + self.transformation_kernel + ".nii.gz"
        # self.data_string = os.path.basename(self.img_path).replace(
        #    self.replace, self.transformation_kernel)[:-len(".nii.gz")]

    def perturbation(self):
        """ Generate perturbation for segmentation growth/shrinkage. """

        perturbation_settings = ImagePerturbationSettingsClass(
            crop_around_roi=self.crop_around_roi,
            perturbation_roi_adapt_type=self.perturbation_roi_adapt_type,
            perturbation_roi_adapt_size=self.distance,
            perturbation_randomise_roi_repetitions=self.repetition
        )
        return perturbation_settings

    def create_settings(self):
        """
        Set default settings for generating response maps and computing feature values.
        """

        if self.kernels == None:
            self.logger.info("No transformation applied!")
            if self.seg_pertubation:
                self.logger.info("Only segmentation pertubation applied!")

            general_settings = GeneralSettingsClass(
                by_slice=False
            )

            image_interpolation_settings = ImageInterpolationSettingsClass(
                by_slice=False,
                interpolate=False,
                anti_aliasing=False
            )

            perturbation_settings = self.perturbation()

            feature_computation_parameters = FeatureExtractionSettingsClass(
                by_slice=False,
                no_approximation=True,
                base_feature_families="none",
            )

            image_transformation_settings = ImageTransformationSettingsClass(
                by_slice=False,
                response_map_feature_families="statistics",
                response_map_feature_settings=None,
                boundary_condition="reflect",
                filter_kernels=None
            )
            settings = SettingsClass(
                general_settings=general_settings,
                post_process_settings=ImagePostProcessingClass(),
                img_interpolate_settings=image_interpolation_settings,
                roi_interpolate_settings=RoiInterpolationSettingsClass(),
                roi_resegment_settings=ResegmentationSettingsClass(),
                perturbation_settings=perturbation_settings,
                img_transform_settings=image_transformation_settings,
                feature_extr_settings=feature_computation_parameters
            )

        else:
            general_settings = GeneralSettingsClass(by_slice=False)

            image_interpolation_settings = ImageInterpolationSettingsClass(by_slice=False)

            if self.modality == "CT":
                resegmentation_intensity_range = [-1000.0, 1000.0]
                resegmentation_settings = ResegmentationSettingsClass(
                    resegmentation_method="threshold",
                    resegmentation_intensity_range=resegmentation_intensity_range
                )
            else:
                resegmentation_settings = ResegmentationSettingsClass()

            feature_computation_settings = FeatureExtractionSettingsClass(
                by_slice=False,
                no_approximation=False,
                base_feature_families="statistics")

            if self.kernels in self.options_of_filters:
                image_trans_config = Transformation_config(
                    rptk_config_json=self.rptk_config_json,
                    Feature_extraction_settings=feature_computation_settings,
                    MIRP_transformations=[self.kernels])
            else:
                self.logger.error("Kernel {} not available!".format(self.kernels))
                image_trans_config = Transformation_config(rptk_config_json=self.rptk_config_json,)
                # print(image_trans_config.get_mirp_transformation_settings())

            image_transformation_settings = image_trans_config.get_mirp_transformation_settings()["Settings"].iloc[0]

            if self.seg_pertubation:
                image_perturbation_settings = self.perturbation()
            else:
                image_perturbation_settings = ImagePerturbationSettingsClass()

            settings = SettingsClass(
                general_settings=general_settings,
                post_process_settings=ImagePostProcessingClass(),
                img_interpolate_settings=image_interpolation_settings,
                roi_interpolate_settings=RoiInterpolationSettingsClass(),
                roi_resegment_settings=resegmentation_settings,
                perturbation_settings=image_perturbation_settings,
                img_transform_settings=image_transformation_settings,
                feature_extr_settings=feature_computation_settings
            )

        return settings

    def generate_experiments(self):
        """
        Generate experiment object.
        """

        settings = self.create_settings()

        experiment = ExperimentClass(
            modality=self.modality,
            subject=os.path.basename(self.img_path)[:-len(".nii.gz")],  # self.sample_id,
            cohort=None,
            write_path=self.output_dir,
            image_folder=os.path.dirname(self.img_path),
            roi_folder=os.path.dirname(self.seg_path),
            roi_reg_img_folder=None,
            image_file_name_pattern=os.path.basename(self.img_path)[:-len(".nii.gz")],
            registration_image_file_name_pattern=None,
            roi_names=[os.path.basename(self.seg_path)[:-len(".nii.gz")]],
            data_str=[""],
            provide_diagnostics=False,
            settings=settings,
            compute_features=False,
            extract_images=True,
            plot_images=False,
            keep_images_in_memory=False
        )

        return experiment
