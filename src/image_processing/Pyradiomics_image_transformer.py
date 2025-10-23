from __future__ import print_function

import os  # needed navigate the system to get the input data
from typing import Any, Union

import SimpleITK as sitk
import pandas as pd
import yaml
import argparse
from typing import List
import numpy as np

from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

import radiomics
from radiomics import *


class Pyradiomics_image_transformer:
    def __init__(self,
                 kernels: Union[List, DataFrame, Series, np.ndarray] = None,
                 yaml_file: str = "",
                 input_csv: str = "",
                 output_dir: str = "",
                 ):
        
        radiomics.setVerbosity(40)
        logger = radiomics.logger
        logger.setLevel(logging.ERROR)

        self.logger = logging.getLogger("Pyradiomics_image_transformer")
        self.kernels = kernels

        self.yaml_file = yaml_file
        self.input_csv = input_csv
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.df = pd.read_csv(self.input_csv)

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


    def save_transformed_img(self, out_folder_name, img_list, input_file_path):
        file_name = os.path.basename(input_file_path)
        out_files = []

        if not os.path.exists(self.output_dir + "/" + out_folder_name):
            os.makedirs(self.output_dir + "/" + out_folder_name)

        # Save files to appropriate location
        for img in img_list:
            src_filepath = os.path.join(self.output_dir, out_folder_name, file_name[:-7] + "_" + img[1] + ".nii.gz")
            if not os.path.exists(src_filepath):
                sitk.WriteImage(img[0], src_filepath, useCompression=True)
            out_files.append(src_filepath)
        return out_files


    def run_from_yaml_file(self):
        # reading Pyradiomics yaml settings file and extract settings for filter application
        with open(self.yaml_file) as file:
            documents = yaml.full_load(file)

            for img_path, seg_path in zip(self.df.Image, self.df.Mask):
                image = sitk.ReadImage(img_path)
                label = sitk.ReadImage(seg_path)
                self.transformed_img_paths_csv = self.transformed_img_paths_csv.append(pd.DataFrame([img_path], columns=['Image']),
                                                                             ignore_index=True)
                self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(seg_path)

                for item, doc in documents.items():

                    for i in doc:
                        if i in options_of_filters:

                            if i == "LoG":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    log_generator = imageoperations.getLoGImage(image,
                                                                                label,
                                                                                sigma=[1])
                                else:
                                    # only sigma is possible
                                    log_generator = imageoperations.getLoGImage(image,
                                                                                label,
                                                                                sigma=doc[i]["sigma"])

                                LoG_img_list = Pyradiomics_image_transformer.extract_2_list(log_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "LoG",
                                                     LoG_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                            if i == "Wavelet":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings: LLH, LHL, LHH, HLL, HLH, HHL, HHH, LLL
                                    Wavelet_generator = imageoperations.getWaveletImage(image,
                                                                                        label)
                                Wavelet_img_list = Pyradiomics_image_transformer.extract_2_list(Wavelet_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "Wavelet",
                                                     Wavelet_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                            if i == "Square":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    Square_generator = imageoperations.getSquareImage(image,
                                                                                      label)
                                Square_img_list = Pyradiomics_image_transformer.extract_2_list(Square_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "Square",
                                                     Square_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                            if i == "SquareRoot":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    SquareRoot_generator = imageoperations.getSquareRootImage(image,
                                                                                              label)
                                SquareRoot_img_list = Pyradiomics_image_transformer.extract_2_list(SquareRoot_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "SquareRoot",
                                                     SquareRoot_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    transformed_img_paths.append(t_img)

                            if i == "Logarithm":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    Logarithm_generator = imageoperations.getLogarithmImage(image,
                                                                                            label)
                                Logarithm_img_list = Pyradiomics_image_transformer.extract_2_list(Logarithm_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "Logarithm",
                                                     Logarithm_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                            if i == "Exponential":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    Exponential_generator = imageoperations.getExponentialImage(image,
                                                                                                label)
                                Exponential_img_list = Pyradiomics_image_transformer.extract_2_list(Exponential_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "Exponential",
                                                     Exponential_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    transformed_img_paths.append(t_img)

                            if i == "Gradient":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    Gradient_generator = imageoperations.getGradientImage(image,
                                                                                          label)
                                Gradient_img_list = Pyradiomics_image_transformer.extract_2_list(Gradient_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "Gradient",
                                                     Gradient_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                            if i == "LBP2D":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    LBP2D_generator = imageoperations.getLBP2DImage(image,
                                                                                    label)
                                LBP2D_img_list = Pyradiomics_image_transformer.extract_2_list(LBP2D_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "LBP2D",
                                                     LBP2D_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                            if i == "LBP3D":
                                # no further parameter set
                                if len(doc[i]) == 0:
                                    # default settings
                                    LBP3D_generator = imageoperations.getLBP3DImage(image,
                                                                                    label)
                                LBP3D_img_list = Pyradiomics_image_transformer.extract_2_list(LBP3D_generator)
                                t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                                                     "LBP3D",
                                                     LBP3D_img_list,
                                                     img_path)

                                for t_img in t_img_paths_list:
                                    self.transformed_img_paths.append(t_img)

                self.transformed_img_paths_csv.append(pd.DataFrame(self.transformed_img_paths, columns=['Image']),
                                                                             ignore_index=True)
                self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(seg_path)

    def run_from_list(self):


        # TODO add for all filters to get the transformed image and save it from input list
        for img_path, seg_path in zip(self.df.Image, self.df.Mask):
            image = sitk.ReadImage(img_path)
            label = sitk.ReadImage(seg_path)
            self.transformed_img_paths_csv = self.transformed_img_paths_csv.append(
                pd.DataFrame([img_path], columns=['Image']),
                ignore_index=True)
            self.transformed_img_paths_csv['Mask'] = self.transformed_img_paths_csv['Mask'].fillna(seg_path)

            for kernel in self.kernels:

                if kernel in self.options_of_filters:

                    if kernel == "LoG":
                        log_generator = imageoperations.getLoGImage(image,
                                                                    label,
                                                                    sigma=[1])
                    if kernel == "Gaussian":
                        gaussian_generator = imageoperations.getGaussianImage(image,
                                                                              label,
                                                                              sigma=[1])
                    if kernel == "Square":
                        Square_generator = imageoperations.getSquareImage(image,
                                                                            label)
                    if kernel == "SquareRoot":
                        SquareRoot_generator = imageoperations.getSquareRootImage(image,
                                                                                    label)
                    if kernel == "Logarithm":
                        Logarithm_generator = imageoperations.getLogarithmImage(image,
                                                                                label)
                    if kernel == "Exponential":
                        Exponential_generator = imageoperations.getExponentialImage(image,
                                                                                        label)
                    if kernel == "Gradient":
                        Gradient_generator = imageoperations.getGradientImage(image,
                                                                                  label)
                    if kernel == "LBP2D":
                        LBP2D_generator = imageoperations.getLBP2DImage(image,
                                                                                label)
                    if kernel == "LBP3D":
                        LBP3D_generator = imageoperations.getLBP3DImage(image,
                                                                                label)

                        LBP3D_img_list = Pyradiomics_image_transformer.extract_2_list(LBP3D_generator)
                        t_img_paths_list = Pyradiomics_image_transformer.save_transformed_img(
                            "LBP3D",
                            LBP3D_img_list,
                            img_path)

                    if kernel == "Wavelet":
                        Wavelet_generator = imageoperations.getWaveletImage(image,
                                                                                    label)


                else:
                    print("Kernel not available")
                    break
                
    def process(self):

        if self.yaml_file != "":
            Pyradiomics_image_transformer.run_from_yaml_file()

        elif self.kernels != None:
            # TODO: Perform if specific kernels for transformation are given! --> currently only running if yaml files are there
            print("Kernel: ", self.kernels, " detected!")



        self.transformed_img_paths_csv.to_csv(self.output_dir + '/transformed_img.csv')







