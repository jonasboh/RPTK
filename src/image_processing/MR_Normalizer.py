import numpy as np
import os
import tqdm
from multiprocessing.pool import Pool
import random
import sys

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
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import Normalizer
from skimage.segmentation import expand_labels
from statistics import mean
from pathlib import Path  # handle path dirs

import gzip
import shutil

# +
import sys

import SimpleITK as sitk
from skimage.measure import label
from skimage.measure import regionprops
import skimage.morphology
import operator
import path
import time
import datetime
import logging
import glob

import concurrent.futures

from sklearn.metrics import jaccard_score
from skimage.segmentation import expand_labels
from statistics import mean

from rptk.src.config.Log_generator_config import LogGenerator
import rptk.src.segmentation_processing.SegProcessor

from pathlib import Path  # handle path dirs

import gzip
import shutil
import pandas as pd

#from src.image_processing.DataHandler import *


class MR_Normalizer:
    """
    Normalize nifti Scans or Segmentations to z-score normalization and N4BF normalization.
    """

    def __init__(self,
                 img_paths=None,
                 logger=None,
                 n_cpu: int = 1,
                 outpath: str = "",
                 chunksize: int = 1
                 ):

        if img_paths is None:
            img_paths = []

        self.img_paths = img_paths
        self.logger = logger
        self.n_cpu = n_cpu
        self.outpath = outpath
        self.chunksize = chunksize

        if not self.outpath.endswith("/"):
            self.outpath = self.outpath + "/"

        if self.logger is None:
            self.logger = LogGenerator(log_file_name=self.outpath + "normalizer.log",
                                       logger_topic="Normalization").generate()
            
            
        os.environ['OMP_NUM_THREADS'] = str(self.n_cpu)

    def z_score_normalize(self, image_path):
        """
        Write normalized Images via z-score normalization
        :param image_path:
        """

        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)

        mean_value = image_array.mean()
        std = image_array.std()
        transformed_image_array = (image_array - mean_value) / (max(std, 1e-8))
        transformed_image_array = transformed_image_array.astype(np.uint16)
        transformed_image_array = np.nan_to_num(transformed_image_array)

        # generate file name
        transformed_file_name = os.path.basename(image_path)[:-len(".nii.gz")] + "_z_score_normalized.nii.gz"

        # write image to preprocessed images  array, out_file_name, path
        # self.logger.info("Z-score normalization: " + os.path.basename(image_path))
        write_nifti_file(transformed_image_array, self.outpath + transformed_file_name, image_path, self.logger)

    def z_normalize_imge_exe(self):
        """
        Execute z-score normalization in parallel
        """
        self.logger.info("Start z-score normalization")
        self.logger.info("Write z-score normalization to " + self.outpath)
        self.logger.info("Z-score normalization of " + str(len(self.img_paths)) + " images")
        
        with tqdm.tqdm(total=len(self.img_paths), desc='Performing Z-score normalization') as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                # futures = {executor.submit(func, row): row for row in chunk}
                # for future in concurrent.futures.as_completed(futures):
                for results in executor.map(self.z_score_normalize, [row for row in self.img_paths], chunksize=self.chunksize):
                    pbar.update(1)
        
        #p = Pool(self.n_cpu)
        #p.map(self.z_score_normalize, self.img_paths, chunksize=self.chunksize)
        #p.close()
        #p.join()

    def N4BiasFieldCorrection(self,
                              image_path,
                              mask_path=None,
                              numberFittingLevels: int = 4,
                              MaxIterations: int = 5):
        """
        Write normalized Images via N4BiasFieldCorrection
        :param image_path: path to image
        :param mask_path: path to mask
        :param numberFittingLevels: number of fitting levels
        :param MaxIterations: max iterations
        """

        inputimage = sitk.ReadImage(image_path, sitk.sitkFloat32)
        image = inputimage

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([MaxIterations] * numberFittingLevels)

        if mask_path is not None:
            maskimage = sitk.ReadImage(mask_path, sitk.sitkUInt8)
            corrected_image = corrector.Execute(image, maskimage)
        else:
            maskimage = sitk.OtsuThreshold(image)
            corrected_image = corrector.Execute(image, maskimage)

        log_bias_field = corrector.GetLogBiasFieldAsImage(inputimage)

        corrected_image_full_resolution = inputimage / sitk.Exp(log_bias_field)

        #corr_array = sitk.GetArrayFromImage(corrected_image_full_resolution)

        #transformed_file_name = os.path.basename(image_path)[:-len(".nii.gz")] + "_N4BF_normalized.nii.gz"

        #write_nifti_file(corr_array, self.outpath + transformed_file_name, image_path, self.logger)

        #log_bias_field_arr = sitk.GetArrayFromImage(log_bias_field)
        #log_bias_field_file_name = os.path.basename(image_path)[:-len(".nii.gz")] + "_N4BF_log_bias.nii.gz"

        #write_nifti_file(log_bias_field_arr, self.outpath + log_bias_field_file_name, image_path, self.logger)

        log_bias_corrected_image_arr = sitk.GetArrayFromImage(corrected_image)
        # replace NaNs with 0
        log_bias_corrected_image_arr = np.nan_to_num(log_bias_corrected_image_arr)
        log_bias_field_corrected_image_name = os.path.basename(image_path)[:-len(".nii.gz")] + "_N4BF_normalized.nii.gz"

        # self.logger.info("N4BF normalization: " + os.path.basename(image_path))
        write_nifti_file(log_bias_corrected_image_arr, self.outpath + log_bias_field_corrected_image_name,
                              image_path, self.logger)

    def N4BF_exe(self):
        """
        Execute N4BiasFieldCorrection in parallel
        :param img_paths: list of image paths
        """

        self.logger.info("Start N4BiasFieldCorrection")
        self.logger.info("Write N4BiasFieldCorrection to " + self.outpath)
        self.logger.info("N4BF normalization of " + str(len(self.img_paths)) + " images")
        
        with tqdm.tqdm(total=len(self.img_paths), desc='Performing N4BiasFieldCorrection') as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                # futures = {executor.submit(func, row): row for row in chunk}
                # for future in concurrent.futures.as_completed(futures):
                for results in executor.map(self.N4BiasFieldCorrection, [row for row in self.img_paths], chunksize=self.chunksize):
                    pbar.update(1)
                    # result = pd.concat([result, results.to_frame().T], ignore_index=True)
        
        #p = Pool(self.n_cpu)
        #p.map(self.N4BiasFieldCorrection, self.img_paths, chunksize=self.chunksize)
        #p.close()
        #p.join()

def write_nifti_file(array, out_file_name, path, logger):
    """
    Write a NIfTI file with the header information from an existing file.
    :param array: numpy.ndarray The array is to be saved to a NIfTI file.
    :param header_file_path : str The path to the existing NIfTI file that contains the header information.
    :param output_file_path : str The path where the output NIfTI file should be saved.
    """

    # Load the existing header information from the header file

    sitk_img = sitk.ReadImage(path)

    if type(array) == list:
        for arr in array:
            sitk_img_out = sitk.GetImageFromArray(arr.astype(np.uint16))
            sitk_img_out.CopyInformation(sitk_img)
            logger.error("ERROR: Need to implement multiple filenames for multiple images")
    else:
        sitk_img_out = sitk.GetImageFromArray(array.astype(np.uint16))
        sitk_img_out.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_img_out, out_file_name, useCompression=True)
