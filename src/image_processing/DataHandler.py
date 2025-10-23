import sys
import os

import SimpleITK as sitk
from skimage.measure import label  # , regionprops
from skimage.measure import regionprops
import skimage.morphology
import argparse
import operator
import path
import time
import datetime
import logging
import glob

from src.config.Log_generator_config import LogGenerator
from src.segmentation_processing.SegProcessor import *

from sklearn.metrics import jaccard_score
from skimage.segmentation import expand_labels
from statistics import mean

from pathlib import Path  # handle path dirs

import gzip
import shutil
import pandas as pd

# Summarize Default Data functions
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
