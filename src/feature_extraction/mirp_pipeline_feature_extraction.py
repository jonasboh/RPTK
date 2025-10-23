from __future__ import print_function

import os, psutil

import numpy as np
import pandas as pd

from p_tqdm import p_map
from threading import Semaphore
from iteration_utilities import duplicates

import sys
# sys.path.append('src')

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.mirp.imageClass import *
from rptk.mirp.roiClass import *

# import mirp_predict_lib as mirp_predict
# from mirp_predict_lib import *

# from mirp_pipeline.Preprocessor import Preprocessor
from rptk.src.config.Experiment_setup import *

from tqdm import *
import glob
import re
import multiprocessing

from threading import Thread
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

import logging

import argparse

parser = argparse.ArgumentParser(description='Extract Radiomics features with MIRP Pipeline. Programmed by Jonas Bohn (2021). No Warranty for usage.', add_help=True)
parser.add_argument('-in', '--input_csv', help='Path to CSV file for Image and Segmentation paths.')
parser.add_argument('-cpu', '--num_cpu', type=int, default=1, help='Number of CPUs to process. Set to 0 to use all available CPUs.')
parser.add_argument('-o', '--output', help='Path to output dir.')
parser.add_argument('-m', '--modality', help='Specify modality (CT or MRI).')

args = vars(parser.parse_args())

out_path = args["output"]
CPUs = args["num_cpu"]
path2confCSV = args["input_csv"]
modality = args["modality"]

# +
if CPUs == 0:
    CPUs = multiprocessing.cpu_count()
    
if CPUs == multiprocessing.cpu_count():
    CPUs = CPUs - 1
# -

log_file_name = os.path.basename(path2confCSV)[:-len(".csv")] + "_mirp_extraction.log"

logger = logging.getLogger("Mirp Pipeline")
logger.setLevel(logging.INFO)

# create file handler which logs info messages
fh = logging.FileHandler(log_file_name, 'w', 'utf-8')
fh.setLevel(logging.INFO)

# creating a formatter
formatter = logging.Formatter('%(name)s - %(levelname)-8s: %(message)s')

# setting handler format
fh.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)

def get_info_from_result_file(file_name, feat_config_string):
    """
    Extracts the patient ID, ROI ID, ROI type, ROI side and ROI number from the file name.
    Input Format: PATID-Number_img_timepoint_config_PATID-Number_seg_Rator_timepoint_config_transformation.csv

    Args:
        file_name (str): Name of the file to extract information from.
        feat_config_string (str): String to identify the feature configuration in the file name.

    Returns:
        dict with:
            patient_ID (str): Patient ID.
            img_timepoint (str): Timepoint of the image.
            img_config (str): Configuration of the image.
            seg_timepoint (str): Timepoint of the segmentation.
            seg_config (str): Configuration of the segmentation.
            transformation_ID (str): ID of the transformation.
            file_name (str): Name of the file.

    """

    if file_name.endswith(".csv"):
        feat_conf_pos = file_name.find(feat_config_string)
        feat_conf = file_name[feat_conf_pos:-len(".csv")]

        # Get configuration from file name
        splitted_file = file_name.split('_')
        img_name = splitted_file[0]
        img_timepoint = splitted_file[2]
        img_config = splitted_file[3]
        seg_name = splitted_file[4]
        seg_rator = splitted_file[6]
        seg_timepoint = splitted_file[7]
        seg_config = splitted_file[8]

        transformation_ID = feat_conf

        entry = pd.DataFrame.from_dict({
            'img_name': [img_name],
            'img_timepoint': [img_timepoint],
            'img_config': [img_config],
            'seg_name': [seg_name],
            'seg_rator': [seg_rator],
            'seg_timepoint': [seg_timepoint],
            'seg_config': [seg_config],
            'transformation_ID': [transformation_ID],
            'file_name': [file_name]
        })

        return entry

def sum_output_files(out_path):
    """
    Get all outfiles from outfolder, seperate informations of the filename in coloumns and aggregate in pd.DataFrame
    
    Args:
        out_path path to output folder
        
    Returns:
        pd.Dataframe with all seperated configurations for feature calculation
    
    """
    
    
    result_file_informations = pd.DataFrame(columns=['img_name', 'seg_config'])

    for file_name in os.listdir(out_path):
        if file_name.endswith("features.csv"):
            entry = get_info_from_result_file(file_name, feature_configuration_string)
            result_file_informations = pd.concat([result_file_informations,entry])
            
    # sort results file
    result_file_informations.sort_values(by=['img_name', 'seg_config'], ascending=True)
            
    return result_file_informations

def get_unique_transformations(transformation_list, feature_configuration_string):
    """
    Add _ chacter to name of transformations to avoid substring included transforamtions
    
    Args:
        List of strings as names of applied transformations
        
    Returns:
        List of unique transformations where '_' has been added to the front
    """
    
    unique_transformations = []
    for transformation in transformation_list:
        unique_transformation = feature_configuration_string + "_" + transformation + "_features"
        unique_transformations.append(unique_transformation)
    
    # add string for non transformed feature calculations
    unique_transformations.append(feature_configuration_string + "_features") 
    
    return unique_transformations

def generate_unique_img_seg_conf_string(row):
    """
    Get all information from pd.Series to concatinate them into unique string
    
    Args:
        row: pd.Series with coloumns for image and segmenations formatioguration
        
    Returns:
        configuration string
    """
    
    unique_sample_index = row["img_name"] + "_" + row["img_timepoint"] + "_" + row["img_config"] + "_" + row["seg_name"] + "_" + row["seg_rator"] + "_" + row["seg_timepoint"] + "_" + row["seg_config"]
    
    return unique_sample_index

def generate_image_file_name(row):
    
    image_file_name = row["img_name"] + "_img_" + row["img_timepoint"] + "_" + row["img_config"] + ".nii.gz" 
    
    return image_file_name

def generate_seg_file_name(row):
    
    seg_file_name = row["seg_name"] + "_seg_" + row["seg_rator"] + "_" + row["seg_timepoint"] + "_" + row["seg_config"] + ".nii.gz"
    
    return seg_file_name

def get_transformations_per_samples(result_file_informations):
    """
    Summarizes all transformations for each file configuration for further checks.
    
    Args:
        pd.DataFrame with img_name img_timepoint img_config seg_name seg_rator seg_timepoint seg_config transformation_ID from sum_output_files function
        
    Return:
        dict with unique ID from the obove mentioned columns and the transformation_ID list as values
    """
    
    
    performed_calculations = {} # ID
    for index, row in result_file_informations.iterrows():
        unique_sample_index = generate_unique_img_seg_conf_string(row)

        if unique_sample_index not in performed_calculations.keys():
            performed_calculations[unique_sample_index] = [row["transformation_ID"]]
        else:
            performed_calculations[unique_sample_index].append(row["transformation_ID"])
            
    return performed_calculations 

def get_missing_transformations(performed_calculations, unique_MIRP_transformations):
    """
    Get missing not performed transformations from summary of all perfomed transformations
    
    Args:
        performed_calculations: dict with unique ID of input files configuration and list of performed transformations as value
        unique_MIRP_transformations: configurated stings which are matching the transformation ID from the feature calculations which should be applied according to the setting
        
    Return:
        missing_transformations: dict with ID as index and non performed transformation for this configuration as value
    
    """
    
    missing_transformations = {}
    # 1. Go thouth the Samples
    for performed_calculation in performed_calculations:
        # 2. go though all transformations for each sample
        for needed_transformations in unique_MIRP_transformations:
            # 3. check if transformations are missing
            if needed_transformations not in performed_calculations[performed_calculation]:
                if needed_transformations not in missing_transformations.keys():
                    missing_transformations[performed_calculation] = [needed_transformations]
                else:
                    missing_transformations[performed_calculation].append(needed_transformations)
                    
    return missing_transformations

def get_duplicated_transformations(performed_transformations, unique_MIRP_transformations):
    """
    Checks if the transformations for each sample has been calculated more than once
    
    Args:
        performed_transformaions: dict with unique ID for each image/seg configuration and a list as value with all applied transformations
        unique_MIRP_transformations: list of all tarsnformations which should be applied
        
    Returns: 
        duplicated_transformations: dict with ID of each image/seg configuration and a list as value with all duplicated transformations
    """
    
    duplicated_transformations = {}
    for performed_transformation in performed_transformations:
        for needed_transformation in unique_MIRP_transformations:
            if performed_transformations[performed_transformation].count(needed_transformation) > 1:
                if performed_transformation not in duplicated_transformations.keys():
                    duplicated_transformations[performed_transformation] = [needed_transformation]
                else:
                    duplicated_transformations[performed_transformation].append(needed_transformation)
                    
    return duplicated_transformations

def get_infos_from_seg(file_name):
    """
    Extracts the patient ID, ROI ID, ROI type, ROI side and ROI number from the file name.
    Input Format: PATID-Number_seg_Rator_timepoint_config_transformation.nii.gz
    """

    splitted_file_name = file_name.split('_')

    patient_ID = splitted_file_name[0]
    rator = splitted_file_name[2]
    timepoint = splitted_file_name[3]
    config = splitted_file_name[4]

    if config.endswith(".nii.gz"):
        config = splitted_file_name[4][:-len(".nii.gz")]

    return patient_ID, rator, timepoint, config

def get_infos_from_image(file_name):
    """
    Extracts the patient ID, ROI ID, ROI type, ROI side and ROI number from the file name.
    Input Format: PATID-Number_img_timepoint_config
    """
    splitted_file_name = file_name.split('_')

    patient_ID = splitted_file_name[0]
    timepoint = splitted_file_name[2]
    config = splitted_file_name[3][:-len(".nii.gz")]

    return patient_ID, timepoint, config


def get_infos_from_config_csv(path2confCSV):
    """
    Get informations from config csv file with coloumns Image:path to image files and Mask: path to segmeantaion file.
    Extract informations for configuration from file names and give a dataframe with all informations:
    
    Args:
        path2confCSV: path to config csv
        
    Returns:
        pd.DataFrame with configuration of image and segmentation
    """
    
    input_csv = pd.read_csv(path2confCSV)

    df = pd.DataFrame(columns=['img_name',
                               'img_timepoint',
                               'img_config',
                               'seg_name',
                               'seg_rator',
                               'seg_timepoint',
                               'seg_config',
                               'img_path',
                               'seg_path'])

    for index, row in input_csv.iterrows():
        image_file_name = os.path.basename(row["Image"])
        seg_file_name = os.path.basename(row["Mask"])

        img_patient_ID, img_timepoint, img_config = get_infos_from_image(image_file_name)
        seg_patient_ID, seg_rator, seg_timepoint, seg_config = get_infos_from_seg(seg_file_name)

        entry = pd.DataFrame.from_dict({
                'img_name': [img_patient_ID],
                'img_timepoint': [img_timepoint],
                'img_config': [img_config],
                'seg_name': [seg_patient_ID],
                'seg_rator': [seg_rator],
                'seg_timepoint': [seg_timepoint],
                'seg_config': [seg_config],
                'img_path': [row["Image"]],
                'seg_path': [row["Mask"]]
            })
        df = pd.concat([df, entry])
        
    return df

def get_missing_samples(input_data, performed_transformations):
    """
    Get performed data and compare to read data.
    Args:
        input_data: pd.DataFrame with all information's about the data from the conf csv
        performed_transformations: dict with unique IDs as keys and list of performed transformations as values
    
    Returns:
        pd.Dataframe with path to images and Masks
    """
    
    
    df = pd.DataFrame(columns=['Image', 'Mask'])

    for index, row in input_data.iterrows():
        unique_sample_index = generate_unique_img_seg_conf_string(row)
        if unique_sample_index not in performed_transformations.keys():
            tmp = pd.DataFrame.from_dict({"Image": [row["img_path"]], "Mask": [row["seg_path"]]})
            df = pd.concat([df, tmp])
            
    return df


def process(experiment):
    logger.info("Processing experiment - " + experiment.subject + " " + experiment.settings.general.config_str)
    experiment.process()
    return experiment

def worker(experiment):
    global ready_list
    ready_list.append(experiment.settings.general.config_str)

if __name__ == "__main__":

    ''' Formatting the files: Files from csv should be in the following format:
        /path/to/image.nii.gz, /path/to/segmentation.nii.gz, /path/to/segmentation_2.nii.gz, /path/to/segmentation_3.nii.gz
        Several segmentwations are possible. These could include segmentations from different ROIs but need the "_<Number or ID>".nii.gz format.
    '''

    p_out = ""
    
    missing_files = {}  # Dictionary with IDs as keys and a list of missing strings as values

    # TODO: Make reasonable config strings
    feature_configuration_string = "raw_intra"  # Sting which is the name of the folder where the features are stored

    out_path_exist = os.path.exists(out_path)
    if not out_path_exist:
        # Create a new directory because it does not exist
        os.makedirs(out_path)

    # generate dir for configuration
    if out_path.endswith("/"):
        p_out = out_path
        out_path = out_path + os.path.basename(path2confCSV)[:-len(".csv")]
    else:
        p_out = out_path + "/"
        out_path = out_path + "/" + os.path.basename(path2confCSV)[:-len(".csv")]

    out_path_exist = os.path.exists(out_path)
    if not out_path_exist:
        # Create a new directory because it does not exist
        os.makedirs(out_path)
    
    logger.info('### Settings ###')
    logger.info('CSV input file path: ' + path2confCSV)
    logger.info('Feature extraction config: ' + feature_configuration_string)
    logger.info('Output folder: ' + out_path)
    logger.info('Log file: ' + log_file_name)
    logger.info('CPUS Used: ' + str(CPUs))
    logger.info('Process ID: ' + str(os.getpid()) + '\n')

    logger.info('Start MIRP feature extraction ...')
    
    preprocessor = Preprocessor(path_2_intra_conf_csv=path2confCSV,
                                 path_2_peri_conf_csv="",
                                 name_mask1="Mask_intra",
                                 name_mask2="Mask_peri",
                                 make_peritumoral_mask=False,
                                 outpath=out_path)

    Samples = preprocessor.combine_segmentations()

    feature_computation_settings = FeatureExtractionSettingsClass(
                                        by_slice=False,
                                        no_approximation=False,
                                        ibsi_compliant=True,
                                        base_feature_families="all",
                                        base_discretisation_method="fixed_bin_size", # Sync with pyradiomics
                                        base_discretisation_bin_width=25.0,  # show best stability in paper
                                        ivh_discretisation_method="fixed_bin_size",
                                        ivh_discretisation_bin_width=25.0,
                                        glcm_distance=1.0,
                                        glcm_spatial_method=["2d_average", "2d_slice_merge", "3d_average", "3d_volume_merge"],
                                        glrlm_spatial_method=["2d_average", "2d_slice_merge", "3d_average", "3d_volume_merge"],
                                        glszm_spatial_method=["2d", "3d"],
                                        gldzm_spatial_method=["2d", "3d"],
                                        ngtdm_spatial_method=["2d", "3d"],
                                        ngldm_spatial_method=["2d", "3d"],
                                        ngldm_distance=[1.0],
                                        ngldm_difference_level=[0.0])

    MIRP_transformations = ["laws", "gabor", "gaussian", "nonseparable_wavelet", "separable_wavelet", "mean"]


    experiment_generator = Experiment_Generator(MIRP_transformations=MIRP_transformations,
                                               modality=modality,
                                               Feature_extraction_settings=feature_computation_settings,
                                               write_path=out_path,
                                               df=Samples,
                                               feature_calc_conf=[feature_configuration_string],
                                               logger=logger,
                                               log_file_name=log_file_name)

    logger.info("Generating experiments ...")
    experiments = experiment_generator.generate_experiments()
    logger.info(f"Need to process {len(experiments)} experiments with {CPUs} CPUs.")

    if len(experiments) > 0:
        logger.info(f"Files go to {experiments[0].write_path}.")

    logger.info("Pool of experiments generated. Starting to process ...")

    pool = multiprocessing.Pool(processes=CPUs, maxtasksperchild=1) #, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    ready_list = []
    result = {}

    with Pool(processes=CPUs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool:
        pool.imap_unordered(process, experiments, chunksize=1)
        with tqdm(total=len(experiments)) as pbar:
            res = pool.apply_async(process, (experiments))
            for i, _ in enumerate(pool.imap_unordered(process, experiments, chunksize=1)):
                pbar.update()
    
    #for exp in experiments:
    #    result[exp.settings.general.config_str] = (pool.apply_async(process, (exp,), callback=worker))
        #with tqdm(total=len(experiments)) as pbar:
    #    for ready in ready_list:
    #        # logger.info("Memory Usage:" + get_memory() + " GB")
    #        result[ready].wait()
    #        #pbar.update()
    #        del result[ready]
    #    ready_list = []

    # clean up
    pool.close()
    pool.join()
    
    result_file_informations = sum_output_files(out_path)
    unique_MIRP_transformations = get_unique_transformations(MIRP_transformations, feature_configuration_string)

    performed_transformations = get_transformations_per_samples(result_file_informations)

    missing_transformations = get_missing_transformations(performed_transformations, unique_MIRP_transformations)

    duplicated_transformations = get_duplicated_transformations(performed_transformations, unique_MIRP_transformations)

    input_data = get_infos_from_config_csv(path2confCSV)
    
    missing_samples = get_missing_samples(input_data, performed_transformations)
    
    logger.info("Checking output files ...")
    
    if len(missing_samples) > 0:
        logger.info("Number of not processed samples/transformations:" + str(len(missing_samples)))
        for index, file in missing_samples.iterrows():
            logger.info("Missing Image:" + os.path.basename(file["Image"]) + " Missing Mask:" + str( os.path.basename(file["Mask"])))
        
        missing_samples.to_csv(p_out + os.path.basename(path2confCSV)[:-len(".csv")] + "_missing_file_summary.csv", index=False)
    else:
        logger.info("All samples/transformations processed!")
                        
    if len(duplicated_transformations) > 1:
        logger.info("Duplicated calculations found!")
        for key in duplicated_transformations:
            logger.info("Duplicated: " + key + " " + str(duplicated_transformations[key]))
        pd.DataFrame.from_dict(duplicated_transformations).to_csv(p_out + os.path.basename(path2confCSV)[:-len(".csv")] + "_duplicated_file_summary.csv", index=False)
                        
    if len(missing_transformations) > 1:
        logger.info("Missing calculations found!")
        for key in missing_transformations:
            logger.info("Duplicated: " + key + " " + str(missing_transformations[key]))
        pd.DataFrame.from_dict(missing_transformations).to_csv(p_out + os.path.basename(path2confCSV)[:-len(".csv")] + "_missing_file_summary.csv", index=False)
                        

    #file_configs, duplicated_file_names = report_duplicated_files_with_different_date(duplicated_files, feature_configuration_string)

    #print(file_configs)
    #print(duplicated_file_names)

    #if len(file_configs) > 0:
    #    logger.info("Detected duplicated files with different date!")
    #    file_configs.to_csv(p_out + os.path.basename(path2confCSV)[:-len(".csv")] + "_duplicated_file_summary.csv", index=False)

    #if len(duplicated_file_names) > 0:
    #    logger.info("Number of duplicated files:" + str(len(duplicated_file_names)))
    #    for file in set(duplicated_file_names):
    #        logger.info("Duplicated file:" + file)
    #        file_path = out_path + "/" + feature_configuration_string + "/" + file
            # os.remove(file_path)


    logger.info("### Finished with all jobs! ###")

