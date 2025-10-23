from __future__ import print_function

import os, psutil
import sys
import numpy as np
import pandas as pd

from p_tqdm import p_map
from threading import Semaphore
from iteration_utilities import duplicates
import yaml

# directory reach
#directory = os.path.dirname(os.path.realpath('__file__'))
#print(directory)
# setting path
#sys.path.append(directory)

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass
from rptk.mirp.imageClass import *
from rptk.mirp.roiClass import *

#from src.mirp_pipeline.mirp.experimentClass import ExperimentClass
#from src.mirp_pipeline.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
#    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
#    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

#from src.mirp_pipeline.mirp.imageClass import *
#from mirp_pipeline.mirp.roiClass import *

# import mirp_predict_lib as mirp_predict
# from mirp_predict_lib import *

#from src.mirp_pipeline.Preprocessor import Preprocessor
#from src.mirp_pipeline.Experiment_setup import *

from tqdm import *
import glob
import re
import multiprocessing

from threading import Thread
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

import logging

class MIRP_Extractor:
    def __init__(self,
                 path2CSV,
                 out_path,
                 path2yaml: str = None,
                 n_cpu: int = os.cpu_count() - 1,
                 cohort_ID: str = None,
                 chunksize: int = 1):

        self.out_path = out_path
        self.path2yaml = path2yaml
        self.n_cpu = n_cpu
        self.path2CSV = path2CSV
        self.cohort_ID = cohort_ID
        self.chunksize = chunksize

        # Will be configured according to data config csv
        self.modality = None

        log_file_name = self.out_path + os.path.basename(self.path2CSV)[:-len(".csv")] + "_mirp_extraction.log"

        self.logger = logging.getLogger("MIRP Pipeline")
        self.logger.setLevel(logging.INFO)

        # create file handler which logs info messages
        fh = logging.FileHandler(log_file_name, 'w', 'utf-8')
        fh.setLevel(logging.INFO)

        # creating a formatter
        formatter = logging.Formatter('%(name)s - %(levelname)-8s: %(message)s')

        # setting handler format
        fh.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)

        # Extract information about the data
        self.data_config = self.extract_info_from_csv()

        if self.modality != None:
            # Extract information about mirp extractor configuration
            self.mirp_config = self.extract_config_from_yaml(modality=self.modality)


    def extract_info_from_csv(self):
        df = pd.read_csv(self.path2CSV)

        # only one Modality is in the Dataset
        if len(df["Modality"].unique()) == 1:
            self.modality = df["Modality"].values[0]
        else:
            self.logger.info("Warning: Multiple modalities in the dataset! Currently supported modalities are CT and MR.")


        return df

    def extract_config_from_yaml(self, modality: str = None):
        """
        Extracts the configuration from the yaml file
        :param modality: "CT" or "MR" for according feature extraction settings
        :return: mirp_config: dict with all settings for the mirp feature extraction
        """

        # If Modality is specified, use default config for CT or MR
        if self.path2yaml != None:
            path2yaml = self.path2yaml
        elif modality == "CT":
            path2yaml = "/config/mirp/CT.yaml"
        elif modality == "MR":
            path2yaml = "/config/mirp/MRI.yaml"
        else:
            self.logger.error("Modality not supported, please correct! Currently supported modalities are CT and MR.")
            print("Modality not supported, please correct! Currently supported modalities are CT and MR.")
            exit()

        with open(path2yaml, 'r') as file:
            mirp_config = yaml.safe_load(file)

        return mirp_config

    def generate_settings(self):
        """
        Generates the settings for the mirp feature extraction
        :return: settings: dict with all settings for the mirp feature extraction
        """

        # Generate settings
        general_settings = GeneralSettingsClass(
            by_slice=self.mirp_config["general_parameters"]["by_slice"]
        )

        # Generate interpolate settings
        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=self.mirp_config["general_parameters"]["by_slice"],
            interpolate=self.mirp_config["image_interpolation_parameters"]["interpolate"],
            anti_aliasing=self.mirp_config["image_interpolation_parameters"]["anti_aliasing"]
        )

        # Generate perturbation settings
        perturbation_settings = ImagePerturbationSettingsClass()

        # Generate feature extraction settings
        feature_computation_parameters = FeatureExtractionSettingsClass(
                by_slice=self.mirp_config["feature_extraction_parameters"]["by_slice"],
                no_approximation=self.mirp_config["feature_extraction_parameters"]["no_approximation"],
                ibsi_compliant=self.mirp_config["feature_extraction_parameters"]["ibsi_compliant"],
                base_feature_families=self.mirp_config["feature_extraction_parameters"]["base_feature_families"],
                base_discretisation_method=self.mirp_config["feature_extraction_parameters"]["base_discretisation_method"],
                base_discretisation_bin_width=self.mirp_config["feature_extraction_parameters"]["base_discretisation_bin_width"],
                ivh_discretisation_method=self.mirp_config["feature_extraction_parameters"]["ivh_discretisation_method"],
                ivh_discretisation_bin_width=self.mirp_config["feature_extraction_parameters"]["ivh_discretisation_bin_width"],
                glcm_distance=self.mirp_config["feature_extraction_parameters"]["glcm_distance"],
                glcm_spatial_method=self.mirp_config["feature_extraction_parameters"]["glcm_spatial_method"],
                glrlm_spatial_method=self.mirp_config["feature_extraction_parameters"]["glrlm_spatial_method"],
                glszm_spatial_method=self.mirp_config["feature_extraction_parameters"]["glszm_spatial_method"],
                gldzm_spatial_method=self.mirp_config["feature_extraction_parameters"]["gldzm_spatial_method"],
                ngtdm_spatial_method=self.mirp_config["feature_extraction_parameters"]["ngtdm_spatial_method"],
                ngldm_spatial_method=self.mirp_config["feature_extraction_parameters"]["ngldm_spatial_method"],
                ngldm_distance=self.mirp_config["feature_extraction_parameters"]["ngldm_distance"],
                ngldm_difference_levels=self.mirp_config["feature_extraction_parameters"]["ngldm_difference_level"],
            )

        # Generate image transformation settings
        image_transformation_settings = ImageTransformationSettingsClass(
                                                                        by_slice=False,
                                                                        response_map_feature_settings=feature_computation_parameters)

        # Generate resegmentation settings
        if self.mirp_config["resegmentation_parameters"]["resegmentation_method"] == "threshold":
            resegmentation_settings = ResegmentationSettingsClass(resegmentation_method=self.mirp_config["resegmentation_parameters"]["resegmentation_method"],
                                                                  resegmentation_intensity_range=self.mirp_config["resegmentation_parameters"]["resegmentation_intensity_range"],
                                                                  resegmentation_sigma=self.mirp_config["resegmentation_parameters"]["resegmentation_sigma"]
                                                                  )
        elif self.mirp_config["resegmentation_parameters"]["resegmentation_method"] == "range":
            resegmentation_settings = ResegmentationSettingsClass(resegmentation_method=self.mirp_config["resegmentation_parameters"]["resegmentation_method"],
                                                                  resegmentation_sigma=self.mirp_config["resegmentation_parameters"]["resegmentation_sigma"]
                                                                  )

        # Generate roi interpolation settings
        roi_interpolate_settings = RoiInterpolationSettingsClass(roi_spline_order=self.mirp_config["roi_interpolate_parameters"]["roi_spline_order"],
                                                                 roi_interpolation_mask_inclusion_threshold=self.mirp_config["roi_interpolate_parameters"]["roi_interpolation_mask_inclusion_threshold"],
                                                                 )
        # Summarize settings
        settings = SettingsClass(
            general_settings=general_settings,
            post_process_settings=ImagePostProcessingClass(),
            img_interpolate_settings=image_interpolation_settings,
            roi_interpolate_settings=roi_interpolate_settings,
            roi_resegment_settings=resegmentation_settings,
            perturbation_settings=perturbation_settings,
            img_transform_settings=image_transformation_settings,
            feature_extr_settings=feature_computation_parameters
        )


        return settings

    def generate_experiments(self):
        """
        Generates the experiments for the mirp feature extraction
        :return: experiments: list of experiments
        """

        if self.modality != None:
            self.setting = self.generate_settings()

        experiments = []

        for index, row_ in self.data_config.iterrows():

            # If there are multiple modalities in the same cohort: select settings according to each sample
            if self.modality == None and self.path2yaml == None:
                modality = row_["Modality"]
                self.mirp_config = self.extract_config_from_yaml(modality=modality)
                setting = self.generate_settings()
            else:
                setting = self.setting

            experiment = ExperimentClass(
                                modality=self.modality,
                                subject=os.path.basename(row_["Image"])[:-len(".nii.gz")]+"_"+os.path.basename(row_["Mask"])[:-len(".nii.gz")],
                                cohort=self.cohort_ID,
                                write_path=self.out_path,
                                image_folder=os.path.dirname(row_["Image"]),
                                roi_folder=os.path.dirname(row_["Mask"]),
                                roi_reg_img_folder=None,
                                image_file_name_pattern=os.path.basename(row_["Image"])[:-len(".nii.gz")],
                                registration_image_file_name_pattern=None,
                                roi_names=[os.path.basename(row_["Mask"])[:-len(".nii.gz")]],
                                data_str=None,
                                provide_diagnostics=self.mirp_config["experiment_parameters"]["provide_diagnostics"],
                                settings=setting,
                                compute_features=self.mirp_config["experiment_parameters"]["compute_features"],
                                extract_images=self.mirp_config["experiment_parameters"]["extract_images"],
                                plot_images=self.mirp_config["experiment_parameters"]["plot_images"],
                                keep_images_in_memory=self.mirp_config["experiment_parameters"]["keep_images_in_memory"]
                            )

            experiments.append(experiment)

        return experiments

    # TODO: extract config information from yaml file, generate experiments, include in settings, generate list, execute in multi threading frame
    # TODO: change functions from file name information extraction to pandas dataframe information extraction
    # TODO: adapt column names from input csv file to new column names

    def process_(self, experiment):
        """
        Processes the experiment
        :param experiment: experiment to be processed
        """

        self.logger.info("Processing experiment - " + experiment.subject + " " + experiment.settings.general.config_str)
        experiment.process()

    def worker(self, experiment):
        global ready_list
        ready_list.append(experiment.settings.general.config_str)

    def execute(self):
        '''
        Executes the mirp feature extraction
        '''

        self.logger.info('### Settings ###')
        self.logger.info('CSV input file path: ' + self.path2CSV)
        self.logger.info('Output folder: ' + self.out_path)
        self.logger.info('CPUS Used: ' + str(self.n_cpu))
        self.logger.info('Process ID: ' + str(os.getpid()) + '\n')

        self.logger.info('### Generate Experiments ###')
        experiments = self.generate_experiments()

        self.logger.info('### Running Experiments ###')
        p = multiprocessing.Pool(processes=self.n_cpu, maxtasksperchild=1)  # , initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

        p.map(self.process_, experiments, chunksize=self.chunksize)
        p.close()
        p.join()

        #with Pool(processes=self.n_cpu, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool:
        #    pool.imap_unordered(process, experiments, chunksize=self.chunksize)
        #    with tqdm(total=len(experiments)) as pbar:
        #        res = pool.apply_async(self.process, (experiments))
        #        for i, _ in enumerate(pool.imap_unordered(process, experiments, chunksize=1)):
        #            pbar.update()

        # clean up
        #pool.close()
        #pool.join()

        self.logger.info("### Finished with all jobs! ###")
