from __future__ import print_function

import os
import psutil
import sys
from threading import Semaphore

import SimpleITK as sitk
import numpy as np
import pandas as pd
import yaml
from iteration_utilities import duplicates
from p_tqdm import p_map
import time
import datetime
import logging
import gc

# from loaders import BarLoader, SpinningLoader, TextLoader

# +
# sys.path.append('src')
# -

# from src.feature_filtering.Radiomics_Filter_exe import RadiomicsFilter
from rptk import rptk
from rptk.src.config.Log_generator_config import LogGenerator

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.mirp.imageClass import *
from rptk.mirp.roiClass import *

# import mirp_predict_lib as mirp_predict
# from mirp_predict_lib import *

# from mirp_pipeline.Preprocessor import Preprocessor
# from mirp_pipeline.Experiment_setup import *

import tqdm
import glob
import re
import multiprocessing

from threading import Thread
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

import logging
import concurrent.futures


# from segmentation_processing.SegProcessor import SegProcessor


class Outfile_validator:
    """
    Class for validating the output files of the RPTK extractor.
    Check if Format of outfile is correct.
    """

    def __init__(self,
                 logger: logging,  # Either PyRadiomics or MIRP
                 df: pd.DataFrame,  # Path to output directory
                 extractor: str,  # Either PyRadiomics or MIRP
                 verbose: int = 1,  # Verbose output level (0=None, 1=Some, 2=Everything)
                 runid: str = None,  # Runid of the experiment
                 output_path: str = None,  # Path to output directory
                 num_cpus: int = 1  # Number of cpus to use
                 ):

        self.logger = logger
        self.df = df
        self.extractor = extractor
        self.verbose = verbose
        self.runid = runid
        self.output_path = output_path
        self.num_cpus = num_cpus

        self.config_columns = ["Modality",
                               "ROI_Label",
                               "Timepoint",
                               "Prediction_Label",
                               "Image",
                               "Mask",
                               "Mask_Transformation",
                               "Image_Transformation",
                               "Rator"]

        self.mirp_config_columns = ["id_subject",
                                    "img_data_config"]

        self.pyradiomics_config_columns = ["original_shape_Elongation",
                                           "original_shape_Flatness"]

        # self.logger.info("Initiating Outfile validator ...")

    def validate_outfile(self):
        """
        Validate the outfile of the extractor.
        Check if all features are present.
        Check if no samples do have nan values.
        :return: pd.DataFrame with missing experiments
        """
        errors = 0
        if self.verbose == 1:
            self.logger.info("Validating outfile ...")

            self.logger.info("Checking for config parameters ...")
        for config_col in self.config_columns:
            if not config_col in self.df.columns:
                errors += 1
                if self.verbose == 1:
                    self.logger.error("No {} column in dataframe. Please check the outfile.".format(config_col))
            else:
                if not self.df[config_col].isnull().all():
                    if self.df[config_col].isnull().sum() > 0:
                        self.logger.error(
                            "Column {} contains some NaN: {}".format(config_col, self.df[config_col].isnull().sum()))
                        self.df[config_col].fillna(0, inplace=True)
                else:
                    self.df.drop(columns=config_col, inplace=True)
                    if self.verbose == 1:
                        self.logger.error("Column {} contains only NaN! Dropping!".format(config_col))
        # if errors > 0:
        #    self.logger.error("Found {} errors for config parameters in the outfile.".format(errors))

        general_config = errors

        if self.extractor == "MIRP":
            if self.verbose == 1:
                self.logger.info("Checking for MIRP config parameters ...")
            for config_col in self.mirp_config_columns:
                if config_col not in self.df.columns:
                    if not self.df.index.name == config_col:
                        errors += 1
                        if self.verbose == 1:
                            self.logger.warning(
                                "No {} column in dataframe. Please check the feature extraction outfile.".format(
                                    config_col))
            if errors > 0:
                if self.verbose == 1:
                    self.logger.error("Found {} errors for mirp config parameters in the outfile.".format(errors))

        if self.extractor == "PyRadiomics":
            if self.verbose == 1:
                self.logger.info("Checking for PyRadiomics config parameters ...")
            for config_col in self.pyradiomics_config_columns:
                if not config_col in self.df.columns:
                    errors += 1
                    if self.verbose == 1:
                        self.logger.warning(
                            "No {} column in dataframe. Please check the feature extraction outfile.".format(
                                config_col))
            if errors > 0:
                if self.verbose == 1:
                    self.logger.error(
                        "Found {} errors for pyradiomics config parameters in the outfile.".format(errors))

        # TODO: There are no nan values :-)
        self.df.fillna("0", inplace=True)
        # restart error counter
        errors = 0
        if self.verbose == 1:
            self.logger.info("Checking for nan values ...")
        nan = self.df[self.df.isna().any(axis=1)]
        if nan.shape[0] > 0:
            # not all samples have nan values
            if nan.shape[0] != self.df.shape[0]:
                for index, row in nan.iterrows():
                    if "Image" in self.df.columns and "Mask" in self.df.columns:
                        # errors += 1
                        id = os.path.basename(row["Image"])[:-len(".nii.gz")] + "_" + \
                             os.path.basename(row["Mask"])[:-len(".nii.gz")]
                        if self.verbose == 1:
                            self.logger.warning("Found nan value in sample: {}".format(id))

            # all samples have nan values
            else:
                if self.verbose == 1:
                    self.logger.error("All Samples have nan values. Please check the outfile.")

                # select columns containing only nan values
                for col in nan.loc[:, nan.isnull().all()]:
                    if self.verbose == 1:
                        self.logger.warning("Feature containing only nan values: {}".format(col))
                    self.df.drop(columns=col, inplace=True)
                    if self.verbose == 1:
                        self.logger.info("Dropping Feature {}".format(col))

                # check if all columns contain only nan values
                if nan.shape[1] != nan.loc[:, nan.isnull().all()].shape[1]:
                    for col in nan.columns:
                        if col not in nan.loc[:, nan.isnull().all()].columns:
                            if self.df[col].isna().sum() > 0:
                                errors += 1
                                if self.verbose == 1:
                                    self.logger.error(
                                        "Feature containing {} nan values: {}".format(self.df[col].isna().sum(), col))
                else:
                    if self.verbose == 1:
                        self.logger.error("All features are nan values. Please check the outfile.")
        else:
            if self.verbose == 1:
                self.logger.info("No nan values found.")

        if errors > 0:
            if self.verbose == 1:
                self.logger.error("Found {} errors for nan values in the outfile.".format(errors - general_config))
                self.logger.error("Please check the outfile.")
            nan.to_csv(self.output_path + self.extractor + self.runid + "_nan_experiments.csv")

            sys.exit()

        self.df.replace("0", np.nan, inplace=True)
        return errors


class MissingExperimentDetector:
    """
    Class to search for missing experiments.
    Seachring in outfile or folder with multiple outfiles
    """

    def __init__(self,
                 extractor: str,  # Either PyRadiomics or MIRP
                 out_path: str,  # Path to output directory
                 verbose: int = 1,  # Verbose output level (0=None, 1=Some, 2=Everything)
                 logger: logging = None,  # Logger object
                 subject_dir: str = "",  # Path to subject directory
                 all_experiments_id=None,  # List of all experiments IDs from Image and Mask
                 RunID: str = "",  # RunID of the experiment
                 num_cpus: int = 1,  # Number of cpus to use
                 error: logging = None,  # Error logger
                 found_output_file_name: str = None
                 ):

        if all_experiments_id is None:
            all_experiments_id = []

        self.extractor = extractor
        self.out_path = out_path
        self.verbose = verbose
        self.logger = logger
        self.subject_dir = subject_dir
        self.all_experiments_id = all_experiments_id
        self.RunID = RunID
        self.num_cpus = num_cpus
        self.error = error

        self.missing_samples = []

        self.found_output_file_name = found_output_file_name  # output file name if there was a file which does contain all experiments

        if self.out_path.endswith("/"):
            self.out_path = self.out_path[:-1]

        if self.found_output_file_name is None:
            file = str(self.extractor) + "_extraction_" + str(self.RunID) + ".csv"
            self.found_output_file_name = self.out_path + "/" + file

        os.makedirs(self.subject_dir + "/", exist_ok=True)

        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.out_path + "RPTK_missing_experiment_detection_" + self.extractor + "_" + self.RunID + ".err",
                logger_topic="RPTK Missing Detection Error"
            ).generate_log()

        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.out_path + "RPTK_missing_experiment_detection_" + self.extractor + "_" + self.RunID + ".log",
                logger_topic="RPTK Missing Detection"
            ).generate_log()

    def check_outfile_naming_and_read(self, done_exps_path_list: list):
        """
        Check if the outfile is named correctly and read it.
        :parameter
        """

        done_experiments = pd.DataFrame()
        failed = pd.DataFrame()
        file = str(self.extractor) + "_extraction_" + str(self.RunID) + ".csv"

        for exp in done_exps_path_list:
            exp_base = os.path.basename(exp)
            # exclude memory usage profile from collection of outfiles
            if "Memory_usage_profile" not in exp_base:
                # get the first file which matches the pattern
                if exp_base == file:
                    self.found_output_file_name = self.out_path + "/" + exp_base
                    self.logger.info("Found output file: " + str(exp_base))
                    done_experiments, failed_ = self.read_file(file=exp)
                    if done_experiments is not None:
                        break
                    else:
                        self.error.warning("Could not read file: " + exp_base)
                        print("Could not read file: " + exp_base)
                        done_experiments = pd.DataFrame()
                        continue
                    if failed_ is not None:
                        failed = failed_
                    else:
                        failed = pd.DataFrame()
                else:
                    self.error.warning(
                        "Outfile does not match configuration! Please check the outfile name " + exp_base)
        
        return done_experiments, failed

    def check_config_columns(self, done_experiments: pd.DataFrame):
        """
        Check if the config columns are in the outfile.
        :param done_experiments:
        :return:
        """
        # print(done_experiments)
        # 1. extract all columns which names include Unnamed.
        unnamed_cols = rptk.get_unnamed_cols(df=done_experiments)
        if len(unnamed_cols) > 0:
            self.logger.info("Found {} unnamed columns!".format(str(len(unnamed_cols))))

        # 2. include into df
        unnamed_done_experiments = done_experiments[unnamed_cols].copy()

        # 3. drop duplicated columns
        unnamed_done_experiments = rptk.drop_cols_with_same_values(df=unnamed_done_experiments)
        unnamed_cols_unique = rptk.get_unnamed_cols(df=unnamed_done_experiments)

        # 4. search for column with config string and drop the others
        config_cols = rptk.get_config_column_name(df=done_experiments, list_of_config_candidates=unnamed_cols_unique)

        drop_cols = [x for x in unnamed_cols if x not in config_cols]

        if len(drop_cols) > 0:
            self.logger.info("Dropping columns: " + str(drop_cols))
        else:
            self.logger.info("No columns to drop.")

        # if no config column is found, return dataframe without configuration
        if len(config_cols) > 0:
            self.logger.info("Found column containing config in " + self.out_path + " : " + str(config_cols))
            done_experiments = done_experiments.drop(drop_cols, axis=1)
            done_experiments = done_experiments.rename(columns={config_cols[0]: "config"})

        else:
            self.logger.info("No config column found.")
            done_experiments = done_experiments.drop(unnamed_cols, axis=1)

        return done_experiments

    def check_outfile_completeness(self):
        """
        Checks if the outfile already exists and contains all experiments which should be processed.
        :return: list with missing experiments IDs
        """
        
        # TODO start loader
        #loader = SpinningLoader(text="Checking Outfiles")
        #loader.start()

        done_experiments_id = []
        done_experiments = pd.DataFrame()
        failed = pd.DataFrame()

        # 1. detect outfile
        self.logger.info("Checking for done experiments in: " + str(self.out_path))
        print("Checking for done experiments in: " + str(self.out_path))

        done_exps_paths = glob.glob(self.out_path + "/*.csv")

        if len(done_exps_paths) > 0:
            # 2. read outfile
            done_experiments, failed = self.check_outfile_naming_and_read(done_exps_path_list=done_exps_paths)
            
            # 2.1 check format of done experiments and find config column
            done_experiments = self.check_config_columns(done_experiments=done_experiments)
            
            if done_experiments.empty:
                correct = [path for path in done_exps_paths if self.extractor in os.path.basename(path)]
                if len(correct) > 0:
                    # outfile is empty
                    self.error.warning("No entry in outfile! Could not find any done experiments in " + str(done_exps_paths))
                    print("No entry in outfile!")
                return done_experiments_id, done_experiments, failed
            else:
                # 3. check format of outfile
                self.logger.info("Found " + str(len(done_experiments)) + " done experiments.")
                print("Found " + str(len(done_experiments)) + " done experiments.")
                
                
                
                # Outfile_checker = Outfile_validator(df=done_experiments,
                #                                     extractor=self.extractor,
                #                                     logger=self.logger,
                #                                     runid=self.RunID,
                #                                     verbose=self.verbose,
                #                                     output_path=self.out_path)
                #
                # error = Outfile_checker.validate_outfile()

                # if error == 0:
                # 4. check for missing experiments
                if self.extractor == "MIRP":
                    if done_experiments.index.name == 'id_subject':
                        done_experiments = self.check_config_columns(done_experiments=done_experiments)
                        done_experiments_id = done_experiments.index.tolist()
                    else:
                        done_experiments = self.check_config_columns(done_experiments=done_experiments)
                        # get ids from done experiments
                        done_experiments_id = done_experiments["id_subject"].tolist()
                    # missing_samples_id = self.get_missing_subjects(done_experiments=done_experiments_id,
                    #                                               all_experiments=self.all_experiments_id)
                elif self.extractor == "PyRadiomics":
                    done_experiments = self.check_config_columns(done_experiments=done_experiments)
                    # Make index from Image and Mask column as id
                    done_experiments = self.make_image_mask_index(df=done_experiments)
                    done_experiments_id = done_experiments.index.tolist()

                    # missing_samples_id = self.get_missing_subjects(done_experiments=done_experiments_id,
                    #                                               all_experiments=self.all_experiments_id)
                else:
                    self.logger.error("Extractor not supported!")
                    sys.exit()
                    
                
        else:
            self.logger.info("No outfile found in " + str(self.out_path))

        self.logger.info("Found " + str(len(done_experiments)) + " done experiments.")
        
        # loader.stop()
        
        return done_experiments_id, done_experiments, failed

        #             done_samples = done_experiments.index.tolist()
        #
        #         if "id_subject" in done_experiments.columns:
        #             done_experiments.index = done_experiments["id_subject"]
        #         elif "Image" in done_experiments.columns:
        #             done_experiments = self.make_image_mask_index(done_experiments)
        #
        #     # sync done experiments with config file
        #     if "Image" not in done_experiments.columns:
        #         self.logger.info("Image not in done experiments!")
        #         # get all related columns for the experiments which are done
        #         done_experiments = pd.concat([self.df, done_experiments], axis=1)
        #         # drop all experiments (rows) which are having nan values (not done yet)
        #         done_experiments = done_experiments.dropna()
        #
        #         done_experiments = self.make_image_mask_index(done_experiments)
        #
        #     for i, row in done_experiments.iterrows():
        #         id_ = os.path.basename(row["Image"])[:-len(".nii.gz")] + "_" + os.path.basename(row["Mask"])[
        #                                                                        :-len(".nii.gz")]
        #         done_samples.append(id_)
        #
        #     if self.extractor == "MIRP":
        #         missing_samples = self.get_missing_experiments(done_samples, self.experiments)
        #     if self.extractor == "PyRadiomics":
        #         missing_samples = self.get_missing_experiments(done_samples, self.df)
        #
        # else:
        #     self.logger.info("No outfile found.")
        #
        # return missing_samples, done_samples

    def get_results_from_subject_folder(self, csv_paths, done_samples):
        """
        Get the results from the subject folder and add them to the outfile.
        :param csv_paths: Path to csv files in subject folder
        :param done_samples: List of done experiments which are already in the outfile
        :return: DataFrame with samples which are done but not in the outfile
        """

        self.logger.info("Found " + str(len(csv_paths)) + " done experiments in " + str(self.subject_dir) + ".")
        done = {"ID": [], "File_path": []}
        done_df = pd.DataFrame(data=done)

        for exp in csv_paths:
            if os.path.getsize(exp) > 0:
                if self.extractor == "MIRP":
                    done_exp = os.path.basename(exp)[:-len("_YEAR-MO-DA_features.csv")]
                else:
                    done_exp = os.path.basename(exp)[:-len(".csv")]

                if done_exp not in done_samples:
                    # self.logger.info("Found new done experiment in subjects folder: " + str(done_exp))
                    done_tmp = {"ID": [done_exp], "File_path": [exp]}
                    done_df = pd.concat([done_df, pd.DataFrame(data=done_tmp)], ignore_index=True)
            else:
                 print("Warning:",os.path.basename(exp),"is empty! Remove file from tmp results!")
                 self.logger.info(str(os.path.basename(exp)) + " is empty! Remove file from results!")
                 os.remove(exp)

        return done_df

    def get_done_experiments_from_subject_folder(self, done_samples, done_experiments):
        """
        Checks if the subject folder already exists and contains all experiments which should be processed.
        The Subject folder only contains MIRP results of each sample so far.
        :return: list: missing experiments
        """
        failed = pd.DataFrame()
        self.logger.info("Checking for done experiments in: " + str(self.subject_dir))

        if os.path.exists(self.subject_dir):
            # check if subject dir is empty
            if len(os.listdir(self.subject_dir)) > 0:
                csv_paths = glob.glob(self.subject_dir + "/*.csv")

                done_files_df = self.get_results_from_subject_folder(csv_paths=csv_paths, done_samples=done_samples)

                # if self.extractor == "MIRP":
                # if there are new done experiments in the subject folder
                if len(done_files_df) > 0:
                    for id in done_files_df["ID"].tolist():
                        done_samples.append(id)

                    self.logger.info("Concat " + str(len(done_files_df)) + " to outfile ...")
                    # concat all csv files to one df
                    df, failed = self.concat_extraction(csv_files=done_files_df["File_path"].tolist())
                    done_experiments = pd.concat([done_experiments, df], axis=0)
                    done_experiments = done_experiments.drop_duplicates()
        
        if len(failed) > 0:
            print("Found {} failed extracted samples.".format(str(len(failed))))
        # self.error.warning("Found {} failed extractions.".format(str(len(failed))))
        
        return done_experiments, done_samples, failed

        #     else:
        #         self.logger.info("Subject directory is empty.")
        #
        #         if self.extractor == "MIRP":
        #             if len(done_samples) > 0:
        #                 missing_samples = self.get_missing_subjects(done_samples, self.experiments)
        #             else:
        #                 self.logger.info("No experiments have been processed yet.")
        #                 missing_samples = self.experiments
        #         self.logger.info("Need to process " + str(len(missing_samples)) + " experiments.")
        #
        # else:
        #     self.logger.info("Subject directory does not exist.")
        #     if self.extractor == "MIRP":
        #         if len(done_samples) > 0:
        #             missing_samples = self.get_missing_subjects(done_samples, self.experiments)
        #         else:
        #             self.logger.info("No experiments have been processed yet.")
        #             missing_samples = self.experiments
        #     self.logger.info("Need to process " + str(len(missing_samples)) + " experiments.")

        # return missing_samples, done_samples

    def get_missing_subjects(self, done_experiments: list, all_experiments: list):
        """
        Returns a list of experiments that have already been processed.
        :param done_experiments: list of experiment IDs that have already been processed.
        :param all_experiments: list or pd.DataFrame of all experiments that should be processed.
        :return: experiments that have not already been processed.
        """

        missing_experiments_id = []

        for id_ in all_experiments:
            found = False
            for experiment in done_experiments:
                if id_ == experiment:
                    found = True
                    break
            # id not in done experiments
            if not found:
                missing_experiments_id.append(id_)

        return missing_experiments_id

        # if len(missing_experiments_id) > 0:
        #     self.logger.info("Found " + str(len(missing_experiments_id)) + " missing experiments.")
        #     for missing_exp in missing_experiments_id:
        #         if self.extractor == "MIRP":
        #             # for MIRP is all_experiments a list
        #             for experiment in all_experiments:
        #                 if missing_exp == experiment.subject:
        #                     missing_experiments.append(experiment)
        #                     break
        #         elif self.extractor == "PyRadiomics":
        #             # for PyRadiomics is all_experiments a pd.DataFrame
        #             # experiments = all_experiments["config"].tolist()
        #             experiments = []
        #             for i, row_ in all_experiments.iterrows():
        #                 experiments.append(row_["Image"][: -len(".nii.gz")] + "_" +
        #                                    os.path.basename(row_["Mask"])[:-len(".nii.gz")])
        #
        #             for experiment in experiments:
        #                 if missing_exp == experiment:
        #                     missing_experiments.append(experiment)
        #                     break

        # return missing_experiments

    def concat_extraction(self, csv_files: list):
        """
        Get all csv files from output folder and concat them into one csv file
        :param csv_files: list of csv files paths
        :return df: pd.DataFrame with concatinated csv dile from csv_files list
        """
        
        
        
        df = None
        failed_samples = pd.DataFrame()

        if not os.path.exists(self.found_output_file_name):
            df = pd.DataFrame()
        else:
            # if some samples are included but maybe new are in the output folder
            df = pd.read_csv(self.found_output_file_name, index_col=0)
            if len(csv_files) > len(df):
                if self.extractor == "PyRadiomics":
                    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Scanning for included feature files"):
                        csv_file = "PyRadiomics_extraction_" + str(
                            os.path.basename(row["Image"])[:-len(".nii.gz")]) + "_" + str(
                            os.path.basename(row["Mask"])[:-len(".nii.gz")]) + ".csv"
                        if csv_file in csv_files:
                            csv_files.remove(csv_file)

                if self.extractor == "MIRP":
                    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Scanning for included feature files"):
                        if "Image" in df.columns:
                            csv_file = str(
                                os.path.basename(row["Image"])[:-len(".nii.gz")]) + "_" + str(
                                os.path.basename(row["Mask"])[:-len(".nii.gz")])
                        elif "id_subject" in df.columns:
                            csv_file = str(row["id_subject"])
                        elif "id_subject" == df.index.name:
                            csv_file = str(i)
                        else:
                            self.error.error(
                                "Wring configuration of extraction file! Please check output of feature extractor as it is missing id_subject, Image, and Mask.")

                        for file in csv_files:
                            if file.startswith(csv_file):
                                csv_files.remove(csv_file)
                                break
                            
            elif len(csv_files) == len(df):
                csv_files = []
            else:
                self.error.warning(
                    "It seems like the extracted amount in the output file is higher than the processed amount in the output folder. Check you data!")
                csv_files = []

        if (df is not None) and (len(csv_files) > 0):
            print("Mem usage: %0.3f MB" % (psutil.Process().memory_info().rss / 1e6))
            print("Found {} done extractions in subject folder from previous run.".format(str(len(csv_files))))

            try:
                with Pool(processes=self.num_cpus) as pool:
                    # Use tqdm with imap to show progress
                    for result, failed in tqdm.tqdm(pool.imap(self.read_file, csv_files), total=len(csv_files), desc="Reading feature files"):
                        # get results together
                        if result is not None:
                            df = pd.concat([df, result], ignore_index=True)
                        # get failed extraction together
                        if failed is not None:
                            failed_samples = pd.concat([failed_samples, failed], ignore_index=True)
                            
                gc.collect()

                # with tqdm.tqdm(total=len(csv_files), desc='Reading feature files') as pbar:
                #     chunk_size = self.num_cpus  # Adjust the chunk size based on your memory constraints
                #     for start in range(0, len(csv_files), chunk_size):
                #         chunk = csv_files[start:start + chunk_size]
                #
                #         with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                #
                #             # for result in executor.map(self.read_file, [file for file in chunk], chunksize=chunk_size):
                #             #     pbar.update(1)
                #             #     if result is not None:
                #             #         df = pd.concat([df, result], ignore_index=True)
                #             #     gc.collect()
                #
                #             futures = {executor.submit(self.read_file, file): file for file in chunk}
                #             for future in concurrent.futures.as_completed(futures):
                #                 if future.result() is not None:
                #                     df = pd.concat([df, future.result()], ignore_index=True)
                #                 pbar.update(1)
                #                 gc.collect()  # Force garbage collection

                        # del result
                        # del chunk  # Free memory for the chunk
                        # del futures
                        # gc.collect()  # Force garbage collection

            except Exception as ex:
                self.error.error("Reading Feature Files Failed! " + str(ex))

            df.to_csv(self.found_output_file_name)
            
            print("Mem usage: %0.3f MB" % (psutil.Process().memory_info().rss / 1e6))

        else:
            df = pd.read_csv(self.found_output_file_name, index_col=0)
            # Process files using multiple CPUs
        # with Pool(processes=self.num_cpus) as pool:
        #     # Use tqdm with imap to show progress
        #     for result in tqdm.tqdm(pool.imap(self.read_file, csv_files),
        #                        total=len(csv_files), desc="Reading feature files"):
        #         if result is not None:
        #             df = pd.concat([df, result], ignore_index=True)

        if df.shape[0] == 0:
            self.logger.info("No features found in csv files.")
            return None
        else:
            self.logger.info("Concatenated feature space: " + str(df.shape[0]) + " Samples "
                             + str(df.shape[1]) + " Features.")
        
        
        # report features with nan values
        if df.isnull().any().any():
            self.error.warning("Number of Features with NaN: " + str(df.isnull().any().sum()))
            self.error.warning("Features with NaN: " + str(df.columns[df.isna().any()].tolist()))
            print("Number of Features with NaN: " + str(df.isnull().any().sum()))
            if df.isnull().any().sum() < 10:
                print("Features with NaN: " + str(df.columns[df.isna().any()].tolist()))
            
            # Get failed features and drop them
            config_features = ['Rater', 'Mask_Transformation', 'Image_Transformation', 'Raw_Image', 'Raw_Mask']
            
            # not all features contain nan --> better drop samples with nan
            if len(df.columns[df.isnull().any()]) != len(df.columns) :
                
                # failed extraction for some features
                remove = []
                for feature in df.columns[df.isnull().all()]:
                    if feature not in config_features:
                        self.error.warning("Failed extracting feature {} for all samples!".format(feature))
                        remove.append(feature)
                
                if len(remove) > 0:
                    print("Found {} failed extracting features for all samples!".format(str(len(remove))))
                    # drop failed features
                    df.drop(remove, axis = 1, inplace = True) 
                
                #remove = []
                #for feature in df.columns[df.isnull().any()]:
                    #if feature not in config_features:
                        #self.error.warning("Failed extracting feature {}".format(feature))
                        #remove.append(feature)
                
                #if len(remove) > 0:
                    #print("Found {} failed extracting features!".format(str(len(remove))))
                    #failed_df = df.copy()
                    #failed_df = failed_df[remove]
                    #failed_df.to_csv(self.out_path + "/failed_extraction.csv")
                    #drop failed features
                    #df.drop(remove, axis = 1, inplace = True) 
                

            if ("Image" in df.columns) and ("Mask" in df.columns):
                if df["Image"].isnull().any() or df["Mask"].isnull().any():
                    self.error.warning("{} Failed extraction. Dropping ...".format(df["Image"].isnull().any().sum()))
                    print("{} Failed extraction. Dropping ...".format(df["Image"].isnull().any().sum()))
                    df = df[df['Image'].notna()]
                    df = df[df['Mask'].notna()]

                # check if Image or Mask are not data path: - Format check
                for i, row in df.iterrows():
                    if ("/" not in row["Image"]) or ("/" not in row["Mask"]):
                        self.error.warning("{} Failed extraction. Dropping ...".format(row["Image"]))
                        print("{} Failed extraction. Dropping ...".format(row["Image"]))
                        df = df[df['Image'] != row["Image"]]
                        df = df[df['Mask'] != row["Mask"]]
        
        if len(failed_samples) > 0:
            print("Detected {} completely failed feature extractions".format(str(len(failed_samples))))
            self.error.warning("Detected {} completely failed feature extractions".format(str(len(failed_samples))))
        
        # for csv_file in tqdm(csv_files, desc='Reading csv files'):
        #     df_ = self.read_file(csv_file)

        # Outfile_checker = Outfile_validator(df=df,
        #                                     extractor=self.extractor,
        #                                     logger=self.logger,
        #                                     runid=self.RunID,
        #                                     verbose=0)
        #
        # error = Outfile_checker.validate_outfile()

        # if error == 0:
        #     df = pd.concat([df, df_])
        # else:
        #     self.logger.error("Error in csv file: " + csv_file)
        # drop all experiments (rows) which are having nan values (not done yet)
        # df = df.dropna()

        # df.to_csv(self.outfile)
        self.logger.info("Concatenated feature space: " + str(df.shape[0]) + " Samples "
                         + str(df.shape[1]) + " Features.")

        return df, failed_samples

    def make_image_mask_index(self, df: pd.DataFrame):
        """
        Creates a pd.Dataframe index from the image and mask filenames.
        :param df:
        :return: df with index of image and mask filenames
        """

        image = df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
        mask = df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

        id = []
        for x, y in zip(image, mask):
            id.append(x + "_" + y)

        ID = pd.Series(index=df.index, data=id)
        df.set_index(ID, inplace=True)

        return df

    def read_file(self, file: str):
        """
        Reads a csv file and returns a pd.DataFrame
        :param file: str
        :return: pd.DataFrame
        """
        try:

            df = pd.read_csv(file)
            if len(df.columns) <= 3:
                df = pd.read_csv(file, sep=";")
                
            if self.extractor == "PyRadiomics":
                
                if "diagnostics_Versions_PyRadiomics" in df.columns:
                    failed = len(df[df["diagnostics_Versions_PyRadiomics"].isnull()])
                    if failed > 0:
                        #print("Warning: Failed extraction detected! {}".format(str(file)))
                        self.error.warning("Failed extraction detected! {}".format(str(file)))
                        #print("Remove failed extraction file!")
                        os.remove(file)
                        return None, df
                elif "diagnostics_Mask-corrected_Size" in df.columns:
                    failed = len(df[df["diagnostics_Versions_PyRadiomics"].isnull()])
                    if failed > 0:
                        #print("Warning: Failed extraction detected! {}".format(str(file)))
                        self.error.warning("Failed extraction detected! {}".format(str(file)))
                        #print("Remove failed extraction file!")
                        os.remove(file)
                        return None, df
                else:
                    #print("Warning: Failed extraction detected! {}".format(str(file)))
                    self.error.warning("Failed extraction detected! {}".format(str(file)))
                    #print("Remove failed extraction file!")
                    os.remove(file)
                    return None, df
                        
        except Exception as e:
            self.logger.error("Error reading file: " + file + " " + str(e))
            return None, None

        return df, None

    def execute(self):
        """
        Main function for Detection of missing experiments 
        """
        self.logger.info("Searching for done experiments in outfiles ...")
        print("Searching for done experiments in outfiles ...")

        # 1. check for done experiments in outfile
        done_experiments_id, done_experiments, failed = self.check_outfile_completeness()

        # If missing_experiments_id is empty --> there was not outfile found
        if len(failed) > 0:
            print("Previous feature extraction failed!")
            self.error.warning("Previous feature extraction failed!")
        
        # 2. check for done experiments in subject folder if there are experiments missing
        if os.path.exists(self.subject_dir):
            # check if all experiments have been processed
            if len(done_experiments_id) != len(self.all_experiments_id):
                
                # Check if experiments have already been processed
                done_experiments, done_experiments_id, failed = self.get_done_experiments_from_subject_folder(
                    done_samples=done_experiments_id,
                    done_experiments=done_experiments)

        # calculate missing experiments
        self.missing_samples = self.get_missing_subjects(done_experiments_id, self.all_experiments_id)

        if len(self.missing_samples) > 0:
            self.logger.info("Detected " + str(len(self.missing_samples)) + " missing experiments")
            print("Detected " + str(len(self.missing_samples)) + " missing experiments")
        if len(done_experiments) > 0:
            self.logger.info("Detected " + str(len(done_experiments)) + " done experiments")
            print("Detected " + str(len(done_experiments)) + " done experiments")

        return self.missing_samples, done_experiments, failed
