# from Segmentation_pertubator import Segmentation_pertubator
# import mirp_image_transformer as mit
# import Image_transformer as it

from typing import Union, List

import os
import logging
from tqdm import *
import pandas as pd
import multiprocessing
import nibabel as nib
from threading import Thread
from multiprocessing import Pool
import os.path
import glob
import concurrent.futures
import gc

import sys

#sys.path.insert(0, 'src')
import rptk.src.image_processing.Image_transformer as it
from rptk.src.config.Log_generator_config import LogGenerator


class Executor:
    """
    Handle image transformations and segmentation perturbations from Pyradimics and MIRP and output
    transformed and resampled images
    """

    def __init__(self,
                 rptk_config_json: str = None,  # path to rptk config json file
                 input_csv: str = None,  # path to csv file for Image and Mask paths
                 kernels: Union[List, str] = None,  # list of kernels to apply
                 n_cpu: int = 1,  # number of cpus to use
                 pyradiomics_yaml_file: str = None,  # path to pyradiomics yaml file
                 modality: str = None,  # modality of image
                 seg_perturbation: bool = False,
                 repetition: int = 3,
                 distance: Union[None, List[float], float] = 0.0,
                 crop_around_roi: bool = False,
                 perturbation_roi_adapt_type: str = "distance",
                 output_dir: str = "",
                 chunksize: int = 10,  # chunksize for multiprocessing
                 logger=None,  # logging output
                 error=None,  # error output
                 to_process: list = None,  # dict with format kernel: [img_paths]
                 use_previous_output: bool = False,  # if True, use previous output from output_dir
                 RunID: str = None,
                 config_file_path: str = None,
                 fast_mode: bool = True,
                ):

        self.kernels = kernels
        self.n_cpu = n_cpu
        self.pyradiomics_yaml_file = pyradiomics_yaml_file
        self.input_csv = input_csv
        self.modality = modality
        self.seg_perturbation = seg_perturbation
        self.repetition = repetition
        self.distance = distance
        self.crop_around_roi = crop_around_roi
        self.perturbation_roi_adapt_type = perturbation_roi_adapt_type
        self.output_dir = output_dir
        self.chunksize = chunksize
        self.logger = logger
        self.error = error
        self.to_process = to_process  # dict (format = {kernel: [img_paths]}) with data to process if get data from previous run
        self.use_previous_output = use_previous_output
        self.RunID = RunID
        self.rptk_config_json = rptk_config_json
        self.fast_mode = fast_mode


        self.kernel_patterns = ["wavelet-", "square.", "squareroot", "_logarithm", "_exponential", "_gradient",
                           "_lbp-2D", "_lbp-3D", "_laws_", "gabor", "gauss", "separable_wavelet", "mean", "log_", "log-"]

        self.kernel_in_config = ["Wavelet", "Square.", "SquareRoot", "Logarithm", "Exponential", "Gradient",
                                "LBP2D", "LBP3D", "laws", "gabor", "gaussian", "separable_wavelet", "mean", "laplacian_of_gaussian",
                                "LoG"]

        # how kernels are written in files:
        self.kernels_in_files = {"Wavelet": "wavelet-",
                                 "SquareRoot": "squareroot",
                                 "Square": "square.",
                                 "Logarithm": "_logarithm",
                                 "Exponential": "_exponential",
                                 "Gradient": "_gradient",
                                 "LBP2D": "_lbp-2D",
                                 "LBP3D": "_lbp-3D",
                                 "laws": "_laws_",
                                 "gabor": "gabor",
                                 "gaussian": "gauss",
                                  #"nonseparable_wavelet": "_wavelet_",  # Maybe not correct
                                 "separable_wavelet": "separable_wavelet",  # Maybe not correct
                                 "mean": "mean",
                                 "laplacian_of_gaussian": "log_",
                                 "LoG": "log-"}

        self.pyradiomics_kernels = ["Wavelet",
                                    "Square",
                                    "LoG",
                                    "SquareRoot",
                                    "Logarithm",
                                    "Exponential",
                                    "Gradient",
                                    "LBP2D",
                                    "LBP3D"]

        self.mirp_kernels = ["laws",
                             "gabor",
                             "gaussian",
                             # "nonseparable_wavelet",
                             "separable_wavelet",
                             "mean",
                             "laplacian_of_gaussian"]

        self.error_num = 0

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # if self.logger is None:
        self.logger = LogGenerator(
            log_file_name=self.output_dir + "RPTK_transformation_" + self.RunID + ".log",
            logger_topic="RPTK Transformation"
        ).generate_log()

        # if self.error is None:
        self.error = LogGenerator(
            log_file_name=self.output_dir + "RPTK_transformation_" + self.RunID + ".err",
            logger_topic="RPTK Transformation error"
        ).generate_log()

        self.count_to_process = 0

    def configurate(self):

        if self.input_csv is not None:
            self.df = pd.read_csv(self.input_csv)
        else:
            self.error.error("No input csv file given! Please provide input_csv parameter!")
            raise ValueError("No input csv file given! Please provide input_csv parameter!")

        if len(os.listdir(self.output_dir)) > 0:
            IDs = set(self.df.loc[:, "ID"].tolist())
            o_files = [os.path.basename(f) for f in os.listdir(self.output_dir) if os.path.isfile(os.path.join(self.output_dir, f))]

            # if processed samples are not given, look into output file and see which samples are already processed
            # if self.to_process is None:
            #     self.to_process = {}
            #
            #     # for each kernel, get all files that are already processed
            #     for kernel in self.kernels:
            #
            #         # look at the ID in the beginning of the file name
            #         for ID in IDs:
            #             # fill the to_process dict with the files that are not processed yet
            #             to_process_files = self.df.loc[self.df["ID"] == ID, "Image"].tolist()
            #             self.to_process[kernel] = to_process_files
            #
            #             # find files which are already processed in output folder
            #             for o_file in o_files:
            #                 # check kernel and ID of file to match to input files
            #                 if self.kernels_in_files[kernel] in o_file and o_file.startswith(ID):
            #
            #                     # find input files and remove them from to_process list
            #                     for i_file in to_process_files:
            #                         if os.path.basename(i_file).startswith(ID):
            #                             if i_file in self.to_process[kernel]:
            #                                 self.to_process[kernel].remove(i_file)
            #                 else:
            #                     break
            # else:
            #     # check if there are already processed files in the output folder to reduce the number of files to process
            #     for kernel in self.kernels:
            #
            #         # look at the ID in the beginning of the file name
            #         for ID in IDs:
            #             # fill the to_process dict with the files that are not processed yet
            #             to_process_files = self.df.loc[self.df["ID"] == ID, "Image"].tolist()
            #
            #             # find files which are already processed in output folder
            #             for o_file in o_files:
            #
            #                 # check kernel and ID of file to match to input files
            #                 if self.kernels_in_files[kernel] in o_file and o_file.startswith(ID):
            #
            #                     # find input files and remove them from to_process list
            #                     for i_file in to_process_files:
            #                         if os.path.basename(i_file).startswith(ID):
            #                             if kernel in self.to_process.keys():
            #                                 if i_file in self.to_process[kernel]:
            #                                     self.to_process[kernel].remove(i_file)
            #                 #else:
            #                 #    break

            self.count_to_process = 0
            for i in self.to_process.values():
                self.count_to_process += len(i)
            if self.use_previous_output and self.count_to_process > 0:
                print("Need to perform image transformation.")

        self.logger.info("### Transformation config:" +
                         "\n\t\t\t\t\t\t\t\t\t\tKernels: " + str(self.kernels) +
                         "\n\t\t\t\t\t\t\t\t\t\tCSV input file path: " + str(self.input_csv) +
                         "\n\t\t\t\t\t\t\t\t\t\tCPUs: " + str(self.n_cpu)
                        )

    def get_pyradiomics(self, kernel, img_path, seg_path):

        if self.pyradiomics_yaml_file is None:

            if kernel in self.kernel_patterns:
                for t_kernel in self.kernels_in_files.keys():
                    if kernel == self.kernels_in_files[t_kernel]:
                        kernel = t_kernel

            # get kernel from user input
            if kernel not in self.pyradiomics_kernels:
                self.logger.info("Kernel {} not in pyradiomics setting!".format(kernel))
                transformer = None
            else:
                # Use config from RPTK if no yaml file is given
                transformer = it.Pyradiomics_image_transformer(rptk_config_json=self.rptk_config_json,
                                                               kernels=[kernel],
                                                               img_path=img_path,
                                                               seg_path=seg_path,
                                                               output_dir=self.output_dir,
                                                               logger=self.logger)
        else:
            # get kernels from yaml file
            transformer = it.Pyradiomics_image_transformer(yaml_file=self.pyradiomics_yaml_file,
                                                           img_path=img_path,
                                                           seg_path=seg_path,
                                                           output_dir=self.output_dir,
                                                           logger=self.logger)
        return transformer

    def get_mirp(self, kernel, img_path, seg_path):

        if kernel in self.kernel_patterns:
            for t_kernel in self.kernels_in_files.keys():
                if kernel == self.kernels_in_files[t_kernel]:
                    kernel = t_kernel

        if kernel not in self.mirp_kernels:
            self.logger.info("Kernel {} not in mirp setting!".format(kernel))
            self.logger.info("Only perturbation of segmentation happening!")

            # Perform only pertubation
            transformer = it.MIRP_image_and_seg_pertubator(kernels=None,
                                                           img_path=img_path,
                                                           seg_path=seg_path,
                                                           modality=self.modality,
                                                           output_dir=self.output_dir,
                                                           seg_pertubation=self.seg_perturbation,
                                                           repetition=self.repetition,
                                                           distance=self.distance,
                                                           crop_around_roi=self.crop_around_roi,
                                                           perturbation_roi_adapt_type=self.perturbation_roi_adapt_type,
                                                           logger=self.logger)

            experiment = transformer.generate_experiments()
        else:
            transformer = it.MIRP_image_and_seg_pertubator(kernels=kernel,
                                                           img_path=img_path,
                                                           seg_path=seg_path,
                                                           modality=self.modality,
                                                           output_dir=self.output_dir,
                                                           seg_pertubation=self.seg_perturbation,
                                                           repetition=self.repetition,
                                                           distance=self.distance,
                                                           crop_around_roi=self.crop_around_roi,
                                                           perturbation_roi_adapt_type=self.perturbation_roi_adapt_type,
                                                           logger=self.logger)

            experiment = transformer.generate_experiments()

        return experiment

    def process(self, experiment):
        if experiment is not None:
            try:
                # self.logger.info("Executing " + str(experiment.roi_names[0]))
                experiment.process()
                return True
            except Exception as ex:
                self.error.error("Failed executing experiment. {}".format(ex))
                raise ValueError(ex)
        else:
            self.error_num += 1
        return None

    def check_for_multiple_kernels_in_file(self, img_path: str, ):

        # check if input file name is not already containing a kernel
        found_kernel = 0
        for kernel in self.kernel_patterns:
            if kernel in img_path:
                num = str(os.path.basename(img_path)).count(kernel)
                found_kernel += num

        return found_kernel

    def generate_experiments(self, experiments:dict, kernel:str, img_path:str, seg_path:str):
        """
        Generate experiments for Image Transformation
        :param experiments: dict for experiments
        :param kernel: kernel to use
        :param img_path: path to image
        :param seg_path: path to segmentation
        :return: dict with experiments
        """

        if kernel in self.pyradiomics_kernels:
            self.count_to_process +=1
            experiment = self.get_pyradiomics(kernel=kernel,
                                                img_path=img_path,
                                                seg_path=seg_path)
                
            if not experiment is None:
                if kernel in experiments.keys():
                    experiments[kernel].append(experiment)
                else:
                    experiments[kernel] = [experiment]

        elif kernel in self.mirp_kernels:
            self.count_to_process +=1
            experiment = self.get_mirp(kernel=kernel,
                                        img_path=img_path,
                                        seg_path=seg_path)

            if kernel in experiments.keys():
                experiments[kernel].append(experiment)
            else:
                experiments[kernel] = [experiment]
        else:
            print("Kernel {} not in PyRadiomics or MIRP setting! Skipping transformation!".format(kernel))
            self.error.warning("Kernel {} not in PyRadiomics or MIRP setting!".format(kernel))
            return None

        return experiments

    def get_experiments(self, df=pd.DataFrame()):
        """
        Generate experiments for Image Transformation
        :param df:
        :return:
        """

        experiments = {}

        if len(df) == 0:
            df = pd.read_csv(self.input_csv)

        self.logger.info('Number of Images: ' + str(len(set(df["Image"].values))))
        self.logger.info("Gathering settings for transformation and perturbations ...")

        skipped_transformations = 0
        transformed_imgs = []
        transformed_samples = {}

        for img_path in list(set(df["Image"])):
            seg_path = df.loc[df["Image"] == img_path, "Mask"].values[0]  # does not matter as it is only about the image
            # self.logger.info("Configuration of " + str(os.path.basename(img_path)))

            ID = df.loc[df["Image"] == img_path, "ID"].values[0]
            SampleIO = df.loc[df["ID"] == ID]

            # Get all transformations from this sample from the csv file
            if "Image_Transformation" in SampleIO.columns:
                if len(SampleIO.loc[~SampleIO["Image_Transformation"].isna()]) > 0:
                    for ker in list(set(SampleIO.loc[~SampleIO["Image_Transformation"].isna(), "Image_Transformation"])):
                        # check if transformations for this ID has been performed already
                        if ker not in transformed_samples.keys():
                            transformed_samples[ker] = list(SampleIO.loc[SampleIO["Image_Transformation"] == ker, "ID"].values)
                        else:
                            transformed_samples[ker] += list(SampleIO.loc[SampleIO["Image_Transformation"] == ker, "ID"].values)
                #else:
                #    self.error.warning("Sample {} has NaN values in input csv".format(str(set(SampleIO["ID"].to_list()))))
                #    print("Sample {} has NaN values in input csv".format(str(set(SampleIO["ID"].to_list()))))
            else:
                self.error.warning("Can not find Image_Transformation in input csv for executing Transformation! Check your input file")

            # if modality is not defined
            if self.modality == None:
                if "Modality" in df.columns:
                    self.modality = df.loc[df["Image"] == img_path, "Modality"].values[0]
                else:
                    self.error.error("Modality not defined! Please either add a column to the csv for Modality or define it for transformation!")
                    raise ValueError("Modality not defined! Please either add a column to the csv for Modality or define it for transformation!")

            if self.kernels is not None:
                # only execute if this has not been done before (images are duplicated in the csv)
                if img_path not in transformed_imgs:
                    for kernel in self.kernels:

                        # if it has already been processed
                        if self.count_to_process != 0 and self.use_previous_output:
                            if kernel in self.to_process.keys():

                                if not kernel in self.kernels_in_files.keys():
                                    self.error.error("Kernel {} is not supported! Please select one of the following: {}".format(kernel, str(self.kernels_in_files.keys())))
                                    raise ValueError("Kernel {} is not supported! Please select one of the following: {}".format(kernel, str(self.kernels_in_files.keys())))

                                if self.kernels_in_files[kernel] in os.path.basename(img_path):  # img_path not in self.find_kernel_in_file[kernel]:
                                    if kernel not in transformed_samples.keys():
                                        transformed_samples[kernel] = list(df.loc[df["Image"] == img_path, "ID"].values)
                                    else:
                                        transformed_samples[kernel].append(
                                            df.loc[df["Image"] == img_path, "ID"].values[0])

                                    skipped_transformations += 1
                                    # self.logger.info("Skipping transformation of " + str(os.path.basename(img_path)) + " for kernel " + str(kernel) + "!")
                                    continue

                        elif self.count_to_process == 0 and self.use_previous_output:
                            # self.logger.info("All transformations have been processed already!")
                            continue

                        found_kernel = self.check_for_multiple_kernels_in_file(img_path)

                        if found_kernel == 1:
                            skipped_transformations += 1

                            if kernel not in transformed_samples.keys():
                                transformed_samples[kernel] = [df.loc[df["Image"] == img_path, "ID"].values[0]]
                            else:
                                transformed_samples[kernel].append(df.loc[df["Image"] == img_path, "ID"].values[0])

                            continue

                        elif found_kernel > 1:
                            self.error.warning("Deleting file with double transformation: " + img_path)
                            # os.remove(img_path)
                            continue

                        if kernel in transformed_samples.keys():
                            # check if output for kernel and ID are already there
                            if ID in transformed_samples[kernel]:
                                continue

                        experiments_ = self.generate_experiments(experiments, kernel, img_path, seg_path)

                        if experiments_ is None:
                            continue
                        else:
                            experiments = experiments_

                    transformed_imgs.append(img_path)
                else:
                    self.logger.info("Skipping already transformaed image {}".format(img_path))

            else:
                self.kernels = self.kernels_in_files.keys()

                found_kernel = self.check_for_multiple_kernels_in_file(img_path)

                if found_kernel == 1:
                    skipped_transformations += 1
                    continue
                elif found_kernel > 1:
                    self.error.warning("Deleting file with double transformation: " + img_path)
                    # os.remove(img_path)
                    continue
                else:
                    # no transformation
                    experiment = self.get_mirp(kernel=self.kernels,
                                               img_path=img_path,
                                               seg_path=seg_path)

                    experiments[None] = [experiment]

        if self.count_to_process == 0 and self.use_previous_output:
            self.logger.info("All transformations have been processed already!")
            print("All transformations have been processed already!")

        if skipped_transformations > 0:
            self.logger.info("Skipped {} transformations based on previous results!".format(skipped_transformations))
            print("Skipped {} transformations based on previous results!".format(skipped_transformations))

        return experiments

    def perform_transformation(self, experiments):

        for kernel in experiments.keys():
            self.logger.info("Executing " + str(kernel))
            self.logger.info("Number of experiments: " + str(len(experiments[kernel])))
            print("Executing " + str(kernel) + " Transformation")
            print("Number of experiments: " + str(len(experiments[kernel])))
            
            #try:
            #    with tqdm(total=len(experiments[kernel]), desc="Performing " + kernel + " Transformation") as pbar:
            #        chunk_size = self.n_cpu  # Adjust the chunk size based on your memory constraints
            #        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            #            for start in range(0, len(entries), chunk_size):
            #                chunk = experiments[kernel][start:start + chunk_size]
            #                futures = {executor.submit(self.process, row): row for row in chunk}
            #                for future in concurrent.futures.as_completed(futures):
            #                    if future is None:
            #                        self.error.warning("Failed Image Transformation {}.".format(kernel))
            #                        print("Failed Image Transformation {}.".format(kernel))
            #                        gc.collect()
            #            del futures
            #                        
            #except Exception as ex:
            #    self.error.error("Performing " + kernel + " Transformation Failed! " + str(ex))
            if self.fast_mode:
                try:
                    #with tqdm(total=len(experiments[kernel]), desc="Performing " + kernel + " Transformation") as pbar:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                        
                        result = list(tqdm(executor.map(self.process, experiments[kernel]),
                                        total=len(experiments[kernel]),
                                        desc="Performing " + kernel + " Transformation"))
                
                        for future in result:
                            if future is None:
                                self.error.warning("Failed Image Transformation {}.".format(kernel))
                                print("Failed Image Transformation {}.".format(kernel))
                                for experiment in experiments[kernel]:
                                    try:
                                        self.process(experiment)
                                    except Exception as ex:
                                        self.error.error("Failed Image Transformation {}: {}".format(kernel, ex))
                                        print("Failed Image Transformation {}: {}".format(kernel, ex))
                                        gc.collect()
                                        continue
                            
                                gc.collect()  # Force garbage collection
                
                        # del chunk  # Free memory for the chunk
                        gc.collect()  # Force garbage collection
                        del result
                
                except Exception as ex:
                    self.error.error("Performing " + kernel + " Transformation Failed! " + str(ex))
                    print("Performing " + kernel + " Transformation Failed! " + str(ex), "Trying to execute it in slow mode ...")

                    for experiment in experiments[kernel]:
                        try:
                            self.process(experiment)
                        except Exception as ex:
                            self.error.error("Failed Image Transformation {}: {}".format(kernel, ex))
                            print("Failed Image Transformation {}: {}".format(kernel, ex))
                            gc.collect()
                            continue
            else:
                for experiment in tqdm(experiments[kernel], total=len(experiments[kernel]), desc="Performing " + kernel + " Transformation"):
                    try:
                        self.process(experiment)
                    except Exception as ex:
                        self.error.error("Failed Image Transformation {}: {}".format(kernel, ex))
                        print("Failed Image Transformation {}: {}".format(kernel, ex))
                        gc.collect()
                        continue

    def run(self):
        """
        Running needed transformations
        """
        self.configurate()

        # Check what images need to process which transormation
        # 1. if all transformations in inout csv are in output folder
        # 2. check and remove not valid transformations (double transformations or transformation in config)
        # 3. check which transformation from the perforemd transformations are missing with the current setting and which are not included


        experiments = self.get_experiments()

        self.logger.info("Executing transformations and perturbations ...")

        self.perform_transformation(experiments)

        if self.error_num > 0:
            self.error.warning("Number of errors: {}".format(str(self.error_num)))

        self.logger.info("### Finished with Image transformation! ###")
