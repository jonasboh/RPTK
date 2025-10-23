import os
import psutil

import numpy as np
import pandas as pd
from multiprocessing.pool import Pool
from functools import partial
import tqdm

import SimpleITK as sitk
from skimage.measure import label
from skimage.measure import regionprops
import skimage.morphology
from skimage.morphology import binary_dilation, binary_erosion, ball
from scipy.ndimage import label

from sklearn.metrics import jaccard_score
from sklearn.preprocessing import Normalizer
from skimage.segmentation import expand_labels

import argparse
import operator
import path
import time
import datetime
import logging
import glob
import re

from statistics import mean
from pathlib import Path  # handle path dirs
import random
import gzip
import shutil
import nibabel as nib
from tqdm.contrib.concurrent import process_map
import concurrent.futures
import tqdm
import signal

from threading import Semaphore

from rptk.mirp.experimentClass import ExperimentClass
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

from rptk.src.config.Log_generator_config import LogGenerator
# from rptk.src.Preprocessing.DataHandler import *

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    """Handler for the timeout signal."""
    raise TimeoutException("Function call timed out.") 

class PeriTumoralSegmentator:
    """
    Class for generating peritumoral regions from binary segmentations
    :parameter: seg_path: Path to segmentation
    :parameter n_cpu: Number of CPUs for parallel processing
    :parameter chunk_size: Chunk size for parallel processing
    :parameter: logger: Logger for log file
    :parameter: error: Error logger
    :parameter: output: Output path for peritumoral region
    :parameter: expand_seg_dist: Distance in mm for surrounding region expansion
    """

    def __init__(self,
                 seg_paths: list,
                 n_cpu: int = 1,
                 chunksize: int = 1,
                 logger=None,
                 error=None,
                 output: str = "",
                 expand_seg_dist: list = None,  # distance in mm for surrounding region expansion
                 timeout : int = 1000,
                 fast_mode: bool = False
                 ):

        # set default values for expand_seg_dist for 3 mm
        if expand_seg_dist is None:
            expand_seg_dist = [3]

        self.seg_paths = seg_paths
        self.n_cpu = n_cpu
        self.chunksize = chunksize
        self.logger = logger
        self.error = error
        self.output = output
        self.expand_seg_dist = expand_seg_dist
        self.timeout = timeout
        self.fast_mode = fast_mode
        
        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.output + "/PeriTumoralSegmentator_" + self.RunID + ".log",
                logger_topic="PeriTumoralSegmentator"
            ).generate_log()
        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.output + "/PeriTumoralSegmentator_" + self.RunID + ".err",
                logger_topic="PeriTumoralSegmentator error"
            ).generate_log()

    def get_peritumoral_region(self, seg_path, distance):
        """
        Extract surrounding region from segmentation
        :param int distance: Distance in mm for surrounding region expansion
        :return: array peritumoral region of size dist mm of binary segmentation with label 1
        """

        sitk_img = sitk.ReadImage(seg_path)
        seg = sitk.GetArrayFromImage(sitk_img)

        expanded = CCExtender(seg_paths=[seg_path],
                                                logger=self.logger,
                                                error=self.error).get_expanded_region(seg_path=seg_path,
                                                                      exp_distance=distance)
        if not expanded is  None:
            peritumoral = expanded - seg
            peritumoral = sitk.GetImageFromArray(peritumoral.astype(np.uint16))
            peritumoral.CopyInformation(sitk_img)
        else:
            peritumoral = sitk_img


        return peritumoral

    def process(self, seg_path):
        """
        Creates and writes peritumoral Seg to folder
        """

        processed_num = 0

        for dist in self.expand_seg_dist:
            peri_seg = self.get_peritumoral_region(seg_path=seg_path, distance=dist)
            peri_seg_name = os.path.basename(seg_path)[:-len(".nii.gz")] + "_" + str(dist) + "_peritumoral.nii.gz"

            if not os.path.isdir(self.output):
                os.makedirs(self.output, exist_ok=True)

            if np.sum(sitk.GetArrayFromImage(peri_seg)) <= 0:
                self.error.warning("Empty Peritumoral Segmentation: " + str(peri_seg_name))
            else:
                # self.logger.info("Generated Peritumoral Segmentation: " + str(peri_seg_name))
                processed_num += 1
                sitk.WriteImage(peri_seg,
                                os.path.join(self.output, peri_seg_name),
                                useCompression=True)


    def exe(self):
        """
        Peritumoral segmentation Multiprocessing function
        """
        self.logger.info("Generating {} Peritumoral Segmentations".format(str(len(self.seg_paths))))

        if self.fast_mode:
            with tqdm.tqdm(total=len(self.seg_paths), desc='Surrounding perturbation') as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                    for results in executor.map(self.process, self.seg_paths, chunksize=self.n_cpu):
                        pbar.update(1)
        else:
            for seg_path in tqdm.tqdm(self.seg_paths, total=len(self.seg_paths), desc="Surrounding perturbation"):
                self.process(seg_path)


class SupervoxelPerturbator:
    """
    Class for segmentation perturbation.
    Adapted from MIRP.
    """

    def __init__(self,
                 img_paths: list,
                 seg_paths: list,
                 modality: list,
                 output: str,
                 n_cpu: int = 1,
                 chunksize: int = 1,
                 repetition: int = 1,
                 distance=None,
                 crop_around_roi: bool = False,
                 perturbation_roi_adapt_type: str = "distance",
                 logger=None,
                 error=None,
                 timeout=100,  # Timeout in seconds
                 ):

        """
        :parameter img_paths: List of path to the image.
        :parameter seg_paths: List of path to the segmentation.
        :parameter distance: Distance in mm to grow or shrink the ROI.
        :parameter crop_around_roi: Crop the image around the ROI.
        :parameter perturbation_roi_adapt_type: Type of perturbation to apply to the ROI.
        :parameter modality: List of Modalities of the images.
        :parameter repetition: Number of repetitions for the perturbation.
        :parameter output: Path to write the perturbed images.
        :parameter n_cpu: Number of CPUs for parallel processing.
        :parameter chunksize: Chunk size for parallel processing.
        :parameter logger: Logger for log file.
        :parameter error: Error handler.
        :parameter timeout: Timeout for processing in seconds.
        """

        # Default distance to grow or shrink the ROI
        if distance is None:
            distance = [-1.0, 0.0, 1.0]

        self.distance = distance
        self.crop_around_roi = crop_around_roi
        self.perturbation_roi_adapt_type = perturbation_roi_adapt_type
        self.modality = modality
        self.n_cpu = n_cpu
        self.chunksize = chunksize
        self.repetition = repetition
        self.output = output
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.logger = logger
        self.error = error
        self.timeout = timeout

        # To store experiments
        self.experiments = []
        
        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.output + "/SupervoxelPerturbator_" + self.RunID + ".log",
                logger_topic="SupervoxelPerturbator"
            ).generate_log()
            
        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.output + "/SupervoxelPerturbator_" + self.RunID + ".err",
                logger_topic="SupervoxelPerturbator error"
            ).generate_log()
    
    def preprocess_and_save_image(self, img_path):
        """
        Load an image, check for NaN values, correct them, and save the corrected image.

        Parameters
        ----------
        img_path : str
            Path to the input NIfTI image file.
        logger : Logger
            Logger for warnings and error messages.

        Returns
        -------
        None
        """
        # Load the NIfTI image data
        img = nib.load(img_path)
        img_data = img.get_fdata()

        # Check for NaN values
        if np.isnan(img_data).any():
            self.error.warning(f"Image contains NaN values! {os.path.basename(img_path)} Correction performed!")
            print(f"Image contains NaN values! {os.path.basename(img_path)} Correction performed!")

            # Replace NaN values with 0 (or another appropriate value)
            np.nan_to_num(img_data, copy=False)

            # Save the corrected image back to disk
            #corrected_img = nib.Nifti1Image(img_data, img.affine, img.header)
            #write_nifti_file(array=img_data,
            #                    out_file_name=img,
            #                    path=img,
            #                    logger=self.error)

            self.logger.info(f"Corrected image saved to {img_path}")
        else:
            self.logger.info(f"No NaN values found in {os.path.basename(img_path)}. No correction needed.")

    def create_settings(self, perturbation_settings: ImagePerturbationSettingsClass, modality: str):
        """
        Set default settings for generating response maps and computing feature values.
        """

        general_settings = GeneralSettingsClass(by_slice=True)

        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=True,
            interpolate=False,
            anti_aliasing=False
        )

        if modality == "CT":
            resegmentation_settings = ResegmentationSettingsClass(resegmentation_method="threshold",
                                                                  resegmentation_intensity_range=[-1000.0, -900.0])
        elif modality == "MR":
            resegmentation_settings = ResegmentationSettingsClass(resegmentation_method="range",
                                                                  resegmentation_sigma=3.0)
        else:
            self.error.error("Image perturbation setting not defined. Modality not supported!")
            raise ValueError("Image perturbation setting not defined. Modality not supported!")

        feature_computation_parameters = FeatureExtractionSettingsClass(
            by_slice=True,
            no_approximation=True,
            base_feature_families="none",
        )

        image_transformation_settings = ImageTransformationSettingsClass(
            by_slice=True,
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
            roi_resegment_settings=resegmentation_settings,
            perturbation_settings=perturbation_settings,
            img_transform_settings=image_transformation_settings,
            feature_extr_settings=feature_computation_parameters,
        )

        return settings

    def generate_experiments(self, perturbation_settings: ImagePerturbationSettingsClass):
        """"
        Generate experiment object from MIRP.
        """

        experiments = []

        for img_path, seg_path, modality in zip(self.img_paths, self.seg_paths, self.modality):
            # create Settings
            settings = self.create_settings(perturbation_settings=perturbation_settings, modality=modality)

            # create Experiments
            experiment_ = ExperimentClass(
                modality=modality,
                subject=os.path.basename(img_path)[:-len(".nii.gz")],
                cohort=None,
                write_path=self.output,
                image_folder=os.path.dirname(img_path),
                roi_folder=os.path.dirname(seg_path),
                roi_reg_img_folder=None,
                image_file_name_pattern=os.path.basename(img_path)[:-len(".nii.gz")],
                registration_image_file_name_pattern=None,
                roi_names=[os.path.basename(seg_path)[:-len(".nii.gz")]],
                data_str=[],
                provide_diagnostics=False,
                settings=settings,
                compute_features=False,
                extract_images=True,
                plot_images=False,
                keep_images_in_memory=False
            )
            experiments.append(experiment_)

        return experiments

    def process_experiment(self, experiment):
        # Set the signal handler for the alarm signal.
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # Start the alarm.
        signal.alarm(self.timeout)
        
        try:
            experiment.process()
            # Cancel the alarm if function completes on time.
            signal.alarm(0)
            return True
        except TimeoutException as e:
            # self.logger.error("Error in processing " + str(experiment.subject) + " " + str(e))
            self.error.warning("Failed processing " + str(experiment.subject) + ": " + str(e))
            self.error.warning(f"The function call for sample {experiment.subject} exceeded the timeout of {self.timeout} sec and was terminated. You can try to increase the timeout. ")
            print(f"The function call for sample {experiment.subject} exceeded the timeout of {self.timeout} sec and was terminated. You can try to increase the timeout.")
            return False

        finally:
            # Ensure the alarm is always canceled.
            signal.alarm(0)

    def exe(self):
        """
        Generate perturbation for segmentation growth and shrinkage.
        """

        # check for NaN values in segmentations
        # for seg_path in self.seg_paths:
        #    seg = nib.load(seg_path).get_fdata()
        #    if np.isnan(seg).any():
        #        self.logger.warning("Segmentation contains NaN values! " + str(os.path.basename(seg_path))
        #                            + " Correction performed!")
        #        np.nan_to_num(seg, copy=False)
        #        write_nifti_file(array=seg,
        #                         out_file_name=seg_path,
        #                         path=seg_path,
        #                         logger=self.logger)

        # for img_path in self.img_paths:
        # seg = nib.load(img_path).get_fdata()
        # if np.isnan(seg).any():
        #    self.logger.warning("Segmentation contains NaN values! " + str(os.path.basename(img_path))
        #                        + " Correction performed!")
        #    np.nan_to_num(seg, copy=False)
        #    write_nifti_file(array=seg,
        #                     out_file_name=img_path,
        #                     path=img_path,
        #                     logger=self.logger)

        # check for NaN values in images
        #for img_path in tqdm.tqdm(self.img_paths, desc="Checking for NaN values in images"):
        #    self.preprocess_and_save_image(img_path)

        perturbation_settings = ImagePerturbationSettingsClass(
            crop_around_roi=self.crop_around_roi,
            perturbation_roi_adapt_type=self.perturbation_roi_adapt_type,
            perturbation_roi_adapt_size=self.distance,
            perturbation_randomise_roi_repetitions=self.repetition
        )

        # Set up the experiment.
        experiments_ = self.generate_experiments(perturbation_settings=perturbation_settings)
        for experiment in experiments_:
            self.experiments.append(experiment)

        if len(self.experiments) != 0:
            # Only use multi threading when list ist longer
            if len(self.experiments) > 1:

                # for experiment in self.experiments:
                #     try:
                #         experiment.process()
                #     except Exception as e:
                #         # self.logger.error("Error in processing " + str(experiment.subject) + " " + str(e))
                #         self.logger.warning("Error in processing " + str(experiment.subject) + " " + str(e))

                # p = Pool(self.n_cpu)

                # p.map(self.process_, self.experiments, chunksize=self.chunksize)
                # p.close()
                # p.join()

                #with tqdm.tqdm(total=len(self.experiments), desc="Supervoxel Randomization") as pbar:
                #    with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                #        for _ in executor.map(self.process_experiment, self.experiments, chunksize=self.n_cpu):
                #            pbar.update(1)

                # Create a multiprocessing pool
                pool = Pool(self.n_cpu)

                # Apply the process function to each item in the list
                with tqdm.tqdm(total=len(self.experiments), desc="Supervoxel Randomization") as pbar:
                    for _ in pool.imap_unordered(self.process_experiment, self.experiments):
                        pbar.update(1)

                # Close the pool
                pool.close()
                pool.join()

            else:
                try:
                    self.experiments[0].process()
                except Exception as e:
                    # self.logger.error("Error in processing " + str(self.experiments[0].subject) + " " + str(e))
                    self.error.warning("Failed processing " + str(self.experiments[0].subject) + " " + str(e))
                # exe(self.experiments[0])
        else:
            self.error.warning("No segmentation to process! Check your segmentation files!")


class RandomWalker:
    """
    Class to perform random walker-based segmentation perturbations.
    
    Attributes:
        parameter (list): list of segmentation paths to process
        iterations (int): Number of iterations for perturbation.
        dilation_iterations (int): Number of dilation steps applied in perturbation.
        erosion_iterations (int): Number of erosion steps applied in perturbation.
        noise_variance (float): Noise level applied to segmentation.
        extreme_perturbation (bool): If True, applies more aggressive perturbations.
        dice_threshold (tuple): Acceptable range for Dice coefficient (lower, upper bound).
        perturbation_factor (float): Scaling factor for perturbation based on segmentation size.
        rng (np.random.Generator): Random number generator.
        output (str): Directory for saving perturbed segmentations.
        max_iterations (int): Define the maximum of iteration until dice criteria fits.
        timeout (int): Define the max time the process should take in sec.
    """

    def __init__(self, 
                 logger: logging.Logger,
                 error: logging.Logger,
                 segs: list,
                 output: str,
                 RunID: str = None,
                 n_cpu: int = 1,
                 chunksize: int = 1,
                 random_walker_iterations: int = 3, 
                 seed: int = 1234, 
                 dilation_iterations: int = 2, 
                 erosion_iterations: int = 2, 
                 noise_variance: float = 0.02, 
                 extreme_perturbation: bool = False,
                 dice_threshold: tuple = (0.90, 0.98), 
                 perturbation_factor: float = 0.05,
                 max_iterations:int = 100,
                 timeout:int = 500,
                 fast_mode: bool = True):
        
        self.logger = logger
        self.error = error
        self.segs = segs
        self.output = output
        self.RunID = RunID
        self.n_cpu = n_cpu
        self.chunksize = chunksize
        self.random_walker_iterations = random_walker_iterations
        self.dilation_iterations = dilation_iterations
        self.erosion_iterations = erosion_iterations
        self.noise_variance = noise_variance
        self.extreme_perturbation = extreme_perturbation
        self.dice_threshold = dice_threshold
        self.perturbation_factor = perturbation_factor  # Scales perturbation by segmentation size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.fast_mode = fast_mode
        
        
        # Parameters to control processing
        self.max_iterations = max_iterations  # Maximum loop cycles
        self.timeout = timeout  # Maximum allowed time in seconds
        self.iteration = 0  # Iteration counter

        if not os.path.exists(self.output):
            os.makedirs(self.output)
            
        if not self.output.endswith("/"):
            self.output += "/"
        
        if self.RunID is None:
            self.RunID = time.strftime("%Y%m%d-%H%M%S")

        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.output + "/RandomWalker_" + self.RunID + ".log",
                logger_topic="RandomWalker"
            ).generate_log()
            
        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.output + "/RandomWalker_" + self.RunID + ".err",
                logger_topic="RandomWalker error"
            ).generate_log()
        
    def _apply_random_change(self, segmentation: np.ndarray, original_seg: np.ndarray) -> np.ndarray:
        """
        Apply perturbation based on the selected mode, with normalization for small segmentations.
        
        Args:
            segmentation (np.ndarray): The segmentation mask to perturb.
            original_seg (np.ndarray): The original segmentation mask.
        
        Returns:
            np.ndarray: The perturbed segmentation mask.
        """
        seg_size = original_seg.sum()
        num_changes = max(1, int(seg_size * self.perturbation_factor))  # Scale perturbations based on size
        
        if self.extreme_perturbation:
            for _ in range(self.dilation_iterations):
                segmentation = binary_dilation(segmentation, ball(1))
            for _ in range(self.erosion_iterations):
                segmentation = binary_erosion(segmentation, ball(1))
        
        # Generate change mask
        change_mask = segmentation ^ binary_dilation(segmentation)
        
        # Randomly decide which pixels to add or remove
        change_indices = np.argwhere(change_mask)
        if len(change_indices) > 0:
            selected_indices = self.rng.choice(len(change_indices), size=min(num_changes, len(change_indices)), replace=False)
            for idx in selected_indices:
                segmentation[tuple(change_indices[idx])] = 1 - segmentation[tuple(change_indices[idx])]
        
        return segmentation
    
    def _compute_dice(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute Dice coefficient between original and perturbed segmentations.
        
        Args:
            original (np.ndarray): Original segmentation mask.
            perturbed (np.ndarray): Perturbed segmentation mask.
        
        Returns:
            float: Dice coefficient.
        """
        intersection = np.logical_and(original, perturbed).sum()
        return 2.0 * intersection / (original.sum() + perturbed.sum())
    
    def _ensure_single_component(self, segmentation: np.ndarray) -> bool:
        """
        Ensure the segmentation remains a single connected component.
        
        Args:
            segmentation (np.ndarray): The segmentation mask.
        
        Returns:
            bool: True if segmentation has one connected component, False otherwise.
        """
        labeled_array, num_features = label(segmentation)
        return num_features == 1

    def segmentation_convex_hull(self, seg_path: str):
        """
        Apply convex hull operation to refine segmentation.
        
        Args:
            seg_path (str): Path to the segmentation file.
        
        Returns:
            np.ndarray: The refined segmentation mask.
        """
        seg_ = sitk.ReadImage(seg_path)
        nda = sitk.GetArrayFromImage(seg_)
        
        convex_nda = np.ndarray(nda.shape)
        for i in range(nda.shape[0]):
            bnda = skimage.morphology.convex_hull_object(nda[i])
            convex_nda[i] = bnda
        
        sitk_img_out = sitk.GetImageFromArray(convex_nda.astype(np.uint16))
        sitk_img_out.CopyInformation(seg_)
        
        convex_dice_score = self._compute_dice(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)), convex_nda)

        self.logger.info(f"Convex hull Dice score for {os.path.basename(seg_path)}: {convex_dice_score:.2f}")

        output_filename = seg_path.replace(".nii.gz", "_convex.nii.gz")
        # output_filename = re.sub(r"dice_\d+\.\d+_", "", output_filename, count=1)
        output_filename = re.sub(r"dice_\d+\.\d+_", "", output_filename)

        sitk.WriteImage(sitk_img_out, output_filename, useCompression=True)

    def segmentation_closing(self, seg_path: str, radius: int):
        """
        Apply morphological closing to refine segmentation and compute Dice score.
        
        Args:
            seg_path (str): Path to the segmentation file.
            radius (int): Radius for morphological closing.
        """
        seg_ = sitk.ReadImage(seg_path)
        nda = sitk.GetArrayFromImage(seg_)
        
        closed_nda = np.ndarray(nda.shape)
        for i in range(nda.shape[0]):
            bnda = skimage.morphology.closing(nda[i], skimage.morphology.disk(radius))
            closed_nda[i] = bnda
        
        sitk_img_out = sitk.GetImageFromArray(closed_nda.astype(np.uint16))
        sitk_img_out.CopyInformation(seg_)
        
        closed_dice_score = self._compute_dice(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)), closed_nda)
        
        self.logger.info(f"Segmentation Closing Dice score for {os.path.basename(seg_path)}: {closed_dice_score:.2f}")

        output_filename = seg_path.replace(".nii.gz", "_closed.nii.gz")
        # output_filename = re.sub(r"dice_\d+\.\d+_", "", output_filename, count=1)
        output_filename = re.sub(r"dice_\d+\.\d+_", "", output_filename)
        
        sitk.WriteImage(sitk_img_out, output_filename, useCompression=True)
        
    def process(self, seg_path: str = None, pbar = None):
        """
        Apply random walker perturbation to a binary segmentation loaded from a file.
        
        Args:
            seg_path (str): Path to the input segmentation file (.nii.gz format).
            pbar : Process bar for tracking progress.
        """
        
        if seg_path is None:
            self.error.error("No segmentation path given! Please select correct input.")
            raise ValueError("No segmentation path given! Please select correct input.")
        else:
            if not os.path.exists(seg_path):
                self.error.error(f"Segmentation file {seg_path} not found! Please check the file path.")
                raise FileNotFoundError(f"Segmentation file {seg_path} not found! Please check the file path.")
        
        seg_nifti = nib.load(seg_path)
        segmentation = seg_nifti.get_fdata().astype(np.uint8)
        perturbed_seg = segmentation.copy()

        self.iteration = 0
        self.start_time = time.time()

        dice_score = 1.0

        for _ in range(self.random_walker_iterations):

            dice_score = 1.0
            # simulate random walker perturbation as long as the dice score is not in the desired range
            while not (self.dice_threshold[0] <= dice_score <= self.dice_threshold[1]):
                perturbed_seg = segmentation.copy()

                try:
                    perturbed_seg = self._apply_random_change(perturbed_seg, segmentation)
                except Exception as e:
                    print("Failed perturbation: " + e)

                # if we stay with the seed we will only have n time the same segmentation
                self.seed +=1
                self.rng = np.random.default_rng(self.seed)

                if self._ensure_single_component(perturbed_seg):

                    try:
                        dice_score = self._compute_dice(segmentation, perturbed_seg)
                    except Exception as e:
                        print("Failed dice calculation: " + e)
                        self.error.warning("Failed dice calculation: " + e)

                    # Save perturbed segmentation
                    perturbed_nifti = nib.Nifti1Image(perturbed_seg, affine=seg_nifti.affine, header=seg_nifti.header)
                    output_filename = os.path.basename(seg_path).replace(".nii.gz", f"_random_walker_{self.iteration}.nii.gz")
                    perturbed_path = os.path.join(self.output if self.output else os.path.dirname(seg_path), output_filename)
                    nib.save(perturbed_nifti, perturbed_path)

                    # Apply segmentation corrections
                    #self.segmentation_closing(perturbed_path, radius=2)
                    #self.segmentation_convex_hull(perturbed_path)

                    # 1. Stop if max iterations are reached
                    if self.iteration >= self.max_iterations:
                        self.logger.info(f"Max iterations {str(max_iterations)} reached. Stopping perturbation ...")
                        break

                    # 2. Stop if the loop runs for too long
                    if time.time() - self.start_time > self.timeout:
                        self.error.warning(f"Timeout of {str(self.timeout)} sec reached. Stopping perturbation ...")
                        break
            self.logger.info("Dice score for " + os.path.basename(seg_path) + " after " + str(self.iteration) + " iterations: " + str(dice_score))
            self.iteration += 1

            if pbar is not None:
                pbar.update(1)
           
    
    def exe(self):
        """
        Execute random segmentation change in multiprocessing
        """

        self.logger.info("Starting Random Walker Perturbation processing ...")
        if self.fast_mode:
            pool = Pool(self.n_cpu)

            # Apply the process function to each item in the list
            with tqdm.tqdm(total=len(self.segs), desc="Random Walker perturbation") as pbar:
                for _ in pool.imap_unordered(self.process, 
                                            self.segs,
                                            chunksize=self.n_cpu):
                    pbar.update(1)

            pool.close()
            pool.join()
        else:
            for seg_path in tqdm.tqdm(self.segs, total=len(self.segs), desc="Random Walker perturbation"):
                self.process(seg_path)


class CCExtender:
    """
    Connected Component Extender for segmentation perturbation
    :param seg_paths: list of path to segmentation
    :param output: output path
    :param expand_seg_dist: list of distances to expand the segmentations
    :param logger: logger
    :param error: error logger
    :param n_cpu: number of cpus to use
    :param chunksize: chunksize for multiprocessing
    :param timeout: timeout for processing
    """

    def __init__(self,
                 seg_paths: list,
                 output: str = None,
                 expand_distance: list = None,
                 logger: logging.Logger = None,
                 error: logging.Logger = None,
                 n_cpu: int = 1,
                 chunksize: int = 1,
                 timeout: int = 500,
                 ):

        # Default settings for expand_seg_dist to 3 mm around the segmentation
        if expand_distance is None:
            expand_distance = [3]

        self.seg_paths = seg_paths
        self.output = output
        self.expand_distance = expand_distance
        self.logger = logger
        self.error = error
        self.n_cpu = n_cpu
        self.chunksize = chunksize
        self.timeout = timeout
        
        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.output + "/CCExtender_" + self.RunID + ".log",
                logger_topic="CCExtender"
            ).generate_log()
            
        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.output + "/CCExtender_" + self.RunID + ".err",
                logger_topic="CCExtender error"
            ).generate_log()

    def get_expanded_region(self, exp_distance, seg_path=None):
        """
        Get expanded region from segmentation (only the ROI with label 1!)
        :return: expanded region of size dist mm of binary segmentation with label 1
        """
        

        if seg_path is None:
            self.error.error("No segmentation path given. Using self.seg_path")
            print("No segmentation path given. Using self.seg_path")

        sitk_img = sitk.ReadImage(seg_path)
        seg = sitk.GetArrayFromImage(sitk_img)

        labels = np.unique(seg)

        # if there is any other label than 1: set it to 0
        if len(labels) > 2:
            self.error.warning("There are more than 2 labels in the segmentation. Only label 1 will be expanded!")
            # set all labels which are not 1 to 0
            seg[seg != 1] = 0

        # Set the signal handler for the alarm signal.
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # Start the alarm.
        signal.alarm(self.timeout)

        try:
            # Call the function.
            expanded = expand_labels(seg, distance=exp_distance)
            # Cancel the alarm if function completes on time.
            signal.alarm(0)
            return expanded

        except TimeoutException:
            # Handle timeout: return None if the function call times out.
            self.error.warning(f"The function call for sample {seg_path} exceeded the timeout of {self.timeout} sec and was terminated. You can try to increase the timeout. ")
            print(f"The function call for sample {seg_path} exceeded the timeout of {self.timeout} sec and was terminated. You can try to increase the timeout.")
            return None

        finally:
            # Ensure the alarm is always canceled.
            signal.alarm(0)
        

        return expanded

    def process(self, seg_path):
        """
        Writes expanded Seg to folder
        """

        sitk_img = sitk.ReadImage(seg_path)
        self.logger.info(f"Expanding Connected Components for {os.path.basename(seg_path)} ...")
        # print(f"Expanding Connected Components for {os.path.basename(seg_path)} ...")

        for dist in self.expand_distance:
            
            expanded_seg = self.get_expanded_region(exp_distance=dist, seg_path=seg_path)
            exp_seg_name = os.path.basename(seg_path)[:-7] + "_" + str(dist) + "_expanded.nii.gz"

            if not os.path.isdir(self.output):
                os.makedirs(self.output, exist_ok=True)

            if not expanded_seg is  None:
                expanded_seg = sitk.GetImageFromArray(expanded_seg.astype(np.uint16))
                expanded_seg.CopyInformation(sitk_img)

                if np.sum(sitk.GetArrayFromImage(expanded_seg)) <= 0:
                    self.logger.warning("Empty Expanded Segmentation:", exp_seg_name)
                else:
                    sitk.WriteImage(expanded_seg,
                                    os.path.join(self.output, exp_seg_name),
                                    useCompression=True)

    def exe(self):
        """
        Multiprocessing function
        """

        #process_map(self.process, self.seg_paths, max_workers=self.n_cpu, chunksize=self.chunksize,
        #            desc='Connected Component Expansion')


        #with tqdm.tqdm(total=len(self.seg_paths), desc="Connected Component Expansion") as pbar:
        #    with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
        #        for _ in executor.map(self.process, self.seg_paths, chunksize=self.n_cpu):
        #            pbar.update(1)

        #for seg_path in tqdm.tqdm(self.seg_paths, total=len(self.seg_paths), desc="Connected Component Expansion"):
        #    self.process(seg_path)
        #for seg_path in self.seg_paths:
        #    self.process(seg_path)

        p = Pool(self.n_cpu)
        with tqdm.tqdm(total=len(self.seg_paths), desc='Connected Component Expansion') as pbar:
            for _ in p.imap(self.process, self.seg_paths, chunksize=self.chunksize):
                pbar.update(1)
        p.close()
        p.join()


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
            logger.warning("Need to implement multiple filenames for multiple images")
    else:
        sitk_img_out = sitk.GetImageFromArray(array.astype(np.uint16))
        sitk_img_out.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_img_out, out_file_name, useCompression=True)


# TODO: Integrate class for re-segmentation
class ReSegmentator:
    def __init__(self):
        pass
