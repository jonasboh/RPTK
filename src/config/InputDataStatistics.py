import os
import nibabel as nib
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from multiprocessing import Pool as mp_pool
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from skimage import measure
from pathlib import Path
from functools import partial
import time
import concurrent.futures
import radiomics
from radiomics import *

import pyvista as pv
import pywt

from rptk.src.config.Log_generator_config import LogGenerator
# from rptk.src.feature_filtering.Radiomics_Filter_exe import RadiomicsFilter
from ydata_profiling import ProfileReport
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Optional
from queue import Queue
from threading import Thread
import requests
import sys
import traceback

class Pool:
    def __init__(self, work, max_concurrent_jobs, max_worker: int = 32) -> None:
        self.max_workers = max_worker
        self.work_queue = Queue(max_concurrent_jobs)
        self.out_queue = Queue()
        self.is_running = True

        def _work():
            while self.is_running:
                item = self.work_queue.get()
                result = work(item)
                self.work_queue.task_done()
                self.out_queue.put(result)

        for _ in range(max_worker):
            Thread(target=_work).start()

    def close(self):
        self.is_running = False

class DataStatGenerator:
    """
    Generates statistics of number of Slices, Slice thickness distribution and distribution of the size of the ROIs
    :param input_path: path to csv input file
    :param data: Data in pd.DataFrame format
    :param out_path: Path to output file
    :param logger: logger for logging the process
    :param error: error logger for logging the errors and warnings
    :param logs_dir: directory for logs
    :param num_cpus: number of CPUs to use
    :param RunID:  RunID for registering the run with the configuration
    :param extract_radiomics_fingerprint: Extract radiomics fingerprint
    :param bin_width: bin width for slice thickness distribution
    :param generate_profile: Generate Pandas profile for parameter summary
    :param profile_format: Format of Pandas profile ("html" or "json")
    """

    def __init__(self,
                 input_path: str,  # path to csv input file
                 data: pd.DataFrame = None,  # Data in pd.DataFrame format
                 out_path: str = None,  # output file
                 logger=None,  # logger for logging the process
                 error=None,  # error logger for logging the errors and warnings
                 logs_dir: str = None,  # directory for logs
                 num_cpus: int = 1,  # number of CPUs to use
                 RunID: str = None,  # RunID
                 extract_radiomics_fingerprint: bool = False,  # Extract radiomics fingerprint
                 bin_width: float = 25.0,  # bin width for slice thickness distribution
                 generate_profile: bool = True,
                 profile_format: str = "html",
                 verbose: int = 0,
                 prediction_label: str = None
                 ):

        self.input_path = input_path
        self.data = data
        self.out_path = out_path
        self.logger = logger
        self.error = error
        self.logs_dir = logs_dir
        self.num_cpus = num_cpus
        self.RunID = RunID
        self.extract_radiomics_fingerprint = extract_radiomics_fingerprint
        self.bin_width = bin_width
        self.generate_profile = generate_profile
        self.profile_format = profile_format
        self.verbose = verbose
        self.prediction_label = prediction_label

        self.radiomics_fingerprint_yaml = os.path.dirname(__file__) + "/PyRadiomics/Data_fingerprint.yaml"

        if self.prediction_label is None:
            self.prediction_label = "Prediction_Label"

        if self.RunID is None:
            self.error.error("RunID is not provided.")
            raise ValueError("RunID is not provided.")

        os.makedirs(self.out_path, exist_ok=True)

        # Config Logger #
        if self.logger is None:
            if self.logs_dir is None:
                self.logger = LogGenerator(
                    log_file_name=self.out_path + "/RPTK_data_statistics_" + self.RunID + ".log",
                    logger_topic="RPTK Data Statistics"
                ).generate_log()

                self.error = LogGenerator(
                    log_file_name=self.out_path + "/RPTK_data_statistics_" + self.RunID + ".err",
                    logger_topic="RPTK Data Statistics Error"
                ).generate_log()
            else:
                self.logger = LogGenerator(
                    log_file_name=self.logs_dir + "/RPTK_data_statistics_" + self.RunID + ".log",
                    logger_topic="RPTK Data Statistics"
                ).generate_log()

                self.error = LogGenerator(
                    log_file_name=self.logs_dir + "/RPTK_data_statistics_" + self.RunID + ".err",
                    logger_topic="RPTK Data Statistics Error"
                ).generate_log()

        if self.input_path is not None:
            if self.data is None:
                self.logger.info("Reading input file: " + self.input_path)
                self.data = pd.read_csv(self.input_path)
            else:
                self.logger.info("Get data from provided data frame.")
        else:
            if self.data is None:
                self.error.error("Input path is not provided.")
                raise ValueError("Input path is not provided.")
            else:
                self.logger.info("Get data from provided data frame.")

        if self.out_path is None:
            self.error.error("Output path is not provided.")
            raise ValueError("Output path is not provided.")

    def count_connected_components(self, mask: np.ndarray):
        """
        Count the number of connected components inside a mask.
        :param mask: Binary mask array where 1 represents foreground and 0 represents background.
        :return int: Number of connected components.
        """
        try:
            # Label connected components in the mask
            labeled_mask = measure.label(mask)

            # Count the number of unique labels (connected components)
            num_connected_components = np.max(labeled_mask)

            del labeled_mask

        except:
            self.error.error("Could not extract number of connected components.")
            raise ValueError("Could not extract number of connected components.")

        return num_connected_components

    def get_radiomics_fingerprint(self, experiment: pd.Series):
        """
        Extract radiomics first order and radiomics shape features from the input data to optimize the feature computation process.
        :param experiment: Data for feature computation
        :return:
        """

        self.logger.info("Extracting radiomics fingerprint!")
        exp = pd.DataFrame([experiment])
        
        try:
            py_result = self.py_extractor.execute(imageFilepath=str(experiment["Image"]),
                                                  maskFilepath=str(experiment["Mask"]))

            py_formated = pd.DataFrame([py_result])
            py_formated.index = exp.index

            # Add experiment configuration to the features
            result = pd.concat([exp, py_formated], axis=1)
            
            return result

        except Exception as ex:
            self.error.warning("Radiomics fingerprint feature extraction failed for sample {}! ".format(experiment["ID"]) + str(ex))
            print("Radiomics fingerprint feature extraction failed for sample {}! ".format(experiment["ID"]) + str(ex))
            
            return None
            # raise RuntimeError("Radiomics fingerprint feature extraction failed! " + str(ex))

            # if self.verbose == 0:
            #     try:
            #         os.system("pyradiomics " + os.path.abspath(self.input_path) + " -o " + os.path.abspath(outfile) + " -f csv -v 2 -j " +
            #                   str(self.num_cpus) + " --param " + os.path.abspath(self.radiomics_fingerprint_yaml) + " &> " + self.out_path +
            #                   "/radiomics_fingerprint_extraction.log")
            #         self.pyradiomics_wait_for_output(outfile=outfile)
            #     except Exception as e:
            #         self.error.error("Could not extract radiomics fingerprint.{}".format(str(e)))
            #         raise ValueError("Could not extract radiomics fingerprint.{}".format(str(e)))
            #
            # else:
            #     try:
            #         os.system("pyradiomics " + os.path.abspath(self.input_path) + " -o " + os.path.abspath(outfile) + " -f csv -j " +
            #                   str(self.num_cpus) + " --param " + os.path.abspath(
            #             self.radiomics_fingerprint_yaml) + " &> " + self.out_path +
            #                   "/radiomics_fingerprint_extraction.log")
            #     except Exception as e:
            #         self.error.error("Could not extract radiomics fingerprint.{}".format(str(e)))
            #         raise ValueError("Could not extract radiomics fingerprint.{}".format(str(e)))
        #else:
        #    self.logger.info("Radiomics fingerprint has already been extracted!")

        # self.logger.info("Radiomics fingerprint extraction done!")
        

    def pyradiomics_wait_for_output(self, outfile: str):
        """
        Wait until extraction is finished
        :param outfile:
        :return:
        """
        if not os.path.exists(outfile):
            time.sleep(1)
            # size = os.path.getsize(outfile)
        while (not os.path.exists(outfile)) and (not os.path.getsize(outfile) > 0):
            time.sleep(10)

        if os.path.isfile(outfile):
            time.sleep(10)
            pyradiomics_output = pd.read_csv(outfile)
            self.logger.info("PyRadiomics extracted {} features for {} samples".format(
                str(len(pyradiomics_output.columns)), str(len(pyradiomics_output))))
        else:
            raise ValueError("%s isn't a file!" % outfile)

    def extract_connected_component_stat(self, file_paths: list = None):
        """
        Get information from NIfTI files in a directory based on certain criteria.
        :param file_paths: List of Paths to NIfTI files.Error processing file
        :return nifti_infos: A list of tuples, each containing the number of slices, slice thickness, and file path.
        """

        nifti_infos = []

        # Recursively search for .nii.gz files
        if file_paths is None:
            if len(self.data["Mask"]) == 0:
                self.error.error(
                    "File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
                raise ValueError(
                    "File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
            else:
                file_paths = self.data["Mask"].tolist()

        # read and convert to numpy array
        array_list = []
        for file_path in file_paths:
            img = nib.load(file_path)
            array_list.append(np.array(img.dataobj))

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.count_connected_components, array_list),
                               total=len(array_list), desc="Extract Number of Components"):
                if result is not None:
                    nifti_infos.append(result)
            del result
            del array_list

        return nifti_infos

    def get_number_of_foreground_labels_in_mask(self, mask: np.ndarray):
        """
        Get the number of unique labels in a mask to estimate how many targets were segmented.
        :param mask: Mask array where values 0> represents foreground and 0 represents background.
        :return num_unique: number of foreground labels.
        """
        try:
            unique_numbers = np.unique(mask)
            num_unique = len(unique_numbers) - 1
        except:
            self.error.error("Could not extract number of foreground labels.")
            raise ValueError("Could not extract number of foreground labels.")

        return num_unique
    
    def get_label_and_component(self, file_path: str):
        """
        Count the number of connected components inside a mask. AND 
        Get the number of unique labels in a mask to estimate how many targets were segmented.
        :param file_path: File path to file where values 0> represents foreground and 0 represents background.
        :return cc_count, label_count: Number of connected components, number of foreground labels.
        """

        img = nib.load(file_path)

        cc_count = self.count_connected_components(mask=np.array(img.dataobj))
        label_count = self.get_number_of_foreground_labels_in_mask(mask=np.array(img.dataobj))

        del img

        return cc_count, label_count

    def extract_foreground_label_stat(self, file_paths: list = None):
        """
        Get information from NIfTI files in a directory based on certain criteria.
        :param file_paths: List of Paths to NIfTI files.
        :return nifti_infos: A list of tuples, each containing the number of slices, slice thickness, and file path.
        """

        nifti_infos = []

        # Recursively search for .nii.gz files
        if file_paths is None:
            if len(self.data["Mask"]) == 0:
                self.error.error("File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
                raise ValueError("File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
            else:
                file_paths = self.data["Mask"].tolist()

        # read and convert to numpy array
        array_list = []
        for file_path in file_paths:
            img = nib.load(file_path)
            array_list.append(np.array(img.dataobj))

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.get_number_of_foreground_labels_in_mask, array_list), total=len(array_list), desc="Extract Number of ROI labels"):
                if result is not None:
                    nifti_infos.append(result)

        return nifti_infos
    
    def extract_cc_count_and_foreground_label(self, file_paths: list = None):
        """
        Get information from NIfTI files in a directory based on certain criteria.
        :param file_paths: List of Paths to NIfTI files.
        :return nifti_infos: A list of tuples, each containing the number of slices, slice thickness, and file path.
        """

        nifti_infos = []

        # Recursively search for .nii.gz files
        if file_paths is None:
            if len(self.data["Mask"]) == 0:
                self.error.error("File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
                raise ValueError("File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
            else:
                file_paths = self.data["Mask"].tolist()

        # Define a per-task timeout in seconds (adjust as needed)
        task_timeout = 30  
        
        with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
            # Submit tasks to the executor for each file path
            futures = {executor.submit(self.get_label_and_component, fp): fp for fp in file_paths}
            
            # Using as_completed to handle futures as they complete, with tqdm for progress
            for future in tqdm(as_completed(futures), total=len(futures),
                            desc="Extract ROI labels and Connected Components"):
                try:
                    result = future.result(timeout=task_timeout)
                    # Check if result is valid
                    if result[0] is not None and result[1] is not None:
                        nifti_infos.append(result)
                except Exception as e:
                    # Handle exceptions (e.g., timeouts, other errors) if needed
                    print(f"Task for {futures[future]} failed: {e}")
                    self.error.error(f"Task for {futures[future]} failed: {e}")

        # Process files using multiple CPUs
        #with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
        #    for result in tqdm(pool.imap(self.get_label_and_component, file_paths),
        #                       total=len(file_paths),
        #                       desc="Extract ROI labels and Connected Components"):
        #        if result[0] is not None and result[1] is not None:
        #            nifti_infos.append(result)
        #    del result
            # del array_list

        return nifti_infos
    
    def extract_slice_statistics(self, file_path: str):
        """
        Process a NIfTI file.
        :param file_path (str): Path to the NIfTI file.
        :return tuple: A tuple containing the number of slices, slice thickness, and file path.
        """

        try:
            nii_img = nib.load(file_path)
            header = nii_img.header
            num_slices = header.get_data_shape()[2]
            slice_thickness = header.get_zooms()[2]

            del header
            del nii_img

            return num_slices, slice_thickness, file_path
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            print(traceback.format_exc())
            return None, None, None

    def get_image_slice_stat(self, file_paths: list = None):
        """
        Get information from NIfTI files in a directory based on certain criteria.
        :param file_paths: List of Paths to NIfTI files.
        :return nifti_infos: A list of tuples, each containing the number of slices, slice thickness, and file path.
        """

        nifti_infos = []

        # Recursively search for .nii.gz files
        if file_paths is None:
            if len(self.data["Image"]) == 0:
                self.error.error("File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
                raise ValueError("File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
            else:
                file_paths = self.data["Image"].tolist()

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.extract_slice_statistics, file_paths), total=len(file_paths), desc="Get nifti info"):
                if result[0] is not None and result[1] is not None:
                    nifti_infos.append(result)

        return nifti_infos


    def get_roi_size(self, file_path: str):
        """
        Get the size of the ROI in a NIfTI file.
        :param file_path (str): Path to the NIfTI file.
        :return tuple: A tuple containing the ROI size and file path.
        """

        try:
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            roi_size = data.sum()

            # free memory
            del data
            del nii_img

            return roi_size, file_path

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            print(traceback.format_exc())
            return None, None

    def calculate_bins(self, image_path, segmentation_path, bin_width):
        """
        Calculate the number of bins for an ROI in an image given a fixed bin width.

        Parameters:
        - image_path: str, path to the .nii.gz file of the image.
        - segmentation_path: str, path to the .nii.gz file of the segmentation.
        - bin_width: float, width of each bin.

        Returns:
        - int, the number of bins.
        """
        try:
            # Load the image and the segmentation mask
            image = nib.load(image_path).get_fdata()
            segmentation = nib.load(segmentation_path).get_fdata()

            # Extract ROI values from the image based on the segmentation
            roi_values = image[segmentation > 0]

            # Calculate the range of ROI values
            min_val, max_val = np.min(roi_values), np.max(roi_values)

            # Calculate the number of bins
            num_bins = np.ceil((max_val - min_val) / bin_width).astype(int)

        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            print(traceback.format_exc())
            return False, os.path.basename(image_path)


        return True, num_bins


    def get_bins(self, row, bin_width):
        """
        Worker function to be called in parallel.
        """

        bins = self.calculate_bins(image_path=row["Image"],
                                   segmentation_path=row["Mask"],
                                   bin_width=bin_width)

        return bins

    def get_default_number_of_bins(self, csv_file=None, bin_width=0):
        """
        Get the default number of bins for each ROI in a directory.
        :parameter
        """

        bin_number = []

        if csv_file is not None:
            # Read CSV file
            df = pd.read_csv(csv_file)
        else:
            df = self.data

        # Use a partial function to fix bin_width argument
        get_bins_partial = partial(self.get_bins, bin_width=bin_width)

        # Apply function in parallel
        # results = tqdm.tqdm(pool.map(get_bins_partial, [row for _, row in df.iterrows()]),
        #                     total=len(df),
        #                     desc="Calculating Bins")

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(get_bins_partial, [row for _, row in df.iterrows()]), total=len(df), desc="Calculating Bins"):
                if (result[0]) and (not isinstance(result[1], str)):
                    bin_number.append(result[1])
                else:
                    self.error.warning(f"Calculating number of bins failed for {result[1]}!")
                    print(f"Calculating number of bins failed for {result[1]}!")

        return bin_number

    def compute_surface_wavelet(self, image, mask):
        """
        Compute a surface wavelet feature by sampling image intensities at the mask surface,
        applying a 1D wavelet decomposition, and returning a summary statistic.
        
        Parameters:
        image: 3D numpy array (image intensities)
        mask: 3D numpy array (binary mask)
        
        Returns:
        wavelet_feature: A scalar summary of the wavelet detail coefficients.
        """
        
        try:
            # Extract the surface of the mask
            verts, _, _, _ = measure.marching_cubes(mask, level=0.5)
            # Round vertex coordinates to nearest voxel indices
            verts_int = np.round(verts).astype(int)
            # Clip coordinates to be within the image bounds
            dims = image.shape
            verts_int = np.clip(verts_int, [0,0,0], np.array(dims)-1)
            
            # Sample image intensities at surface vertices
            intensities = image[verts_int[:,0], verts_int[:,1], verts_int[:,2]]
            
            # Apply a 1D wavelet decomposition (using Haar wavelet here)
            coeffs = pywt.wavedec(intensities, 'db1', level=2)
            # Exclude the approximation coefficients (coeffs[0]) and compute the mean absolute value of detail coefficients
            detail_features = [np.mean(np.abs(c)) for c in coeffs[1:]]
            wavelet_feature = np.mean(detail_features)
        except Exception as e:
            self.error.error(f"Error computing surface wavelet feature: {e}")
            print(f"Error computing surface wavelet feature: {e}")
            wavelet_feature = None
        
        return wavelet_feature

    def calculate_surface_wavelet_from_series(self, row: pd.Series):
        """
        Wrapper function that takes a pandas Series (with 'Image' and 'Mask' columns containing
        file paths to .nii.gz files) and returns the computed feature.
        
        Parameters:
        row: pandas Series with columns 'Image' and 'Mask'
        
        Returns:
        wavelet_feature: A scalar summary of the wavelet detail coefficients.
        """
        try:
            mask_nii = nib.load(row['Mask'])
            mask_data = mask_nii.get_fdata()
            mask = (mask_data > 0).astype(np.uint8)

            image_nii = nib.load(row['Image'])
            image_data = image_nii.get_fdata()

            wavelet_feature = self.compute_surface_wavelet(image_data, mask)
            sample = os.path.basename(row['Image'])
        except Exception as e:
            self.error.error(f"Error computing surface wavelet feature: {e}")
            wavelet_feature = None
            try:
                sample = os.path.basename(row['Image'])
            except:
                sample = "No Image path in data or no sample name available"

        return sample, wavelet_feature

    def get_surface_wavelet(self, csv_file=None):
        """
        Get the surface wavelet feature for each ROI in a directory.
        :return wavelet_features: A list of tuples, each containing the wavelet feature and file path.
        """
        wavelet_features = []

        if csv_file is not None:
            # Read CSV file
            df = pd.read_csv(csv_file)
        else:
            df = self.data

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.calculate_surface_wavelet_from_series, [row for _, row in df.iterrows()]), total=len(df), desc="Calculating Surface Wavelet"):
                if not result[1] is None:
                    wavelet_features.append(result[1])
                else:
                    self.error.warning(f"Calculating wavelet surface features failed for {result[0]}!")
                    print(f"Calculating wavelet surface features failed for {result[0]}!")
                    wavelet_features.append(np.nan)

        return wavelet_features

    def compute_mesh_curvature(self, mask_path: str):
        """
        Extract a surface mesh from the binary mask and compute the mean curvature
        using PyVista's curvature filter.
        
        Parameters:
        mask_path: path to 3D binary mask file
        
        Returns:
        mean_curvature: Mean of the computed curvature values over the mesh vertices.
        """

        try:
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            mask = (mask_data > 0).astype(np.uint8)

            # Use marching cubes to get the surface mesh (vertices and faces)
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            # faces from marching_cubes are returned as (N, 3) but PyVista expects a flat array with a leading count per face.
            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            
            # Create a PyVista mesh
            mesh = pv.PolyData(verts, faces_pv)
            
            # Compute mean curvature
            curvature = mesh.curvature(curv_type="mean")
            mean_curvature = np.mean(curvature)
            sample_name = os.path.basename(mask_path)

        except Exception as e:
            self.error.error(f"Error computing mesh curvature feature: {e}")
            print(f"Error computing mesh curvature feature: {e}")
            mean_curvature = None
            try:
                sample_name = os.path.basename(mask_path)
            except:
                sample_name = "No Mask path in data or no sample name available"

        return sample_name, mean_curvature

    def get_mesh_curvature(self, csv_file=None):
        """
        Get the mean curvature feature for each ROI in a directory.
        :return curvature_features: A list containing the curvature feature
        """
        mesh_curvature = []

        if csv_file is not None:
            # Read CSV file
            df = pd.read_csv(csv_file)
        else:
            df = self.data

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.compute_mesh_curvature, [row["Mask"] for _, row in df.iterrows()]), total=len(df), desc="Calculating Mesh Curvature"):
                if not result[1] is None:
                    mesh_curvature.append(result[1])
                else:
                    self.error.warning(f"Calculating mesh curvature features failed for {result[0]}!")
                    print(f"Calculating mesh curvature features failed for {result[0]}!")
                    mesh_curvature.append(np.nan)

        return mesh_curvature

    def fractal_dimension_3d(self, mask_path, min_box=2, max_box=None, n_boxes=10):
        """
        Estimate the fractal dimension of a 3D binary mask using box counting.
        
        Parameters:
        mask_path : path to binary mask
        min_box : smallest box size (default 2)
        max_box : largest box size (default: half the smallest image dimension)
        n_boxes : number of box sizes to try
        
        Returns:
        fd : estimated fractal dimension
        """
        try:
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            mask = (mask_data > 0).astype(np.uint8)

            sample_name = os.path.basename(mask_path)

            # Ensure binary mask
            dims = np.array(mask.shape)
            if max_box is None:
                max_box = dims.min() // 2

            # Generate box sizes spaced logarithmically
            box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), num=n_boxes, dtype=int))
            counts = []
            
            for box_size in box_sizes:
                count = 0
                # Iterate over boxes in 3D
                for i in range(0, dims[0], box_size):
                    for j in range(0, dims[1], box_size):
                        for k in range(0, dims[2], box_size):
                            sub_box = mask[i:min(i+box_size, dims[0]),
                                        j:min(j+box_size, dims[1]),
                                        k:min(k+box_size, dims[2])]
                            if np.any(sub_box):
                                count += 1
                counts.append(count)
            
            # Avoid zeros
            counts = np.array(counts, dtype=float)
            valid = counts > 0
            box_sizes = box_sizes[valid]
            counts = counts[valid]
            
            # Fit a line to the log-log data: log(counts) = -D * log(box_size) + C
            coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
            fd = -coeffs[0]

        except Exception as e:
            self.error.error(f"Error computing 3D fractal dimension feature: {e}")
            print(f"Error computing 3D fractal dimension feature: {e}")
            fd = None
            try:
                sample_name = os.path.basename(mask_path)
            except:
                sample_name = "No Mask path in data or no sample name available"

        return sample_name, fd

    def get_fractal_dimension_3d(self, csv_file: str = None):
        """
        Get the fractal dimension feature for each ROI in a directory.
        :return fd_features: A list of tuples, each containing the fractal dimension feature and file path.
        """
        fd_features = []

        if csv_file is not None:
            # Read CSV file
            df = pd.read_csv(csv_file)
        else:
            df = self.data

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.fractal_dimension_3d, [row["Mask"] for _, row in df.iterrows()]), total=len(df), desc="Calculating Fractal Dimension"):
                if not result[1] is None:
                    fd_features.append(result[1])
                else:
                    self.error.warning(f"Calculating fractal dimension features failed for {result[0]}!")
                    print(f"Calculating fractal dimension features failed for {result[0]}!")
                    fd_features.append(np.nan)

        return fd_features

    def get_segmentation_roi_size(self, file_paths: list = None):
        """
        Get the size of the ROIs in a directory.
        :return roi_infos: A list of tuples, each containing the ROI size and file path.
        """

        roi_info = []

        if file_paths is None:
            if len(self.data["Mask"]) == 0:
                self.error.error("No mask files are provided for extracting information about the ROIs and Mask column in input data is empty.")
                raise ValueError("No mask files are provided for extracting information about the ROIs and Mask column in input data is empty.")
            else:
                # Recursively search for .nii.gz files
                file_paths = self.data["Mask"].tolist()

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.imap(self.get_roi_size, file_paths), total=len(file_paths), desc="Get roi size"):
                if result[0] is not None and result[1] is not None:
                    roi_info.append(result)

        return roi_info

    def count_segmented_values(self, image_path: str, segmentation_path: str):
        """
        Count the number of unique values present in the segmented area of the image.

        Parameters:
            image_path (str): Path to the image file.
            segmentation_path (str): Path to the segmentation file in .nii.gz format.

        Returns:
            int: Number of unique values present in the segmented area.
        """
        try:
            # Load the image and segmentation files
            image_data = nib.load(image_path).get_fdata()
            segmentation_data = nib.load(segmentation_path).get_fdata()

            # Mask the image with segmentation
            segmented_area = image_data[np.where(segmentation_data != 0)]

            # Count the number of unique values in the segmented area
            unique_values = np.unique(segmented_area)
            num_unique_values = len(unique_values)

            # Compute statistics of gray values
            min_gray = np.min(segmented_area)
            max_gray = np.max(segmented_area)
            mean_gray = np.mean(segmented_area)

            del image_data
            del segmented_area


            return image_path, min_gray, max_gray, mean_gray, num_unique_values
        except Exception as e:
            print(f"Error processing Image file {os.path.basename(image_path)} Mask file {os.path.basename(segmentation_path)}: {e}")
            print(traceback.format_exc())
            return image_path, None, None, None, None

    def extract_grey_values_in_roi(self, img_paths: list = None, seg_paths: list = None):
        """
        Extract the grey values in the segmented area of the image.
        :param img_paths: list of path to the image files
        :param seg_paths: list of path to the segmentation files
        :return: number of unique values in the segmented area of the image
        """


        nifti_infos = []
        # Recursively search for .nii.gz files
        if img_paths is None and seg_paths is None:
            if len(self.data["Image"]) == 0:
                self.error.error(
                    "File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
                raise ValueError(
                    "File paths are not provided for extracting information from nifi files and Image column in input data is empty.")
            else:
                img_paths = self.data["Image"].tolist()
                seg_paths = self.data["Mask"].tolist()
        else:
            self.error.error("No mask files are provided for extracting information about the ROIs.")
            raise ValueError("No mask files are provided for extracting information about the ROIs.")

        # Process files using multiple CPUs
        with mp_pool(processes=self.num_cpus) as pool:
            # Use tqdm with imap to show progress
            for result in tqdm(pool.starmap(self.count_segmented_values, zip(img_paths, seg_paths)), total=len(img_paths),
                               desc="Extract Grey Valus in ROI"):
                if result[1] is not None:
                    nifti_infos.append(result)
                else:
                    print("Could not extract Grey values from " + str(result[0]))
                    self.error.warning("Could not extract Grey values from " + str(result[0]))
                    nifti_infos.append((result[0], np.nan, np.nan, np.nan, np.nan))
            del result

        return nifti_infos

    def get_data_statistics(self):
        """
        Get the statistics of the data.
        :return: A dictionary containing the statistics of the data.
        """

        # Get the number of slices and slice thickness of the images
        nifti_infos = self.get_image_slice_stat()  # Get the number of slices and slice thickness of the images

        # Get the size of the ROIs
        roi_infos = self.get_segmentation_roi_size()
        
        roi_cc_and_labels = self.extract_cc_count_and_foreground_label()

        number_of_grey_values = self.extract_grey_values_in_roi()

        number_of_bins = self.get_default_number_of_bins(bin_width=int(self.bin_width))

        surface_wavelet = self.get_surface_wavelet()

        mesh_curvature = self.get_mesh_curvature()

        fractural_dimension_3d = self.get_fractal_dimension_3d()

        # Calculate the statistics
        num_slices = [info[0] for info in nifti_infos]
        slice_thickness = [info[1] for info in nifti_infos]
        images = [info[2] for info in nifti_infos]
        roi_sizes = [info[0] for info in roi_infos]
        masks = [info[1] for info in roi_infos]
        cc_count = [info[0] for info in roi_cc_and_labels]
        roi_labels = [info[1] for info in roi_cc_and_labels]
        min_gray = [info[1] for info in number_of_grey_values]# (image_path, min_gray, max_gray, mean_gray, num_unique_values)
        max_gray = [info[2] for info in number_of_grey_values]
        mean_gray = [info[3] for info in number_of_grey_values]
        num_unique_values = [info[4] for info in number_of_grey_values]
        number_of_bins = [info for info in number_of_bins]

        # generate a pd.Dataframe with columns as ['Image', 'Mask', 'num_slices', 'slice_thickness', 'roi_size']
        data = pd.DataFrame({'Image': images,
                             'Mask': masks,
                             'Number_of_slices': num_slices,
                             'Slice_thickness': slice_thickness,
                             'ROI_size': roi_sizes,
                             "Number_of_ROI_labels": roi_labels,
                             "Number_of_connected_components": cc_count,
                             "Number_of_grey_values_in_ROI": num_unique_values,
                             "Min_grey_value_in_ROI": min_gray,
                             "Max_grey_value_in_ROI": max_gray,
                             "Mean_grey_value_in_ROI": mean_gray,
                             "Number_of_bins": number_of_bins,
                             "Surface_wavelet": surface_wavelet,
                             "Mesh_curvature": mesh_curvature,
                             "Fractal_dimension_3D": fractural_dimension_3d})

        # Calculate Normalization for feautures
        normalization_features = ["Surface_wavelet", "Mesh_curvature", "Fractal_dimension_3D"]
        for col in normalization_features:
            if col in data.columns:
                col_zscore = col + '_zscore'
                data[col_zscore] = (data[col] - data[col].mean()) / data[col].std(ddof=0)
                data = data.drop(columns=[col])

        return data

    def make_boxplot(self, df: pd.DataFrame, out_file: str = None, title: str=None):
        """
        Make a boxplot of the data. including jittered data points for num_slices, slice_thickness and roi_size.
        :param df: pd.DataFrame of input parameter
        :param out_file: str: output file path to save the plot
        :param title: str: title of the plot
        """
        log_scale = False

        if "ROI size" in df.columns or 'Number of grey values in ROI' in df.columns:
            log_scale = True

        vals, names, xs = [], [], []
        for i, col in enumerate(df.columns):
            vals.append(df[col].values)
            names.append(col)
            xs.append(np.random.normal(i + 1, 0.04,
                                       df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

            plt.boxplot(vals, labels=names)
            palette = ['r', 'g', 'b', 'y']
            for x, val, c in zip(xs, vals, palette):
                plt.scatter(x, val, alpha=0.4, color=c)

        if log_scale:
            plt.yscale("log")

        plt.title(title, fontsize=20)

        plt.savefig(out_file,
                    bbox_inches='tight',
                    dpi=200)

        plt.close()

    def get_profile(self, df):
        """
        Generate a pandas profile for the data
        """

        if "Image" in df.columns:
            for img in df["Image"].values:
                df.loc[df["Image"] == img, "Image"] = os.path.basename(img)

        if "Mask" in df.columns:
            for msk in df["Mask"].values:
                df.loc[df["Mask"] == msk, "Mask"] = os.path.basename(msk)

        not_interesting_columns = ["Versions",
                                   "diagnostics_Configuration_Settings",
                                   "diagnostics_Configuration_EnabledImageTypes",
                                   "diagnostics_Image-original_Hash",
                                   "diagnostics_Mask-original_Hash",
                                   ]

        for dropping in not_interesting_columns:
            for col in df.columns:
                if dropping in col:
                    df.drop(col, axis=1, inplace=True)

        if self.generate_profile:
            try:
                self.logger.info("Generating Data Profile.")
                profile = ProfileReport(df, title="RPTK Data Profile")
                if len(df.columns) > 100:
                    self.logger.info("Setting out profile to json as the data is too much.")
                    self.profile_format = "json"
                if self.profile_format == "html":
                    profile.to_file(self.out_path + "/Data_stats/RPTK_Data_Fingerprint.html")
                elif self.profile_format == "json":
                    profile.to_file(self.out_path + "/Data_stats/RPTK_Data_Fingerprint.json")
                else:
                    self.error.warning("Format \"{}\" not known to save the profile! Can not save the profile!".format(self.profile_format))
            except Exception as e:
                self.error.warning("Data fingerprint profile generation failed! " + str(e))
                print("Warning: Data fingerprint profile generation failed! " + str(e))

    def run(self):
        """
        Run the DataStatistics class.
        :return:
        """
        # Data Fingerprint:
        # ROI mean size
        # ROI number
        # Distribution of spacingsLogGenerator
        # Time relation
        # Image Modality
        # Intensity distribution
        # Image dimension

        # check if analysis has beendone before
        if not os.path.exists(self.out_path + "/Data_stats/Input_Data_stats.csv"):

            # Get the statistics of the data
            stat_data = self.get_data_statistics()

            # Add Prediction Label to data stat
            if self.prediction_label in self.data.columns:
                stat_data[self.prediction_label] = self.data.loc[
                    (self.data["Image"] == stat_data["Image"]) & (self.data["Mask"] == stat_data["Mask"]), self.prediction_label]
            else:
                self.error.error(
                    "{} is missing in input data! Please provide the label to predict.".format(self.prediction_label))
                raise ValueError(
                    "{} is missing in input data! Please provide the label to predict.".format(self.prediction_label))

            self.data = stat_data

            # plot figures
            if not os.path.exists(self.out_path + "/Data_stats/plots/"):
                os.makedirs(self.out_path + "/Data_stats/plots/")
        
            title = str(Path(self.out_path).parents[1]).split("/")[-1]

            # Make a boxplot of Image data
            self.make_boxplot(df=self.data[['Number_of_slices']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Number_of_slices.png",
                              title=title)

            # Make a boxplot of Image data
            self.make_boxplot(df=self.data[['Slice_thickness']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Slice_thickness.png",
                              title=title)

            # Make a boxplot of Mask data
            self.make_boxplot(df=self.data[['ROI_size']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/ROI_size.png",
                              title=title)

            # Make a boxplot of Mask data
            self.make_boxplot(df=self.data[['Number_of_ROI_labels']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/ROI_labels.png",
                              title=title)

            # Make a boxplot of Mask data
            self.make_boxplot(df=self.data[['Number_of_connected_components']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/CC_count.png",
                              title=title)

            # Make a boxplot of the number of grey values in the ROI
            self.make_boxplot(df=self.data[['Number_of_grey_values_in_ROI']].copy(),
                                out_file=self.out_path + "/Data_stats/plots/Number_of_grey_values_in_ROI.png",
                                title=title)

            # Make a boxplot of the number of grey values in the ROI
            self.make_boxplot(df=self.data[['Min_grey_value_in_ROI']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Min_grey_values_in_ROI.png",
                              title=title)

            # Make a boxplot of the number of grey values in the ROI
            self.make_boxplot(df=self.data[['Max_grey_value_in_ROI']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Max_of_grey_values_in_ROI.png",
                              title=title)

            # Make a boxplot of the number of grey values in the ROI
            self.make_boxplot(df=self.data[['Mean_grey_value_in_ROI']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Mean_of_grey_values_in_ROI.png",
                              title=title)

            # Make a boxplot of the number of bins in the ROI
            self.make_boxplot(df=self.data[['Number_of_bins']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Number_of_bins_in_ROI.png",
                              title=title)

            # Make a boxplot of the Surface wavelet in the ROI
            # self.make_boxplot(df=self.data[['Surface_wavelet']].copy(),
            #                   out_file=self.out_path + "/Data_stats/plots/Surface_wavelet.png",
            #                   title=title)
            
            if "Surface_wavelet_zscore" in self.data.columns:
                self.make_boxplot(df=self.data[['Surface_wavelet_zscore']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Surface_wavelet_zscore.png",
                              title=title)

            # Make a boxplot of the Mesh curvature in the ROI
            # self.make_boxplot(df=self.data[['Mesh_curvature']].copy(),
            #                   out_file=self.out_path + "/Data_stats/plots/Mesh_curvature.png",
            #                   title=title)

            if "Mesh_curvature_zscore" in self.data.columns:
                self.make_boxplot(df=self.data[['Mesh_curvature_zscore']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Mesh_curvature_zscore.png",
                              title=title)

            # Make a boxplot of the Fractal dimension 3D in the ROI
            # self.make_boxplot(df=self.data[['Fractal_dimension_3D']].copy(),
            #                   out_file=self.out_path + "/Data_stats/plots/Fractal_dimension_3D.png",
            #                   title=title)
            
            if "Fractal_dimension_3D_zscore" in self.data.columns:
                self.make_boxplot(df=self.data[['Fractal_dimension_3D_zscore']].copy(),
                              out_file=self.out_path + "/Data_stats/plots/Fractal_dimension_3D_zscore.png",
                              title=title)

            

            # Save the data statistics to a csv file

            if not os.path.exists(self.out_path + "/Data_stats/"):
                os.makedirs(self.out_path + "/Data_stats/")

            if self.extract_radiomics_fingerprint:

                from rptk.src.feature_filtering.DataConfigurator import DataConfigurator

                radiomics.setVerbosity(30)
                #logger = radiomics.logger
                #logger.setLevel(logging.ERROR)

                self.py_extractor = featureextractor.RadiomicsFeatureExtractor(
                    os.path.abspath(self.radiomics_fingerprint_yaml))
                radiomics_fingerprint = pd.DataFrame()

                try:
                    with tqdm(total=len(self.data), desc='Extracting Radiomics Fingerprint') as pbar:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
                            futures = {executor.submit(self.get_radiomics_fingerprint, row): row for i, row in
                                       self.data.iterrows()}
                            
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    result = future.result()
                                    if result is not None:
                                        radiomics_fingerprint = pd.concat([radiomics_fingerprint, result])
                                except Exception as exc:
                                    logging.error(f"Generated an exception: {exc}")
                                pbar.update(1)
                                
                except Exception as ex:
                    print("PyRadiomics Feature Extraction Failed! " + str(ex))
                    self.error.error("PyRadiomics Feature Extraction Failed! " + str(ex))

                data_config = DataConfigurator(out_path=self.out_path,
                                               logger=self.logger,
                                               error=self.error,
                                               data=radiomics_fingerprint,
                                               extractor="PyRadiomics",
                                               n_cpu=self.num_cpus,
                                               RunID=self.RunID)

                data_config.config_pyradiomics_features()

                radiomics_fingerprint = data_config.data

                # fingerprint.to_csv(self.out_path + "/Data_stats/Radiomics_fingerprint.csv", index=False)
                self.data = pd.concat([self.data, radiomics_fingerprint], axis=1)

                # drop duplicated columns
                df2 = self.data.loc[:, ~self.data.columns.duplicated()]
                self.data = df2

            self.data.to_csv(self.out_path + "/Data_stats/Input_Data_stats.csv", index=False)

            self.logger.info("Input Data Statistics: \n" +
                             "\t\t\t\t\t\t\t\t\t\tmean Number of slices:" + str(round(self.data['Number_of_slices'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tNumber of slices std:" + str(round(self.data['Number_of_slices'].std(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tmean Slice thickness:" + str(round(self.data['Slice_thickness'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tSlice thickness std:" + str(round(self.data['Slice_thickness'].std(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tmean ROI size:" + str(round(self.data['ROI_size'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tROI size std:" + str(round(self.data['ROI_size'].std(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tmean Number of ROI labels:" + str(round(self.data['Number_of_ROI_labels'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tNumber of ROI labels std:" + str(round(self.data['Number_of_ROI_labels'].std(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tmean Number of connected components:" + str(round(self.data['Number_of_connected_components'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tNumber of connected components std:" + str(round(self.data['Number_of_connected_components'].std(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tmean Number of grey values in ROI:" + str(round(self.data['Number_of_grey_values_in_ROI'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tNumber of grey values in ROI std:" + str(round(self.data['Number_of_grey_values_in_ROI'].std(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tNumber of bins in ROI mean:" + str(round(self.data['Number_of_bins'].mean(), 2)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\tNumber of bins in ROI std:" + str(round(self.data['Number_of_bins'].std(), 2)) + "\n")
            
            if self.data.shape[1] < 300: # if not too many features in dataframe
                # Make a summary Profile of the data fingerprint
                self.get_profile(df=self.data.copy())
            else:
                print("Large dataset detected! Profile generation would take very long time. Skipping data profile generation.")

        else:
            self.data = pd.read_csv(self.out_path + "/Data_stats/Input_Data_stats.csv")
            self.logger.info("Data statistics are already extracted and saved to the output directory.")
            print("Data statistics are already extracted and saved to the output directory.")

        self.success = True

        self.logger.info("Data statistics are successfully extracted and saved to the output directory.")

        return self.success, self.data
