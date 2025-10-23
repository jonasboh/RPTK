from multiprocessing import Pool
import numpy as np
import os
import SimpleITK as sitk
import argparse
import tqdm
import shutil

import sys
import os
import gc

from skimage.measure import label, regionprops
import skimage.morphology
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

import more_itertools
from more_itertools import batched

#from src.image_processing.MR_Normalizer import MR_Normalizer
#from src.segmentation_processing import SegProcessor
#from src.config.Log_generator_config import *
#from src.image_processing.DataHandler import *

import gzip
import shutil
import pandas as pd
import concurrent.futures
from functools import partial

class SegmentationFilter:
    def __init__(self,
                 df: pd.DataFrame = pd.DataFrame(),  # dataframe with patient information
                 multilabel_seg_path: str = "",  # path to multilabel segmentation
                 list_of_seg_paths=None,  # list of segmentation paths
                 list_of_seg_arrays=None,  # list of segmentation arrays
                 out_path: str = "",  # output path
                 n_cpus: int = 1,  # number of cpus for multiprocessing
                 logger=None,  # logger for log file
                 error=None,  # error logger
                 runid: str = None,  # run id for logging
                 roi_threshold: int = 3,  # minimum number of voxels in a region of interest
                 max_num_rois: int = 1,  # maximum number of regions of interest (sorted by size) = 0 all ROI's are kept
                 resampling: bool = False,  # resample to 1mm isotropic voxels
                 scanner_protocols=None,  # list of scanner protocols
                 consider_multiple_labels = False,  # consider multiple labels in the segmentation
                 use_previous_output = False,  # use previous output
                 fast_mode = False  # fast mode for processing
                 ):

        if list_of_seg_arrays is None:
            list_of_seg_arrays = []
        if list_of_seg_paths is None:
            list_of_seg_paths = []

        self.df = df
        self.multilabel_seg_path = multilabel_seg_path
        self.list_of_seg_paths = list_of_seg_paths
        self.list_of_seg_arrays = list_of_seg_arrays
        self.out_path = out_path
        self.n_cpus = n_cpus
        self.roi_threshold = roi_threshold
        self.max_num_rois = max_num_rois
        self.resampling = resampling
        self.scanner_protocols = scanner_protocols
        self.consider_multiple_labels = consider_multiple_labels
        self.use_previous_output = use_previous_output
        self.fast_mode = fast_mode

        # Outcome Parameters
        self.empty_paths = []  # list of empty segmentation paths
        self.connected_compenents_sum = 0  # number of connected components
        self.segmentations_over_max_num_rois = []  # list of segmentation paths with more than max_num_rois
        self.segmentations_under_roi_threshold = []  # list of segmentation paths with ROIs smaller than roi_threshold
        self.removed_small_rois_sum = 0  # number of segmentations with removed small ROIs
        self.segmentations_with_multiple_labels = []  # list of segmentation paths with multiple labels

        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        if runid is None:
            self.runid = time.strftime("%Y%m%d-%H%M%S")
        else:
            self.runid = runid

        if logger is None:
            self.logger = LogGenerator(log_file_name=self.out_path + "/SegmentationFilter_" + self.runid + ".log",
                                       logger_topic="Segmentation Filtering").generate()
        else:
            self.logger = logger

        if error is None:
            self.error = LogGenerator(log_file_name=self.out_path + "/SegmentationFilter_" + self.runid + ".err",
                                       logger_topic="Segmentation Filtering Failure").generate()
        else:
            self.error = error

        if self.multilabel_seg_path == "":
            self.error.warning("No multilabel segmentation path given!")
            raise ValueError("No multilabel segmentation path given!")

    def check_empty_segmentations(self, list_of_seg_paths=None):
        """
        Check if segmentation is empty.
        :param list_of_seg_paths: list with segmenation paths for checking
        """

        list_of_seg_arrays = None
        not_empty = []
        not_empty_paths = []

        # if no list of segmentation paths is given, use the list of segmentation paths from the object
        if list_of_seg_paths is None:
            list_of_seg_paths = self.list_of_seg_paths

        if len(list_of_seg_paths) == 0:
            print("No Segmentation to process! Skipping!")
            return not_empty, not_empty_paths

        # if no list of segmentation arrays is given, read arrays from the segmentation paths
        if (list_of_seg_arrays is None) and (len(self.list_of_seg_arrays) == 0):
            list_of_seg_arrays = []
            for seg in list_of_seg_paths:
                seg_ = sitk.ReadImage(seg)
                list_of_seg_arrays.append(sitk.GetArrayFromImage(seg_))

        # if no list of segmentation arrays is given, use the list of segmentation arrays from the object
        elif (list_of_seg_arrays is None) and (len(self.list_of_seg_arrays) != 0):
            list_of_seg_arrays = self.list_of_seg_arrays

        else:
            print("Error: No segmentation paths or arrays given!")
            sys.exit("Error:No data given for quality check!")

        if len(list_of_seg_arrays) != len(list_of_seg_paths):
            print("Error: Number of segmentation paths and arrays do not match!")
            sys.exit("Error:No data given for quality check!")



        for seg, path in tqdm.tqdm(zip(list_of_seg_arrays, list_of_seg_paths),
                                   total=len(list_of_seg_arrays), desc="Check empty Segmentations"):

            if np.sum(seg) <= 0:
                # remove empty segmentation from df
                self.df.drop(self.df[self.df["Mask"] == path].index, inplace=True)

                self.error.warning("Detected empty segmentation: " + str(path))
                self.empty_paths.append(path)
            else:
                not_empty.append(seg)
                not_empty_paths.append(path)

        if len(self.empty_paths) > 0:
            self.error.warning("Found " + str(len(self.empty_paths)) + " empty segmentations!")
            self.error.info("Empty segmentation summary: " + self.out_path + "/empty_segmentations_" + self.runid + ".csv")
            print("Found " + str(len(self.empty_paths)) + " empty segmentations!")
            
            empty = pd.DataFrame(self.empty_paths, columns=["empty_segmentations"])
            empty.to_csv(self.out_path + "/empty_segmentations_" + self.runid + ".csv", index=False)

        self.logger.info("Found " + str(len(not_empty_paths)) + " not empty segmentations!")

        return not_empty, not_empty_paths

    def ignore_small_segments(self, seg_array, seg_path):
        """
        Ignore small segments in a binary segmentation.

        Parameters
        ----------
        seg_array : ndarray or sitk.Image
            Binary segmentation mask.
        seg_path : str
            Path to segmentation.
            
        Returns
        -------
        ndarray
            Binary segmentation mask with small segments removed.
        """

        # Convert to numpy array if it's a SimpleITK image
        if isinstance(seg_array, sitk.SimpleITK.Image):
            seg_array = sitk.GetArrayFromImage(seg_array)

        seg_array = seg_array.astype(np.uint8)  # Ensure low-memory type
        labeled = label(seg_array, connectivity=1)  # Label connected components
        seg_props = regionprops(labeled)

        # Modify the original mask to save memory
        removed = 0
        for prop in seg_props:
            if prop.area < int(self.roi_threshold):
                labeled[labeled == prop.label] = 0  # Set small regions to background
                removed += 1
                self.segmentations_under_roi_threshold.append(seg_path)
                self.error.warning(f"Found small segment with {prop.area} voxels in {os.path.basename(seg_path)}.")

        if removed > 0:
            self.error.warning(
                f"Removed {removed} small segment/s from {os.path.basename(seg_path)}!"
            )
            self.removed_small_rois_sum += removed

        # Convert the labeled array back to a binary mask
        new_mask = (labeled > 0).astype(np.uint8)  # Cast to 0/1 binary mask
        
        return new_mask


    def convert_seg_to_multilabel(self, seg, seg_path):
        """
        assigns different labels to connected segmentation components ordered according to size
        :param seg_path: path to segmentation
        :param seg: segmentation array (binary)
        :return: pd.DataFrame multilabel segmentation array (uint16) with labels ordered according to size of connected
        components (largest component = 1, second largest = 2, ...)
        """

        os.mkdir(self.multilabel_seg_path)

        # load segmentation and convert to binary
        seg[seg != 0] = 1

        blobs_labels, num_labels = measure.label(seg, background=0, return_num=True)
        print("Scan: " + str(os.path.basename(seg_path)))
        print("Number of labels: " + str(num_labels))

        props = measure.regionprops(blobs_labels)

        # select n largest components
        dict_label_area = {}
        for i in range(num_labels):
            dict_label_area[props[i].label] = props[i].area

        sorted_d = dict(sorted(dict_label_area.items(), key=operator.itemgetter(1), reverse=True))
        img_arr_out = np.zeros(blobs_labels.shape)
        selected_labels = list(sorted_d.keys())[0:num_labels]

        ordered_labels = []
        for i in range(num_labels):
            img_arr_out[blobs_labels == selected_labels[i]] = i + 1
            ordered_labels.append((i + 1, img_arr_out[blobs_labels == selected_labels[i]].shape[0]))

        if self.scanner_protocols is None:
            new_name = os.path.basename(seg_path)[:-len(".nii.gz")] + '_multilabel.nii.gz'
        else:
            scanner_protocol = self.df[self.df["Mask"] == seg_path]["Image_Transformation"].values[0]
            new_name = os.path.basename(seg_path)[:-len(".nii.gz")] + scanner_protocol + '_multilabel.nii.gz'

        write_nifti_file(img_arr_out, self.multilabel_seg_path + new_name, seg_path, self.error)

        #sitk_img_out = sitk.GetImageFromArray(img_arr_out.astype(np.uint16))
        #sitk_img_out.CopyInformation(sitk_img)
        #sitk.WriteImage(sitk_img_out,
        #                os.path.join(self.output + "/multilabel_seg/",
        #                             os.path.basename(seg_path)[:-len(".nii.gz")] + '_multilabel.nii.gz'),
        #                useCompression=True)

        multi_label = pd.DataFrame({"ID": [os.path.basename(seg_path)[:-len(".nii.gz")]],
                                    "Num_Label": [num_labels],
                                    "SegPath": [os.path.join(self.multilabel_seg_path,
                                                             os.path.basename(seg_path)[:-len(".nii.gz")] +
                                                             '_multilabel.nii.gz')]})

        return multi_label


    def sort_rois(self, seg_array, path):
        """
        Gets a segmentation and returns a list of its ROIs sorted by size in descending order.

        Args:
            segmentation (ndarray): A 2D or 3D array representing the segmentation mask.
            roi_threshold (int): Minimum size of an ROI to be kept.

        Returns:
            list of ndarray: A list of 2D or 3D arrays, each representing an ROI in the segmentation mask.
                             The ROIs are sorted by size in descending order.
                             :param path:
                             :param seg_array:
        """

        sorted_rois = []
        list_of_labels = []
        dict_label_area = {}

        connected_comp_labels, num_labels = label(seg_array, background=0, return_num=True)

        if num_labels > 1:
            if num_labels > self.max_num_rois:
                self.segmentations_over_max_num_rois.append(os.path.basename(path))
                self.error.warning("Number of connected Components in {} larger than max_num_rois: {}".format(
                    os.path.basename(path), num_labels))
            else:
                self.error.warning(
                    "Number of connected Components in {}: {}".format(os.path.basename(path), num_labels))
        else:
            if num_labels == 0:
                self.error.warning(
                    "No Component found in {}: {}".format(os.path.basename(path), num_labels))
                if path not in self.empty_paths:
                    self.empty_paths.append(os.path.basename(path))

        self.connected_compenents_sum += num_labels

        props = regionprops(connected_comp_labels)

        # Sort the labels by area size
        for i in range(num_labels):
            dict_label_area[props[i].label] = props[i].area

        sorted_d = dict(sorted(dict_label_area.items(), key=operator.itemgetter(1), reverse=True))

        if self.max_num_rois != 0:
            selected_labels = list(sorted_d.keys())[0:int(self.max_num_rois)] # TODO: get number of segmentations with more then max_num_rois
        else:
            selected_labels = list(sorted_d.keys())

        if num_labels > 1:
            self.error.warning("Labels renamed by Size: " + str(sorted_d))
            self.error.warning("Selected labels with max area: " + str(selected_labels))

        # make an array for each selected label
        for i in range(num_labels):
            if len(selected_labels) > i:
                img_arr_out = np.zeros(connected_comp_labels.shape)
                img_arr_out[connected_comp_labels == selected_labels[i]] = 1
                list_of_labels.append(i)  # for segmentation naming
                sorted_rois.append(img_arr_out)

        return sorted_rois, list_of_labels

    def multi_label(self, seg_array, path):
        """
        Check if the array has multiple labels.
        :parameter: seg_array: ndarray
        :return: bool
        """

        unique_labels = np.unique(seg_array)    # get unique labels
        num_labels = len(unique_labels) - 1    # remove background label

        if num_labels > 1:
            print("File {} has {} labels".format(os.path.basename(path), num_labels))
            return True
        else:
            return False

    def count_rois(self, seg_array, path):
        """
        Count the number of connected components in a segmentation file.
        :parameter: seg_array: ndarray
        :return: list: list of connected components
        :return: list: list of labels
        """

        list_of_labels = []

        # Count connected components in the segmentation - Label connected regions of an integer array.
        connected_comp_labels, num_labels = label(seg_array, background=0, return_num=True)

        if num_labels > 1:
            self.error.warning("Detected multiple labeled Components in {}: {}".format(os.path.basename(path),
                                                                                       num_labels))

        print("Number of Connected components in {}: {}".format(os.path.basename(path), num_labels))

        u_label = np.unique(seg_array)
        for label_ in tqdm.tqdm(u_label, desc="Count Components in Segmentation"):
            if label_ != 0:
                list_of_labels.append(label_)

        if len(list_of_labels) > 1:
            self.error.warning("Detected multiple Components in {}: {}".format(os.path.basename(path), len(list_of_labels)))

        print("Number of Components in {}: {}".format(os.path.basename(path), len(list_of_labels)))

        return connected_comp_labels, list_of_labels

    def split_segmentation_file(self, seg_array, seg_path):
        """
        Checks if a segmentation file in .nii.gz format has multiple ROIs with label 1.

        Args:
            filename (str): The path to the segmentation file.
            roi_threshold (int): The minimum size of an ROI to be included in the output.

        Returns:
            list of ndarray: A list of 3D arrays, each representing an ROI in the segmentation file.
            :param seg_path:
            :param seg_array:
        """
        # List for storing the ROIs
        rois = []

        # Check if the data array has multiple labels
        if self.multi_label(seg_array, seg_path):
            # Check which labels are in the mask (label 1 or 2 or 3 ...)
            vals, counts = np.unique(seg_array, return_counts=True)

            # There are more than 2 labels in the mask (more than 0 (Background) & 1 (Foreground))
            self.logger.info("Found multiple labels in " + os.path.basename(seg_path) + ": "
                               + str(vals) + " Counts: " + str(counts))
            self.segmentations_with_multiple_labels.append(seg_path)

            if self.consider_multiple_labels:
                # set all labels in mask to 1 (Foreground) --> ROI
                seg_array[seg_array > 0] = 1

        # If data contains multiple ROIs with the same label
        ## connected_comp_labels, list_of_labels = self.count_rois(seg_array, seg_path) --> ??? Maybe?
        sorted_rois, list_of_labels = self.sort_rois(seg_array, seg_path)
        if len(sorted_rois) > 1:
            self.logger.info("Found multiple ROIs with same label in " + os.path.basename(seg_path) + ": "
                               + str(len(sorted_rois)))

        for roi in sorted_rois:
            rois.append(roi)

        return rois, list_of_labels

    def write_new_segmentation(self, array, base_name, summary, path, label, img):

        # Write segmentations
        #if self.resampling:
        #    base_name = base_name + "_resampled"

        new_name = base_name + ".nii.gz"
        write_nifti_file(array, self.multilabel_seg_path + "/" + new_name, path, self.error)
        # self.logger.info("New segmentation file name: {}".format(new_name))

        try:
            summary_tmp = pd.DataFrame({"Image": [self.df.loc[self.df["Image"] == img, "Image"].values[0]],
                                        "Mask": [os.path.abspath(self.multilabel_seg_path + "/" + new_name)],
                                        "Modality": [self.df.loc[self.df["Mask"] == path, "Modality"].values[0]],
                                        "Raw_Mask": [path],
                                        "ROI_threshold": [self.roi_threshold],
                                        "Max_num_rois": [self.max_num_rois],
                                        "ROI_count": [label],
                                        })

            summary = pd.concat([summary, summary_tmp], ignore_index=True)
        except:
            self.error.warning("Error in writing radiomics filtering summary file for segmentation: {}".format(path))
            raise IOError("Error in writing radiomics filtering summary file for segmentation: {}".format(path))

        # replace the path in self.df with the new path
        self.df.loc[self.df["Image"] == img, "Mask"] = os.path.abspath(self.multilabel_seg_path + "/" + new_name)

        return summary

    def split_segmentation_files(self, seg, path=None, summary=None):
        """
        Perform Segmetaton file cleaning based on connected components.
        :param seg:
        :param path: str path to entry
        :param summary: pd.DataFrame for parameters included
        :return:
        """
        processed_imgs = []

        if path is None:
            path = seg[1]
            real_seg = seg[0]
            seg = real_seg

        try:
            list_of_rois, list_of_labels = self.split_segmentation_file(seg, path)

            # Write segmentations for each ROI/Label
            for label, array in zip(list_of_labels, list_of_rois):

                base_name = os.path.basename(path)[:-len(".nii.gz")] + "_roiL_" + str(label) + "_roiT_" + str(
                    self.roi_threshold) + "_roiN_" + str(self.max_num_rois)

                if self.scanner_protocols is not None:
                    protocols4sample = self.df.loc[self.df["Mask"] == path, "Image_Transformation"].values.tolist()
                    if len(protocols4sample) > 1:
                        # got multiple image transformations for same segmentation file
                        for protocol in protocols4sample:
                            if protocol in self.scanner_protocols:
                                # get image with segmentation and protocol
                                imgs = self.df.loc[self.df["Mask"] == path, "Image"].values.tolist()
                                for img in imgs:
                                    if protocol == self.df.loc[self.df["Image"] == img, "Image_Transformation"].values[0]:
                                        if img not in processed_imgs:
                                            name = base_name + "_" + str(protocol)
                                            summary = self.write_new_segmentation(array, name, summary, path, label, img)
                                            processed_imgs.append(img)
                                            break

                    if len(protocols4sample) == 1:
                        # got one image transformation for same segmentation file
                        protocol = protocols4sample[0]
                        if protocol in self.scanner_protocols:
                            # get image with segmentation and protocol
                            img = self.df.loc[self.df["Mask"] == path, "Image"].values[0]
                            name = base_name + "_" + str(protocol)
                            summary = self.write_new_segmentation(array, name, summary, path, label, img)

                else:
                    img = self.df.loc[self.df["Mask"] == path, "Image"].values[0]
                    summary = self.write_new_segmentation(array, base_name, summary, path, label, img)

                # if no transformation is given, use the original segmentation file name
                if len(self.scanner_protocols) == 0:
                    # get image with segmentation and protocol
                    # Write segmentations
                    # if self.resampling:
                    #    base_name = base_name + "_resampled"

                    new_name = base_name + ".nii.gz"
                    write_nifti_file(array, self.multilabel_seg_path + "/" + new_name, path, self.error)

                    # replace the path in self.df with the new path
                    self.df.loc[self.df["Mask"] == path, "Mask"] = os.path.abspath(
                        self.multilabel_seg_path + "/" + new_name)
        except Exception as e:
            self.error.warning("Failed to split segmentation file! {}".format(str(e)))
            return None

        return summary

    def ignore_small_segments_and_split_segmentation_files(self, seg_path, summary=None):
        """
        """
        seg = seg_path[0]
        path = seg_path[1]

        # ignore small segments
        cleaned_seg = self.ignore_small_segments(seg, path)

        # split segmentation file if needed
        try:
            # Process the segmentation file and update the summary
            results = self.split_segmentation_files(cleaned_seg, path, summary=summary)
            if results is None:
                self.error.error(f"Failed splitting segmentation file {path}")
                print(f"Failed splitting segmentation file {path}")

        except Exception as e:
            self.error.warning(f"Error splitting segmentation file {path}: {e}")
            print(f"Error splitting segmentation file {path}: {e}")

        return results

    def process(self):
        """
        Process the segmentation files in the list of paths.
        :return:
        """

        summary = pd.DataFrame({"Image": [],
                                "Mask": [],
                                "Modality": [],
                                "Raw_Mask": [],
                                "ROI_threshold": [],
                                "Max_num_rois": [],
                                "ROI_count": []})

        # for processed samples with different protocols but the same segmentation
        processed_imgs = []
        list_of_seg_paths = self.df["Mask"].values.tolist()
        if self.use_previous_output:
            self.logger.info("Using previous output ...")

            self.done_segs = []

            # search outout files in previous output folder
            self.previous_output_path = self.out_path + "/" + self.runid + "_preprocessing_out.csv"
            if os.path.isfile(self.previous_output_path):
                previous_results = pd.read_csv(self.previous_output_path, sep=",", header=0, index_col=False)
                previous_results_seg_IDs = previous_results.loc[:, "Mask"].values.tolist()
                previous_results_seg_IDs = [x for x in previous_results_seg_IDs if self.multilabel_seg_path in x]
                previous_results_seg_IDs = [os.path.basename(x[:-len(".nii.gz")]) for x in previous_results_seg_IDs]
                for seg_ID in self.df["Mask"].values.tolist():
                    for p_seg in previous_results_seg_IDs:
                        if p_seg.startswith(os.path.basename(seg_ID)[:-len(".nii.gz")]):
                            self.done_segs.append(seg_ID)

                list_of_seg_paths = self.df["Mask"].values.tolist()
                list_of_seg_paths = [x for x in list_of_seg_paths if x not in self.done_segs]

                self.logger.info("Number of processed segmentations: {}".format(len(self.done_segs)))
            else:
                self.logger.info("No previous output found in {}".format(self.previous_output_path))

        # Get the list of processed images
        processed_imgs = self.df["Image"].unique().tolist()
        self.logger.info("Number of processed images: {}".format(len(processed_imgs)))

        # Check if the list of paths is empty
        self.list_of_seg_arrays, self.list_of_seg_paths = self.check_empty_segmentations(list_of_seg_paths=list_of_seg_paths)

        if len(self.list_of_seg_arrays) != 0 and len(self.list_of_seg_paths) != 0:
            
            # for big datasets --> too much memory consumption
            # Filter Segmentations by removing small ROI components and splitting the ROI if there are more components accepted
            partial_ignore_small_segments_and_split_segmentation_files = partial(self.ignore_small_segments_and_split_segmentation_files, summary=summary)
            with tqdm.tqdm(total=len(self.list_of_seg_paths), desc="Preprocess Segmentation Files") as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
                    # Generate batches for better memory usage
                    batches = list(batched(zip(self.list_of_seg_arrays, self.list_of_seg_paths), self.n_cpus))
                    for batch in batches:
                        # Prepare arguments for multiprocessing
                        args = [(seg, path) for seg, path in batch]

                        # Map the function to the pool of workers
                        for results in executor.map(partial_ignore_small_segments_and_split_segmentation_files, args, chunksize=self.n_cpus):
                            pbar.update(1)
                            if results is None:
                                self.error.error(f"Failed splitting segmentation file {path}")
                                print(f"Failed splitting segmentation file {path}")

                        # Perform cleanup after processing the batch
                        gc.collect()

            #else:

                # Remove small components of ROIs
            #    cleaned_segs = []
                # for seg, path in zip(self.list_of_seg_arrays, self.list_of_seg_paths):
            #    for seg, path in tqdm.tqdm(zip(self.list_of_seg_arrays, self.list_of_seg_paths),
            #                            total=len(self.list_of_seg_paths),
            #                            desc="Filter for Segmentation Artifacts"):
            #        cleaned_segs.append(self.ignore_small_segments(seg, path))

            #    print("Memory usage for processing: " + str(sys.getsizeof(cleaned_segs)) + " bytes")

                # Split the segmentation files into ROIs and save them
                # Update 26.07.24 for faster computation of big datasets and better error handling

                # Partially apply the summary parameter to split_segmentation_files
                #
                # if there is a memory overload, use the slow mode
            #    if not self.fast_mode:
                    # Progress bar for tracking progress
            #        with tqdm.tqdm(total=len(cleaned_segs), desc="Split Segmentation Files") as pbar:
            #            for seg, path in zip(cleaned_segs, self.list_of_seg_paths):
            #                try:
            #                    # Process the segmentation file and update the summary
            #                    results = self.split_segmentation_files(seg, path, summary=None)
            #                    if results is None:
            #                        self.error.error(f"Failed splitting segmentation file {path}")
            #                        print(f"Failed splitting segmentation file {path}")##

            #                except Exception as e:
            #                    self.error.warning(f"Error processing {path}: {e}")
            #                    print(f"Error processing {path}: {e}")

            #                # Update progress bar and clean up memory
            #                pbar.update(1)
            #                gc.collect()
            #    else:
            #        # Process files using multiple CPUs
            #        partial_split_segmentation_files = partial(self.split_segmentation_files, summary=summary)
            #        with tqdm.tqdm(total=len(cleaned_segs), desc="Split Segmentation Files") as pbar:
            #            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            #                # Generate batches for better memory usage
            #                batches = list(batched(zip(cleaned_segs, self.list_of_seg_paths), self.n_cpus))
            #                
            #                for batch in batches:
            #                    # Prepare arguments for multiprocessing
            #                    args = [(seg, path) for seg, path in batch]#

            #                    # Map the function to the pool of workers
            #                    for results in executor.map(partial_split_segmentation_files, args, chunksize=self.n_cpus):
            #                        pbar.update(1)
            #                        if results is None:
            #                            self.error.error(f"Failed splitting segmentation file {path}")
            #                            print(f"Failed splitting segmentation file {path}")
            #                            # summary = pd.concat([summary, results], ignore_index=True)

            #                    # Perform cleanup after processing the batch
            #                    gc.collect()

            #    del cleaned_segs
            #    del summary
            # Give summary information after filtering
            # Number of ROIs, Connected components, Segmentation files, Segmentation labels, Empty segmentations
            self.logger.info("Summary of Segmentation Filtering: \n" +
                             "\t\t\t\t\t\t\t\t\t\t\t\t\t\tNumber of empty Segmentations: " + str(
                len(self.empty_paths)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\t\t\t\t\tNumber of connected Components: " + str(
                self.connected_compenents_sum) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\t\t\t\t\tNumber of Segmentations with more than {} connected Components: ".format(
                                 str(self.max_num_rois)) +
                             str(len(self.segmentations_over_max_num_rois)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\t\t\t\t\tSegmentations with Components smaller than {} Voxles: ".format(
                                 str(self.roi_threshold)) +
                             str(len(self.segmentations_under_roi_threshold)) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\t\t\t\t\tRemoved Components (smaller than {} voxels or more than {} components): ".format(
                                 str(self.roi_threshold), str(self.max_num_rois)) +
                             str(self.removed_small_rois_sum) + "\n" +
                             "\t\t\t\t\t\t\t\t\t\t\t\t\t\tSegmentations with multiple labels: " +
                             str(len(self.segmentations_with_multiple_labels)) + "\n")

            if os.path.isfile(self.out_path + "/" + self.runid + "_seg_parameters.csv"):
                previous_summary = pd.read_csv(self.out_path + "/" + self.runid + "_seg_parameters.csv", sep=',')
                #summary = pd.concat([previous_summary, summary], ignore_index=True)
                #summary.drop_duplicates(inplace=True)


                #summary.to_csv(self.out_path + "/" + self.runid + "_seg_parameters.csv",
                #               mode='a',
                #               index=False,
                #               header=False)
            #else:
            #summary.to_csv(self.out_path + "/" + self.runid + "_seg_parameters.csv", index=False)
            #self.logger.info("Results in " + str(self.out_path + "/" + self.runid + "_seg_parameters.csv\n"))
        else:
            self.logger.info("No segmentations needed to get processed! Check filtering of previous run.")
            print("No segmentations needed to get processed! Check filtering of previous run.")

            if os.path.isfile(self.previous_output_path):
                df = pd.read_csv(self.previous_output_path, sep=',', header=0, index_col=False)
                self.df = pd.concat([self.df, df], verify_integrity=True, ignore_index=True)

        return self.df

class SegmentationReader:
    def __init__(self,
                 logger,
                 list_of_seg_paths=None
                 ):

        if list_of_seg_paths is None:
            list_of_seg_paths = []

        self.list_of_seg_paths = list_of_seg_paths
        self.logger = logger

    def loadSeg(self, seg_path):
        """
        Read segmentation file and convert to array
        :parameter: seg_path: str
        :return: ndarray
        """
        seg = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(seg)

        return seg_array

    def loadSegfromList(self):
        """
        Load segmentation from list of paths.
        return df and list of path of segs and list of seg arrays
        """

        seg = []

        for seg_i in tqdm.tqdm(self.list_of_seg_paths, desc='Reading Seg Files'):
            seg_arr = self.loadSeg(seg_i)
            seg.append(seg_arr)

        self.logger.info("Loaded " + str(len(seg)) + " segmentations")
        print("Loaded " + str(len(seg)) + " segmentations")

        return seg

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
