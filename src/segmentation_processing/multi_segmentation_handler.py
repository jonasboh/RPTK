import pandas as pd
import os
import shutil
import glob
import re
import sys
import operator
import SimpleITK as sitk
import numpy as np
from skimage.measure import label, regionprops
import nibabel as nib

# +
# # + Modailty
# # + Label --> Needed
# # + ID --> default Image Name + Seg Name
# # + timepoint --> default 0
# # + Image configuration (Reconstructionkernel,Phantom,cropped Scan) --> default 0
# # + Seg Rator --> default: 0
# # + Object description --> default: 0

import logging
from tqdm import tqdm


class Multi_segmentation_handler:
    """
    Handle Segmentations with multiple ROIs but one label or segmentations with multiple labels.
    Separate these segmentations for radiomics feature extraction.
    """

    def __init__(self,
                 seg_path: str,
                 out_path: str,
                 roi_threshold: int,
                 prefix: str = "",
                 max_num_rois: int = 0,
                 logger: logging.Logger = logging.getLogger("Multi Segmentation Formatting")):

        self.seg_path = seg_path
        self.out_path = out_path
        self.roi_threshold = roi_threshold
        self.logger = logger
        self.max_num_rois = max_num_rois
        self.prefix = prefix

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        if not self.out_path.endswith('/'):
            self.out_path = self.out_path + "/"

    # 1. Read segmentation file
    # 2. Check for multi check for multiple objects
    # 3. Check for multi check for multiple labels
    # 4. Sort by area size
    # 5. Separate each area into a single segmentation
    # 6. Write each segmentation to file

    def read_segmentation(self):
        """
        Read a segmentation file in .nii.gz format and return the data array.
        :return: nadarray
        """

        seg = sitk.ReadImage(self.seg_path)
        seg_array = sitk.GetArrayFromImage(seg)
        # print("Original segmentation shape: ", seg_array.shape)

        return seg_array

    def count_rois(self, seg_array):
        """
        Count the number of ROIs in a segmentation file.
        :parameter: seg_array: ndarray
        :return: list: list of connected components
        :return: list: list of labels
        """

        list_of_labels = []
        
        # Count connected components in the segmentation
        connected_comp_labels, num_labels = label(seg_array, background=0.0, return_num=True)
        self.logger.info("Number of Connected components in {}: {}".format(os.path.basename(self.seg_path), num_labels))

        u_label = np.unique(seg_array)
        for label_ in u_label:
            if label_ != 0:
                list_of_labels.append(label_)

        self.logger.info("Number of labels in {}: {}".format(self.seg_path, len(list_of_labels)))

        return connected_comp_labels, list_of_labels

    def write_multi_label_data_to_csv(self):
        """
        Write the number of labels from a segmentation file to a csv file.
        :parameter: seg_array: ndarray
        :return: CSV file with the number of labels
        """

        # read segmentation
        seg_array = self.read_segmentation()
        sorted_filtered_connected_comp_labels = np.zeros(seg_array.shape)

        #print("raw_image shape:", seg_array.shape)
        #print("raw_image data type:", type(seg_array))
        #print("raw_image sum:", seg_array.sum())
        # get connected components and labels
        connected_comp_labels, list_of_labels = self.count_rois(seg_array)
        #print("connected_comp_labels data type:", type(connected_comp_labels), connected_comp_labels.shape)

        # get properties of connected components
        seg_props = regionprops(connected_comp_labels)
        #print("seg_props data type:", type(seg_props))
        #print("###################")
        connected_comp_labels_prop = []
        connected_comp_areas = []
        for prop in seg_props:
            connected_comp_labels_prop.append(prop.label)
            connected_comp_areas.append(prop.area)

        # sort connected components by area size

        filtered_array = self.ignore_small_segments(seg_array)
        filtered_connected_comp_labels, filtered_list_of_labels = self.count_rois(filtered_array)

        #print("filtered_array:", filtered_array.shape, type(filtered_array), filtered_array.sum())
        #print("filtered_connected_components:", type(filtered_connected_comp_labels), filtered_connected_comp_labels.shape)

        filtered_seg_props = regionprops(filtered_connected_comp_labels)

        filtered_connected_comp_labels_prop = []
        filtered_connected_comp_areas = []
        for prop in filtered_seg_props:
            filtered_connected_comp_labels_prop.append(prop.label)
            filtered_connected_comp_areas.append(prop.area)
        # TODO: check where the array gets the wrong shape
        #print("###################")

        sorted_rois, list_of_labels_sort = self.sort_rois(filtered_array)
        #print("sorted_rois:", type(sorted_rois), len(sorted_rois))
        sorted_filtered_connected_comp_labels_prop = []
        sorted_filtered_connected_comp_areas = []

        for array, lab in zip(sorted_rois, list_of_labels_sort):
            #print("sorted array shape:", array.shape)
            #print("sum of sorted array:", array.sum())
            #print("max of sorted array:", array.max())
            #print("sorted label:", list_of_labels, lab)
            sorted_filtered_connected_comp_labels, sorted_filtered_list_of_labels = self.count_rois(array)
            sorted_filtered_seg_props = regionprops(sorted_filtered_connected_comp_labels)
            #print("sorted_filtered_connected_comp_labels shape", sorted_filtered_connected_comp_labels.shape)
            #print("sorted_filtered_list_of_labels", sorted_filtered_list_of_labels)
            #sorted_filtered_connected_comp_labels_prop = []
            #sorted_filtered_connected_comp_areas = []
            #print("type sorted_filtered_seg_props", type(sorted_filtered_seg_props))
            for props in sorted_filtered_seg_props:
                sorted_filtered_connected_comp_areas.append(props.area)
                sorted_filtered_connected_comp_labels_prop.append(props.label)
                #print("sorted_filtered_seg_props:", props.label, props.area)
        #print("sorted_filtered_connected_comp_labels_prop", sorted_filtered_connected_comp_labels_prop)
        #print("###################")

        df = pd.DataFrame({"Segmentation": [self.seg_path],
                           "Shape_of_segmentation": [seg_array.shape],
                           "Size_of_segmentation": [seg_array.sum()],
                           "Number_of_labels": [len(list_of_labels)],
                           "Labels": [list_of_labels],

                           "Shape_of_connected_components": [connected_comp_labels.shape],
                           "Size_of_connected_components": [connected_comp_areas],
                           "Number_of_connected_components": [len(connected_comp_labels_prop)],
                           "Labels_Connected_component": [connected_comp_labels_prop],

                           "Shape_of_filtered_segmentation": [filtered_array.shape],
                           "Size_of_filtered_segmentation": [filtered_array.sum()],
                           "Number_of_filtered_labels": [len(filtered_list_of_labels)],
                           "Labels_filtered": [filtered_list_of_labels],

                           "Shape_of_filtered_connected_components": [filtered_connected_comp_labels.shape],
                           "Size_of_filtered_connected_components": [filtered_connected_comp_areas],
                           "Number_of_filtered_connected_components": [len(filtered_connected_comp_labels_prop)],
                           "Labels_filtered_connected_component": [filtered_connected_comp_labels_prop],

                           #"Sorted_connected_component_labels": [sorted_filtered_connected_comp_labels],
                           #"Sorted_size_of_connected_components": [sorted_filtered_connected_comp_areas],

                           "Shape_of_sorted_filtered_connected_components": [sorted_filtered_connected_comp_labels.shape
                                                                             ],
                           "Size_of_sorted_filtered_connected_components": [sorted_filtered_connected_comp_areas],
                           "Number_of_sorted_filtered_connected_components": [
                               len(sorted_filtered_connected_comp_labels_prop)],
                           "Labels_sorted_filtered_connected_component": [sorted_filtered_connected_comp_labels_prop],
                           })

        if not os.path.exists(self.out_path + "/" + self.prefix + '_multi_label_summary.csv'):
            df.to_csv(self.out_path + "/" + self.prefix + '_multi_label_summary.csv', index=False)
        else:
            df.to_csv(self.out_path + "/" + self.prefix + '_multi_label_summary.csv', mode='a', index=False, header=False)

    def multi_label(self, seg_array):

        """
        Check if the array has multiple labels.
        :parameter: seg_array: ndarray
        :return: bool
        """

        unique_labels = np.unique(seg_array)
        num_labels = len(unique_labels) - 1

        self.logger.info("File {} has {} labels".format(self.seg_path, num_labels))

        if num_labels > 1:
            return True
        else:
            return False

    def ignore_small_segments(self, seg_array):
        """
        Ignore small segments in a binary segmentation.

        Parameters
        ----------
        seg_array : ndarray
            Binary segmentation mask.
        roi_threshold: int
            Minimum size of a segment to be kept.

        Returns
        -------
        ndarray
            Binary segmentation mask with small segments removed.
        """

        labeled = label(seg_array)  # label connected regions
        seg_props = regionprops(labeled)  # extract properties of labeled regions

        # create a new binary mask with only regions larger than the threshold
        removed = 0
        new_mask = np.zeros(labeled.shape)

        for p in seg_props:
            if p.area >= int(self.roi_threshold):
                removed += 1
                new_mask[labeled == p.label] = p.label

        # print(new_mask.sum())

        self.logger.info(
            "Removed {} small segments from {} by threshold {}".format(removed, self.seg_path, self.roi_threshold))

        return new_mask

    def sort_rois(self, seg_array):
        """
        Gets a segmentation and returns a list of its ROIs sorted by size in descending order.

        Args:
            segmentation (ndarray): A 2D or 3D array representing the segmentation mask.
            roi_threshold (int): Minimum size of an ROI to be kept.

        Returns:
            list of ndarray: A list of 2D or 3D arrays, each representing an ROI in the segmentation mask.
                            The ROIs are sorted by size in descending order.
        """

        sorted_rois = []
        list_of_labels = []
        dict_label_area = {}

        connected_comp_labels, num_labels = label(seg_array, background=0, return_num=True)
        props = regionprops(connected_comp_labels)

        # Sort the labels by area size
        for i in range(num_labels):
            dict_label_area[props[i].label] = props[i].area

        sorted_d = dict(sorted(dict_label_area.items(), key=operator.itemgetter(1), reverse=True))

        if self.max_num_rois != 0:
            selected_labels = list(sorted_d.keys())[0:int(self.max_num_rois)]
        else:
            selected_labels = list(sorted_d.keys())

        self.logger.info("sorted labels: " + str(sorted_d))
        self.logger.info("selected labels with max area: " + str(selected_labels))


        # make an array for each selected label
        for i in range(num_labels):
            if len(selected_labels) > i:
                img_arr_out = np.zeros(connected_comp_labels.shape)
                img_arr_out[connected_comp_labels == selected_labels[i]] = 1
                list_of_labels.append(i)  # for segmentation naming
                sorted_rois.append(img_arr_out)

        return sorted_rois, list_of_labels

    def split_segmentation_file(self, seg_array):
        """
        Checks if a segmentation file in .nii.gz format has multiple ROIs with label 1.

        Args:
            filename (str): The path to the segmentation file.
            roi_threshold (int): The minimum size of an ROI to be included in the output.

        Returns:
            list of ndarray: A list of 3D arrays, each representing an ROI in the segmentation file.
        """

        # List for storing the ROIs
        rois = []

        # Check if the data array has multiple labels
        if self.multi_label(seg_array):
            connected_comp_labels, list_of_labels = self.count_rois(seg_array)
            # take the labels and separate them into individual ndarrays
            for lab in list_of_labels:
                tmp_array = np.zeros(seg_array.shape)
                tmp_array[seg_array == lab] = lab
                rois.append(tmp_array)

        # If data contains multiple ROIs with the same label
        else:
            sorted_rois, list_of_labels = self.sort_rois(seg_array)
            for roi in sorted_rois:
                # print(type(roi))
                rois.append(roi)

        return rois, list_of_labels

    def process(self):
        """"
        Process a segmentation file with multiple segmentations.
        :parameter: seg_path: str
        :return: list: list of filenames
        :return: list: list of arrays
        """

        # write a summary of multi-label data to CSV
        self.write_multi_label_data_to_csv()

        seg_array = self.read_segmentation()

        # Filter for small ROIs
        filtered_seg_array = self.ignore_small_segments(seg_array)

        list_of_components, list_of_labels = self.count_rois(seg_array)

        # If segmentation has multiple labels --> Labels might have meaning for the ROI - Do not change segmentation labels
        if self.multi_label(seg_array):
            # We have multiple labels but more components --> split the segmentation
            if len(list_of_components) > len(list_of_labels):
                list_of_arrays, list_of_labels = self.split_segmentation_file(seg_array)
        else:
            list_of_arrays, list_of_labels = self.sort_rois(filtered_seg_array)

        return list_of_labels, list_of_arrays

    def write_nifti_file(self, array, out_file_name):
        """
        Write a NIfTI file with the header information from an existing file.

        Parameters
        ----------
        array: numpy.ndarray
            The array is to be saved to a NIfTI file.
        header_file_path : str
            The path to the existing NIfTI file that contains the header information.
        output_file_path : str
            The path where the output NIfTI file should be saved.
        """
        # print("write_nifti_file")

        # Load the existing header information from the header file
        # header = nib.load(self.seg_path).header

        sitk_img = sitk.ReadImage(self.seg_path)

        if type(array) == list:
            for arr in array:
                sitk_img_out = sitk.GetImageFromArray(arr.astype(np.uint16))
                sitk_img_out.CopyInformation(sitk_img)
                self.logger.error("ERROR: Need to implement multiple filenames for multiple segmentations")
                # sitk.WriteImage(sitk_img_out, out_file_name) # Problem need multiple filenames for multiple segmentations

                # Create a new NIfTI image with the header information and the given array
                # nifti_image = nib.Nifti1Image(arr, affine=None) #, header=header)
                # print("save",output_file_path)

                # TODO coordinates are not matching correctly and segmentations seem to be separated
                # Save the new NIfTI image to the output file
                # nib.save(nifti_image, out_file_name)
        else:
            sitk_img_out = sitk.GetImageFromArray(array.astype(np.uint16))
            sitk_img_out.CopyInformation(sitk_img)
            sitk.WriteImage(sitk_img_out, out_file_name)

            # Create a new NIfTI image with the header information and the given array
            # nifti_image = nib.Nifti1Image(array, affine=None) #, header=header)
            # print("save",output_file_path)
            # Save the new NIfTI image to the output file
            # nib.save(nifti_image, out_file_name)
