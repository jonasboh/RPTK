import numpy as np
import os
import tqdm
from multiprocessing.pool import Pool
import random
import sys

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
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import Normalizer
from skimage.segmentation import expand_labels
from statistics import mean

from pathlib import Path  # handle path dirs

import gzip
import shutil

# +
import pandas as pd
from image_processing.Transform_Executer import Executor
from rptk.mirp import *
from rptk.mirp.experimentClass import ExperimentClass
#from Resampler import Resampler
from rptk.mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

#from segmentation_processing.SegmentationFilter import SegmentationFilter, SegmentationReader
# TODO correct import statement for new path

# # + Modailty
# # + Label --> Needed
# # + ID --> default Image Name + Seg Name
# # + timepoint --> default 0
# # + Image configuration (Reconstructionkernel,Phantom,cropped Scan) --> default 0
# # + Seg Rator --> default: 0
# # + Object description --> default: 0

# +
class Perturbator:

    def __init__(self,
                segPath: str = "",
                imgPath: str = "",  # if single files or a folder with same modality
                modality: str = "",  # Modality if single img and mask are provided or multiple in folder
                outpath: str = "",
                chunksize: int = 10,  # Chunksize for multiprocessing
                n_cpu: int = os.cpu_count() - 1,  # CPU used for multi threading
                perturbation_roi_adapt_size = None,  # Perturbation size for roi
                perturbation_roi_adapt_type: str = "distance",  # "fraction"
                roi_pert_repetiton: int = 3,  # How many times the roi perturbation is repeated
                label_of_interest: int = 1,  # Label which will be considered for segmentation
                dice_threshold: float = 0.92,  # Dice threshold for segmentation acceptance
                peritumoral_seg: bool = False,  # Peritumoral segmentation (yes/no)
                peri_dist: int = 3,  # Distance for surrounding region: default 3mm
                expand_seg_dist = None,  # Distance for expand segmentation: default 1
                random_walker_min_change: int = -3,  # Random Walker min change: default -3
                random_walker_max_change: int = 3,  # Random Walker max change: default 3
                random_walker_iterations: int = 2,  # Random Walker iterations: default 2
                seg_closing_radius: int = 10,  # Segmentation closing radius: default 10
                seg_closing_iterations: int = 1,  # Segmentation closing iterations: default 1
                ):

        self.segPath = segPath
        self.imgPath = imgPath
        self.modality = modality
        self.outpath = outpath
        self.chunksize = chunksize
        self.n_cpu = n_cpu
        self.perturbation_roi_adapt_size = perturbation_roi_adapt_size
        self.perturbation_roi_adapt_type = perturbation_roi_adapt_type
        self.roi_pert_repetiton = roi_pert_repetiton
        self.label_of_interest = label_of_interest
        self.dice_threshold = dice_threshold
        self.peritumoral_seg = peritumoral_seg
        self.peri_dist = peri_dist
        self.expand_seg_dist = expand_seg_dist
        self.random_walker_min_change = random_walker_min_change
        self.random_walker_max_change = random_walker_max_change
        self.random_walker_iterations = random_walker_iterations
        self.seg_closing_radius = seg_closing_radius
        self.seg_closing_iterations = seg_closing_iterations


    def generate_expanded_seg(self, seg_path, expand_seg_dist=None):
        """
        Writes expanded Seg to folder
        """

        if expand_seg_dist is None:
            expand_seg_dist = []

        path2img = ""

        if len(expand_seg_dist) == 0:
            expand_seg_dist = self.expand_seg_dist

        sitk_img = sitk.ReadImage(seg_path)

        for dist in expand_seg_dist:
            expanded_seg = self.get_expanded_region(seg_path, dist)
            exp_seg_name = os.path.basename(seg_path)[:-7] + "_" + str(dist) + "_expanded.nii.gz"

            if not os.path.isdir(self.output + "/pertub_seg/"):
                os.mkdir(self.output + "/pertub_seg/")

            expanded_seg = sitk.GetImageFromArray(expanded_seg.astype(np.uint16))
            expanded_seg.CopyInformation(sitk_img)

            if (np.sum(sitk.GetArrayFromImage(expanded_seg)) <= 0):
                self.logger.warning("Empty Expanded Segmentation:", exp_seg_name)
            else:
                sitk.WriteImage(expanded_seg,
                                os.path.join(self.output + "/pertub_seg/", exp_seg_name),
                                useCompression=True)

        expanded_seg_name = os.path.join(self.output + "/pertub_seg/", exp_seg_name)
        prefix = os.path.basename(expanded_seg_name).split("_")[0]

        self.logger.info("Calculate extended Segmentation for " + prefix)

        # Segmentation permutations from expanded segmentations

        # find images to extended segmentation for pertubation
        for img, mod in zip(self.df["Image"].values, self.df["Modality"].values):
            if os.path.basename(img).startswith(prefix):
                path2img = img
                modality = mod
                break

        if path2img != "":
            self.make_roi_pertubation(
                df=pd.DataFrame(),
                out_path=self.output + "/pertub_seg/",
                modality=modality,
                path2img=path2img,
                path2msk=expanded_seg_name,
                perturbation_roi_adapt_size=self.perturbation_roi_adapt_size,
                perturbation_roi_adapt_type=self.perturbation_roi_adapt_type,
                roi_pert_repetiton=self.roi_pert_repetiton
            )
        else:
            self.logging.warning("No Pertubation possible for " + str(prefix))

    def get_peritumoral_region(self, seg_path, dist):
        """
        extract surrounding region of size dist mm of binary segmantation with label 1
        :param name:
        :return:
        """

        sitk_img = sitk.ReadImage(seg_path)
        seg = sitk.GetArrayFromImage(sitk_img)

        expanded = get_expanded_region(seg_path, dist)

        peritumoral = expanded - seg
        peritumoral = sitk.GetImageFromArray(peritumoral.astype(np.uint16))
        peritumoral.CopyInformation(sitk_img)

        return peritumoral

    def generate_peritumoral_seg(self, seg_path):
        """
        Writes peritumoral Seg to folder
        """

        peri_seg = self.get_peritumoral_region(seg_path, self.peri_dist)
        peri_seg_name = os.path.basename(seg_path)[:-len(".nii.gz")] + "_surrounding.nii.gz"

        if not os.path.isdir(self.output + "/surrounding_seg/"):
            os.mkdir(self.output + "/surrounding_seg/")

        if (np.sum(sitk.GetArrayFromImage(peri_seg)) <= 0):
            self.logger.warning("Empty Peritumoral Segmentation:", peri_seg_name)
        else:
            sitk.WriteImage(peri_seg,
                            os.path.join(self.output + "/surrounding_seg/", peri_seg_name),
                            useCompression=True)

    def make_roi_pertubation(self,
                             out_path: str,
                             df: pd.DataFrame(),
                             path2csv: str = "",
                             modality: str = "CT",
                             path2img: str = "",  ## if no df is given can define a single image and mask
                             path2msk: str = "",  ## if no df is given can define a single image and mask
                             perturbation_roi_adapt_size: list = [-2.0, +2.0],
                             perturbation_roi_adapt_type: str = "distance",
                             roi_pert_repetiton: int = 3,
                             ):

        """
        Generate randomized masks
        path2csv
        out_path
        perturbation_roi_adapt_size
        perturbation_roi_adapt_type
        n_cpus
        """

        if len(df) == 0:
            if path2csv != "":
                df = pd.read_csv(path2csv)

        general_settings = GeneralSettingsClass(
            by_slice=True
        )

        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=True,
            interpolate=False,
            anti_aliasing=False
        )

        perturbation_settings = ImagePerturbationSettingsClass(
            crop_around_roi=False,
            perturbation_randomise_roi_repetitions=roi_pert_repetiton,
            perturbation_roi_adapt_size=perturbation_roi_adapt_size,
            perturbation_roi_adapt_type=perturbation_roi_adapt_type
        )

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
        if modality == "CT":
            resegmentation_settings = ResegmentationSettingsClass(resegmentation_method="threshold",
                                                                  resegmentation_intensity_range=[-1000.0, -900.0])
        elif modality == "MR":
            resegmentation_settings = ResegmentationSettingsClass(resegmentation_method="range",
                                                                  resegmentation_sigma=3.0)
        else:
            raise ValueError("Image perturbation setting not defined. Modality not supported!")

        settings = SettingsClass(
            general_settings=general_settings,
            post_process_settings=ImagePostProcessingClass(),
            img_interpolate_settings=image_interpolation_settings,
            roi_interpolate_settings=RoiInterpolationSettingsClass(roi_spline_order=1),
            roi_resegment_settings=resegmentation_settings,
            perturbation_settings=perturbation_settings,
            img_transform_settings=image_transformation_settings,
            feature_extr_settings=feature_computation_parameters
        )

        out_files = glob.glob(os.path.join(out_path, "*.nii.gz"))

        skipping_files = []

        experiments = []
        if len(df) != 0:

            for img_path in df["Image"].values:
                img_file = os.path.basename(img_path)[:-len(".nii.gz")]
                for out_f in out_files:
                    out_file = os.path.basename(out_f)
                    if out_file.startswith(img_file):
                        skipping_files.append(img_path)
                        break

            for img_path, seg_path in zip(df["Image"], df["Mask"]):

                if "Modality" in df.columns:
                    modality = df.loc[df["Image"] == img_path, "Modality"].values[0]

                img_file = os.path.basename(img_path)
                seg_file = os.path.basename(seg_path)
                if img_path in skipping_files:
                    self.logger.info("Skipping Pertubation for " + os.path.basename(img_path))
                    continue
                else:
                    experiment = ExperimentClass(
                        modality=modality,
                        subject=img_file[:-len(".nii.gz")],
                        cohort=None,
                        write_path=out_path,
                        image_folder=os.path.dirname(img_path),
                        roi_folder=os.path.dirname(seg_path),
                        roi_reg_img_folder=None,
                        image_file_name_pattern=img_file[:-len(".nii.gz")],
                        registration_image_file_name_pattern=None,
                        roi_names=[seg_file[:-len(".nii.gz")]],
                        data_str=[""],
                        provide_diagnostics=False,
                        settings=settings,
                        compute_features=False,
                        extract_images=True,
                        plot_images=False,
                        keep_images_in_memory=False
                    )

                    experiments.append(experiment)
        else:
            img_path = path2img
            seg_path = path2msk

            img_file = os.path.basename(img_path)
            seg_file = os.path.basename(seg_path)

            for out_f in out_files:
                out_file = os.path.basename(out_f)
                if out_file.startswith(img_file):
                    skipping_files.append(img_path)
                    break

            if img_path in skipping_files:
                self.logger.info("Skipping Perturbation for " + os.path.basename(img_path))

            else:
                experiment = ExperimentClass(
                    modality=modality,
                    subject=img_file[:-len(".nii.gz")],
                    cohort=None,
                    write_path=out_path,
                    image_folder=os.path.dirname(img_path),
                    roi_folder=os.path.dirname(seg_path),
                    roi_reg_img_folder=None,
                    image_file_name_pattern=img_file[:-len(".nii.gz")],
                    registration_image_file_name_pattern=None,
                    roi_names=[seg_file[:-len(".nii.gz")]],
                    data_str=[""],
                    provide_diagnostics=False,
                    settings=settings,
                    compute_features=False,
                    extract_images=True,
                    plot_images=False,
                    keep_images_in_memory=False
                )

                experiments.append(experiment)

        self.experiments = experiments
        if len(self.experiments) != 0:
            # Only use multi threading when list ist longer
            if len(self.experiments) > 1:
                p = Pool(self.n_cpu)

                p.map(exe, self.experiments, chunksize=self.chunksize)
                p.close()
                p.join()
            else:
                exe(self.experiments[0])

    def get_expanded_region(self, man_seg_path, dist):
        """
        return expanded region of size dist mm of binary segmentation with label 1
        :param name:
        :return:
        """
        sitk_img = sitk.ReadImage(man_seg_path)
        seg = sitk.GetArrayFromImage(sitk_img)

        labels = np.unique(seg)

        # if there is any other label than 1: set it to 0
        if len(labels) > 2:
            # set all labels which are bigger then 1 to 0
            seg[seg > 1] = 0
            # set all labels which are not 0 to 1
            seg[seg != 0] = 1

        expanded = expand_labels(seg, distance=dist)

        return expanded

    def calc_dice4control(self, seg_path, path_2_seg_folder):
        """
        Calculated Dice and only copies file if dice is above self.dice_threshold
        """

        seg_ = sitk.ReadImage(seg_path)
        y_true = sitk.GetArrayFromImage(seg_)

        out_acc_seg = self.output + "/acc_pertub_seg/"

        if not os.path.isdir(out_acc_seg):
            os.mkdir(out_acc_seg)

        pert_seg = glob.glob(path_2_seg_folder)
        prefix = os.path.basename(seg_path)[:-len(".nii.gz")]

        for done_seg in glob.glob(out_acc_seg + "/*.nii.gz"):
            if prefix in os.path.basename(done_seg):
                for seg in pert_seg:
                    if prefix in os.path.basename(seg):
                        pert_seg.remove(seg)
                        self.logger.info("Perturbed Segmentation " + str(os.path.basename(seg)) + " already accepted")
                        break

        self.logger.info("Calculate Dice for " + prefix)
        for seg in pert_seg:
            if prefix in os.path.basename(seg):

                seg_ = sitk.ReadImage(seg)
                y_pred = sitk.GetArrayFromImage(seg_)

                ddice = self.dice_coeff(y_true, y_pred)

                if round(ddice, 2) >= self.dice_threshold:
                    new_file_name = os.path.basename(seg)[:-(len(".nii.gz"))] + "_dice_" + str(
                        round(ddice, 2)) + ".nii.gz"
                    shutil.copy2(seg, out_acc_seg + "/" + new_file_name)
                    self.logger.info(
                        "Perturbed Segmentation " + str(os.path.basename(seg)) + " accepted with dice " + str(
                            round(ddice, 2)))
                else:
                    self.logger.info(
                        "Perturbed Segmentation " + str(os.path.basename(seg)) + " NOT accepted with dice " + str(
                            round(ddice, 2)))

    def segmentation_closing(self, seg_path, radius):

        seg_ = sitk.ReadImage(seg_path)
        nda = sitk.GetArrayFromImage(seg_)

        closed_nda = np.ndarray(nda.shape)
        for i in range(nda.shape[0]):
            bnda = skimage.morphology.isotropic_closing(nda[i], radius=radius)
            closed_nda[i] = bnda

        sitk_img_out = sitk.GetImageFromArray(closed_nda.astype(np.uint16))
        sitk_img_out.CopyInformation(seg_)
        sitk.WriteImage(sitk_img_out,
                        seg_path[:-len(".nii.gz")] + "_closed.nii.gz",
                        useCompression=True)

        return closed_nda

    def segmentation_convex_hull(self, seg_path):

        seg_ = sitk.ReadImage(seg_path)
        nda = sitk.GetArrayFromImage(seg_)

        closed_nda = np.ndarray(nda.shape)
        for i in range(nda.shape[0]):
            bnda = skimage.morphology.convex_hull_object(nda[i])
            closed_nda[i] = bnda

        sitk_img_out = sitk.GetImageFromArray(closed_nda.astype(np.uint16))
        sitk_img_out.CopyInformation(seg_)
        sitk.WriteImage(sitk_img_out,
                        seg_path[:-len(".nii.gz")] + "_convex.nii.gz",
                        useCompression=True)

        return closed_nda

    def dice_coeff(self, y_true, y_pred):
        """
        Computes soerensen-dice coefficient.

        compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred`.

        Args:
         y_true: 3-dim Numpy array of type bool. The ground truth mask.
         y_pred: 3-dim Numpy array of type bool. The predicted mask.

        Returns:
         the dice coeffcient as float. If both masks are empty, the result is NaN.
        """

        volume_sum = y_true.sum() + y_pred.sum()

        if volume_sum == 0:
            return np.NaN

        volume_intersect = (y_true & y_pred).sum()

        return 2 * volume_intersect / volume_sum

    def random_walker_exe(self, segs: list):
        """
        Execute random segmentation change for one segmentation
        """

        # segs = glob.glob(self.output + "/multilabel_seg/*.nii.gz")

        p = Pool(self.n_cpu)
        p.map(self.random_walker, segs, chunksize=self.chunksize)
        p.close()
        p.join()

    def random_walker(self, seg_path):
        """
        Random Walker segmentation perturbation in all 3 directions (x,y,z)
        """
        self.logger.info("Random Walker for " + str(os.path.basename(seg_path)))
        for i in range(self.random_walker_iterations):

            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path, sitk.sitkFloat32))
            new_seg = seg.copy()

            for slice_ in tqdm.tqdm(range(len(seg)), desc='Random Walker segmentation perturbation x,y of ' + str(os.path.basename(seg_path))):

                # horizontal
                new_seg[slice_] = self.random_walker_seg_change(seg[slice_], new_seg[slice_])

                # vertical
                output = self.random_walker_seg_change(seg[slice_].T, new_seg[slice_].T)
                new_seg[slice_] = output.T

            for slice_ in tqdm.tqdm(range(len(seg.T)), desc='Random Walker segmentation perturbation z of ' + str(os.path.basename(seg_path))):
                # z-axis
                output = self.random_walker_seg_change(seg.T[slice_], new_seg.T[slice_])
                new_seg.T[slice_] = output

            output = self.output + "/pertub_seg/"

            # generate file name
            transformed_file_name = os.path.basename(seg_path)[:-len(".nii.gz")] + "_random_walker_" + str(
                i) + ".nii.gz"

            # write file
            self.write_nifti_file(new_seg, output + transformed_file_name, seg_path)

    def random_walker_seg_change(self, input_, output_):

        for row_ in range(len(input_)):

            # Go horizontal
            for i in range(len(input_[row_])):

                if i < len(input_[row_]):

                    if np.sum(input_[row_][i]) > 0:

                        # Get start of island
                        change = random.randint(self.random_walker_min_change, self.random_walker_max_change)

                        # Start is on the endge
                        if i == 0:

                            # Get End
                            for j in range(len(input_[row_][i + 1:])):
                                if np.sum(input_[row_][i + 1 + j]) == 0:
                                    if change > 0:
                                        if (i + 1 + j + change) < len(input_[row_]):
                                            for ch in range(change):
                                                output_[row_][i + 1 + j + ch] = 1.

                                            break
                                        else:
                                            for ch in range(len(input_[row_]) - (i + 1 + j)):
                                                output_[row_][i + 1 + j + ch] = 1.
                                            break
                                    else:
                                        if (i + 1 + j + change) < len(input_[row_]):
                                            for ch in range(change):
                                                output_[row_][i + 1 + j + ch] = 0.
                                            break
                                        else:
                                            for ch in range(len(input_[row_]) - (i + 1 + j)):
                                                output_[row_][i + 1 + j + ch] = 0.
                                            break

                            break
                        else:
                            # go back from start
                            if (i - change) > 0:
                                if change > 0:
                                    # add values
                                    for ch in range(change):
                                        output_[row_][i - ch] = 1.
                                else:
                                    # remove values
                                    for ch in range(change):
                                        output_[row_][i - ch] = 0.

                            else:
                                if change > 0:
                                    for ch in range(change):
                                        output_[row_][i - ch] = 1.
                                else:
                                    for ch in range(change):
                                        output_[row_][i - ch] = 0.

                                        # go to end of island
                            for j in range(len(input_[row_][i + 1:])):
                                if np.sum(input_[row_][i + 1 + j]) == 0:
                                    if change > 0:
                                        if (i + 1 + j + change) < len(input_[row_]):
                                            for ch in range(change):
                                                output_[row_][i + 1 + j + ch] = 1.
                                            break
                                        else:
                                            for ch in range(len(input_[row_]) - (i + 1 + j)):
                                                output_[row_][i + 1 + j + ch] = 1.
                                            break
                                    else:
                                        if (i + j + change) < len(input_[row_]):
                                            for ch in range(change):
                                                output_[row_][i + j + ch] = 0.
                                            break
                                        else:
                                            for ch in range(len(input_[row_]) - (i + 1 + j)):
                                                output_[row_][i + j + ch] = 0.
                                            break
                            break
        return output_
