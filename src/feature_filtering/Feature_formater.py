import pandas as pd
import numpy as np
import logging
import tqdm
from matplotlib import pyplot as plt

from rptk.src.config.Log_generator_config import LogGenerator


class FeatureFormatter:
    """
    Structure Feature space and separate features in feature classes
    :parameter
    """

    def __init__(self,
                 logger=None,
                 error=None,
                 features: list = None,
                 kernels_in_files: list = None,
                 extractor: str = "MIRP",
                 output_path: str = None,
                 additional_ROIs: list = [],  # list of stings containing names of the ROIs in the feature names
                 generate_feature_profile_plot: bool = True,
                 RunID: str = "RPTK",
                 non_RPTK_format:bool = False,
                 no_clinical_included:bool = True):

        self.format_columns = ["ID",
                               "Image",
                               "Mask",
                               "Mask_Transformation",
                               "Image_Transformation",
                               "Prediction_Label",
                               "Modality",
                               "ROI_Label",
                               "Rater",
                               "Timepoint"]

        if kernels_in_files is None:
            # self.kernels_in_feature_space = {"Wavelet-HHH": "Wavelet-HHH",
            #                                  "Wavelet-LHH": "Wavelet-LHH",
            #                                  "Wavelet-HLH": "Wavelet-HLH",
            #                                  "Wavelet-HHL": "Wavelet-HHL",
            #                                  "Wavelet-HLL": "Wavelet-HLL",
            #                                  "Wavelet-LHL": "Wavelet-LHL",
            #                                  "Wavelet-LLH": "Wavelet-LLH",
            #                                  "Wavelet-LLL": "Wavelet-LLL",
            #                                  "Square": "Square",
            #                                  "log-": "LoG",  # LoG
            #                                  "squareroot": "SquareRoot",
            #                                  "logarithm": "Logarithm",
            #                                  "exponential": "Exponential",
            #                                  "gradient": "Gradient",
            #                                  "lbp-2D": "LBP2D",
            #                                  "lbp-3D": "LBP3D",
            #                                  "laws": "Laws",
            #                                  "gabor": "Gabor",
            #                                  "gauss": "Gaussian",
            #                                  "wavelet": "Nonseparable_Wavelet",
            #                                  "separable_wavelet": "Separable_Wavelet",
            #                                  "mean": "Mean",
            #                                  "laplacian_of_gaussian": "Laplacian_of_Gaussian"}

            kernels_in_files = ["Wavelet-HHH",
                                "Wavelet-LHH",
                                "Wavelet-HLH",
                                "Wavelet-HHL",
                                "Wavelet-HLL",
                                "Wavelet-LHL",
                                "Wavelet-LLH",
                                "Wavelet-LLL",
                                "wavelet-HHH",
                                "wavelet-LHH",
                                "wavelet-HLH",
                                "wavelet-HHL",
                                "wavelet-HLL",
                                "wavelet-LHL",
                                "wavelet-LLH",
                                "wavelet-LLL",
                                "WaveletHHH",
                                "WaveletLHH",
                                "WaveletHLH",
                                "WaveletHHL",
                                "WaveletHLL",
                                "WaveletLHL",
                                "WaveletLLH",
                                "WaveletLLL",
                                "LoG",
                                "SquareRoot",
                                "LBP2D",
                                "LBP3D",
                                "Square",
                                "Logarithm",
                                "Gradient",
                                "Exponential",
                                "log-",
                                "squareroot",
                                "square",
                                "logarithm",
                                "exponential",
                                "gradient",
                                "lbp-2D",
                                "lbp-3D",
                                "laws",
                                "gabor",
                                "gauss",
                                "wavelet",
                                "separable",
                                "mean",
                                "log"]
            
        self.logger = logger
        self.features = features
        self.kernels_in_files = kernels_in_files
        self.extractor = extractor
        self.error = error
        self.output_path = output_path
        self.additional_ROIs = additional_ROIs
        self.generate_feature_profile_plot = generate_feature_profile_plot
        self.RunID = RunID
        self.non_RPTK_format = non_RPTK_format
        self.no_clinical_included = no_clinical_included

        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.output_path + "/RPTK_feature_profiling_" + self.RunID + ".log",
                logger_topic="RPTK Feature Profiling"
            ).generate_log()

        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.output_path + "/RPTK_feature_profiling_" + self.RunID + ".err",
                logger_topic="RPTK Feature Profiling error"
            ).generate_log()

        if self.features is None:
            self.error.error("No Features provided for profiling!")
            raise ValueError("No Features provided for profiling!")
        
        self.mirp_feature_class_IDs = ["morphological",
                                       "local_intensity",
                                       "intensity-based_statistics",
                                       "intensity-volume_histogram",
                                       "intensity_histogram",
                                       "grey_level_co-occurrence_matrix",
                                       "grey_level_run_length_matrix",
                                       "grey_level_size_zone_matrix",
                                       "grey_level_distance_zone_matrix",
                                       "neighbourhood_grey_tone_difference_matrix",
                                       "neighbouring_grey_level_dependence_matrix",
                                       "diagnostic"]

        self.pyradiomics_feature_class_IDs = ["diagnostics",
                                        "morphological",
                                        "firstorder",
                                        "grey_level_co-occurrence_matrix",
                                        "grey_level_distance_zone_matrix",
                                        "grey_level_run_length_matrix",
                                        "grey_level_size_zone_matrix",
                                        "neighbourhood_grey_tone_difference_matrix",
                                        ]

        self.py_feature_class_ID = []

    
    def get_feature_class(self):
        
        if self.extractor == "MIRP":
            return self.mirp_feature_class_IDs
        elif self.extractore == "PyRadiomics":
            return self.py_feature_class_IDs
        else:
            self.logger.warning("Extractor not integrated yet!")
            return None

    def get_pyradiomics_features(self, feature_class):

        if feature_class == "diagnostics":
            diagnostics = ["Versions_PyRadiomics",
                           "Versions_Numpy",
                           "Versions_SimpleITK",
                           "Versions_PyWavelet",
                           "Versions_Python",
                           "Configuration_Settings",
                           "Configuration_EnabledImageTypes",
                           "Image-original_Hash",
                           "Image-original_Dimensionality",
                           "Image-original_Spacing",
                           "Image-original_Size",
                           "Image-original_Mean",
                           "Image-original_Minimum",
                           "Image-original_Maximum",
                           "Mask-original_Hash",
                           "Mask-original_Spacing",
                           "Mask-original_Size",
                           "Mask-original_BoundingBox",
                           "Mask-original_VoxelNum",
                           "Mask-original_VolumeNum",
                           "Mask-original_CenterOfMassIndex",
                           "Mask-original_CenterOfMass",
                           "Mask-resegmented_Spacing",
                           "Mask-resegmented_Size",
                           "Mask-resegmented_BoundingBox",
                           "Mask-resegmented_VoxelNum",
                           "Mask-resegmented_VolumeNum",
                           "Mask-resegmented_CenterOfMassIndex",
                           "Mask-resegmented_CenterOfMass",
                           "Mask-resegmented_Mean",
                           "Mask-resegmented_Minimum",
                           "Mask-resegmented_Maximum"]
            # TODO: remove all feature classes from feature names!!
            return diagnostics

        elif feature_class == "morphological":
            morphological = ["Elongation",
                             "Flatness",
                             "LeastAxisLength",
                             "MajorAxisLength",
                             "Maximum2DDiameterColumn",
                             "Maximum2DDiameterRow",
                             "Maximum2DDiameterSlice",
                             "Maximum3DDiameter",
                             "MeshVolume",
                             "MinorAxisLength",
                             "Sphericity",
                             "SurfaceArea",
                             "SurfaceVolumeRatio",
                             "VoxelVolume"]

            return morphological

        elif feature_class == "firstorder":
            firstorder = ["10Percentile",
                          "90Percentile",
                          "Energy",
                          "Entropy",
                          "InterquartileRange",
                          "Kurtosis",
                          "Maximum",
                          "Mean",
                          "MeanAbsoluteDeviation",
                          "Median",
                          "Minimum",
                          "Range",
                          "RootMeanSquared",
                          "Skewness",
                          "TotalEnergy",
                          "Uniformity",
                          "Variance"]

            return firstorder

        elif feature_class == "grey_level_co-occurrence_matrix":
            grey_level_co_occurrence_matrix = ["Autocorrelation",
                                               "ClusterProminence",
                                               "ClusterShade",
                                                "ClusterTendency",
                                                "Contrast",
                                                "Correlation",
                                                "DifferenceAverage",
                                                "DifferenceEntropy",
                                                "DifferenceVariance",
                                                "Id",
                                                "Idm",
                                                "Idmn",
                                                "Idn",
                                                "Imc1",
                                                "Imc2",
                                                "InverseVariance",
                                               "JointAverage",
                                               "JointEnergy",
                                               "JointEntropy",
                                               "MCC",
                                               "MaximumProbability",
                                               "SumAverage",
                                               "SumEntropy",
                                               "SumSquares"]

            return grey_level_co_occurrence_matrix

        elif feature_class == "grey_level_run_length_matrix":
            grey_level_run_length_matrix = ["GrayLevelNonUniformity",
                                            "GrayLevelNonUniformityNormalized",
                                            "GrayLevelVariance",
                                            "HighGrayLevelRunEmphasis",
                                            "LongRunEmphasis",
                                            "LongRunHighGrayLevelEmphasis",
                                            "LongRunLowGrayLevelEmphasis",
                                            "LowGrayLevelRunEmphasis",
                                            "RunEntropy",
                                            "RunLengthNonUniformity",
                                            "RunLengthNonUniformityNormalized",
                                            "RunPercentage",
                                            "RunVariance",
                                            "ShortRunEmphasis",
                                            "ShortRunHighGrayLevelEmphasis",
                                            "ShortRunLowGrayLevelEmphasis"]

            return grey_level_run_length_matrix

        elif feature_class == "grey_level_size_zone_matrix":
            grey_level_size_zone_matrix = ["GrayLevelNonUniformity",
                                            "GrayLevelNonUniformityNormalized",
                                             "GrayLevelVariance",
                                             "HighGrayLevelZoneEmphasis",
                                             "LargeAreaEmphasis",
                                             "LargeAreaHighGrayLevelEmphasis",
                                             "LargeAreaLowGrayLevelEmphasis",
                                             "LowGrayLevelZoneEmphasis",
                                             "SizeNonUniformity",
                                             "SizeNonUniformityNormalized",
                                             "SizeZoneNonUniformityNormalized",
                                             "SizeZoneNonUniformity",
                                             "SizeVariance",
                                             "SmallAreaEmphasis",
                                             "SmallAreaHighGrayLevelEmphasis",
                                             "SmallAreaLowGrayLevelEmphasis",
                                             "ZoneEntropy",
                                             "ZonePercentage",
                                             "ZoneVariance"]

            return grey_level_size_zone_matrix

        elif feature_class == "neighbourhood_grey_tone_difference_matrix":
            neighbouring_grey_tone_difference_matrix = ["Busyness",
                                                        "Coarseness",
                                                        "Complexity",
                                                        "Contrast",
                                                        "Strength"]

            return neighbouring_grey_tone_difference_matrix

        elif feature_class == "grey_level_distance_zone_matrix":
            grey_level_dependence_matrix = ["DependenceEntropy",
                                            "DependenceNonUniformity",
                                            "DependenceNonUniformityNormalized",
                                            "DependenceVariance",
                                            "GrayLevelNonUniformity",
                                            "GrayLevelVariance",
                                            "HighGrayLevelEmphasis",
                                            "LargeDependenceEmphasis",
                                            "LargeDependenceHighGrayLevelEmphasis",
                                            "LargeDependenceLowGrayLevelEmphasis",
                                            "LowGrayLevelEmphasis",
                                            "SmallDependenceEmphasis",
                                            "SmallDependenceHighGrayLevelEmphasis",
                                            "SmallDependenceLowGrayLevelEmphasis"]

            return grey_level_dependence_matrix

    def check_pyradiomics_feature_format(self, input_features: list):

        format = False

        first_class = []
        feature_class = []
        features = []
        morph = []
        loc = []
        stat = []
        #ivh = []
        #ih = []
        gcm = []
        grlm = []
        gszm = []
        gdzm = []
        ngt = []
        #ngl = []
        #diag = []
        #img_data = []

        tmp = pd.DataFrame()
        feature_format = pd.DataFrame()

        for col in input_features:
            if "ID." in col:
                continue
            if col != "Image" and col != "Mask":
                if self.non_RPTK_format:
                    if "_" in col:
                        splitted_ID = col.split("_")
                        kernel = splitted_ID[0]
                        feature = splitted_ID[2]
                    else:
                        self.logger.info(f"Feature {col} does not follow feature format ...")
                        print(f"Feature {col} does not follow feature format ...")
                        continue
                else:
                    # check for clinical or other features
                    if "_" in col:
                        splitted_ID = col.split("_", 1)
                    else:
                        # might be clinical features
                        self.logger.info(f"Feature {col} does not follow feature format ...")
                        splitted_ID = col.split("_")
                        tmp = self.get_feature_format(feature_class="Clinical_feature",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                        feature_format = pd.concat([feature_format, tmp], ignore_index=True)
                        continue

                if splitted_ID[0] == "original" and splitted_ID[1].startswith("shape"):  # morphology
                    morph.append(splitted_ID[2:])

                    tmp = self.get_feature_format(feature_class="morphological",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "original" and splitted_ID[1].startswith("firstorder"):  # local intensity
                    loc.append(splitted_ID[2:])
                    tmp = self.get_feature_format(feature_class="firstorder",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "diagnostics":  # intensity-based statistics
                    stat.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="diagnostics",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)



                elif splitted_ID[0] == "original" and splitted_ID[1].startswith("glcm"):  # grey level co-occurrence matrix
                    gcm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_co-occurrence_matrix",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "original" and splitted_ID[1].startswith("glrlm"):  # grey level run length matrix
                    grlm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_run_length_matrix",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "original" and splitted_ID[1].startswith("glszm"):  # grey level size zone matrix
                    gszm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_size_zone_matrix",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)
                    

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "original" and splitted_ID[1].startswith("gldm"):  # grey level distance zone matrix
                    gdzm.append(splitted_ID[1:])

                    tmp = self.get_feature_format(feature_class="grey_level_distance_zone_matrix",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "original" and splitted_ID[1].startswith("ngtdm"):  # neighbourhood grey tone difference matrix
                    ngt.append(splitted_ID[1:])

                    tmp = self.get_feature_format(feature_class="neighbourhood_grey_tone_difference_matrix",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                else:
                    format = False

                    for conf in self.format_columns:
                        if conf in col:
                            format = True
                            break

                    if self.no_clinical_included:
                        if not format:
                            if splitted_ID[0] != "index":
                                self.error.warning("Unknown feature class sorting as first class feature: " + splitted_ID[0] +
                                                " " + col)
                                print("Unknown feature class sorting as first class feature: " + splitted_ID[0] + " " + col)
                                first_class.append(splitted_ID[0])
                                if len(splitted_ID) > 1:
                                    feature_class.append(splitted_ID[1])
                                if len(splitted_ID) > 2:
                                    features.append(splitted_ID[2])
                            else:
                                # not processing index features as they are wrong!
                                continue


                        
                        
                    else:
                        splitted_ID = col.split("_")
                        tmp = self.get_feature_format(feature_class="Clinical_feature",
                                                splitted_feature_str=splitted_ID,
                                                feature_name=col)

                        feature_format = pd.concat([feature_format, tmp], ignore_index=True)

        return feature_format

    def get_mirp_features(self, feature_class):
        # IBSI/MIRP feature space
        # stat_mad_raw_intra_zscore

        if feature_class == "img_data":
            img_data = ["settings_id", "modality", "config", "noise_level", "noise_iter", "rotation_angle",
                        "roi_randomise_iter",
                        "roi_adapt_size", "translate_x", "translate_y", "translate_z", "voxel_size", "roi"]
            return img_data

        elif feature_class == "diagnostic":
            diagnostics = ["img_dim_x_init_img", "img_dim_y_init_img", "img_dim_z_init_img", "vox_dim_x_init_img",
                           "vox_dim_y_init_img", "vox_dim_z_init_img", "mean_int_init_img", "min_int_init_img",
                           "max_int_init_img", "int_map_dim_x_init_roi", "dint_map_dim_y_init_roi",
                           "int_map_dim_z_init_roi",
                           "int_bb_dim_x_init_roi", "int_bb_dim_y_init_roi", "int_bb_dim_z_init_roi",
                           "int_vox_dim_x_init_roi",
                           "int_vox_dim_y_init_roi", "int_vox_dim_z_init_roi", "int_vox_count_init_roi",
                           "int_mean_int_init_roi",
                           "int_min_int_init_roi", "int_max_int_init_roi", "mrp_map_dim_x_init_roi",
                           "mrp_map_dim_y_init_roi",
                           "mrp_map_dim_z_init_roi", "mrp_bb_dim_x_init_roi", "mrp_bb_dim_y_init_roi",
                           "mrp_bb_dim_z_init_roi",
                           "mrp_vox_dim_x_init_roi", "mrp_vox_dim_y_init_roi", "mrp_vox_dim_z_init_roi",
                           "mrp_vox_count_init_roi",
                           "mrp_mean_int_init_roi", "mrp_min_int_init_roi", "mrp_max_int_init_roi",
                           "img_dim_x_interp_img",
                           "img_dim_y_interp_img", "img_dim_z_interp_img", "vox_dim_x_interp_img",
                           "vox_dim_y_interp_img",
                           "vox_dim_z_interp_img", "mean_int_interp_img", "min_int_interp_img", "max_int_interp_img",
                           "int_map_dim_x_interp_roi",
                           "int_map_dim_y_interp_roi", "int_map_dim_z_interp_roi", "int_bb_dim_x_interp_roi",
                           "int_bb_dim_y_interp_roi",
                           "int_bb_dim_z_interp_roi", "int_vox_dim_x_interp_roi", "int_vox_dim_y_interp_roi",
                           "int_vox_dim_z_interp_roi",
                           "int_vox_count_interp_roi", "int_mean_int_interp_roi", "int_min_int_interp_roi",
                           "int_max_int_interp_roi",
                           "mrp_map_dim_x_interp_roi", "mrp_map_dim_y_interp_roi", "mrp_map_dim_z_interp_roi",
                           "mrp_bb_dim_x_interp_roi",
                           "mrp_bb_dim_y_interp_roi", "mrp_bb_dim_z_interp_roi", "mrp_vox_dim_x_interp_roi",
                           "mrp_vox_dim_y_interp_roi",
                           "mrp_vox_dim_z_interp_roi", "mrp_vox_count_interp_roi", "mrp_mean_int_interp_roi",
                           "mrp_min_int_interp_roi",
                           "mrp_max_int_interp_roi", "img_dim_x_reseg_img", "img_dim_y_reseg_img",
                           "img_dim_z_reseg_img", "vox_dim_x_reseg_img",
                           "vox_dim_y_reseg_img", "vox_dim_z_reseg_img", "mean_int_reseg_img", "min_int_reseg_img",
                           "max_int_reseg_img",
                           "int_map_dim_x_reseg_roi", "int_map_dim_y_reseg_roi", "int_map_dim_z_reseg_roi",
                           "int_bb_dim_x_reseg_roi",
                           "int_bb_dim_y_reseg_roi", "int_bb_dim_z_reseg_roi", "int_vox_dim_x_reseg_roi",
                           "int_vox_dim_y_reseg_roi",
                           "int_vox_dim_z_reseg_roi", "int_vox_count_reseg_roi", "int_mean_int_reseg_roi",
                           "int_min_int_reseg_roi",
                           "int_max_int_reseg_roi", "mrp_map_dim_x_reseg_roi", "mrp_map_dim_y_reseg_roi",
                           "mrp_map_dim_z_reseg_roi",
                           "mrp_bb_dim_x_reseg_roi", "mrp_bb_dim_y_reseg_roi", "mrp_bb_dim_z_reseg_roi",
                           "mrp_vox_dim_x_reseg_roi",
                           "mrp_vox_dim_y_reseg_roi", "mrp_vox_dim_z_reseg_roi", "mrp_vox_count_reseg_roi",
                           "mrp_mean_int_reseg_roi",
                           "mrp_min_int_reseg_roi", "mrp_max_int_reseg_roi"]
            return diagnostics

        elif feature_class == "morphological":
            morphology = ["volume", "vol_approx", "area_mesh", "av", "comp_1", "comp_2", "sph_dispr",
                          "sphericity", "asphericity", "com", "diam", "pca_maj_axis", "pca_min_axis", "pca_least_axis",
                          "pca_elongation", "pca_flatness", "area_dens_aee", "vol_dens_aabb", "area_dens_aabb", "vol_dens_aee", "vol_dens_conv_hull",
                          "area_dens_conv_hull", "integ_int", "moran_i", "geary_c"]
            return morphology

        elif feature_class == "local_intensity":
            local_intensity = ["peak_loc", "peak_glob"]
            return local_intensity

        elif feature_class == "intensity-based_statistics":
            intensity_based_statistics = ["mean", "var", "skew", "kurt", "median", "min", "p10", "p90", "max", "iqr",
                                          "range",
                                          "mad", "rmad", "medad", "cov", "qcod", "energy", "rms"]
            return intensity_based_statistics

        elif feature_class == "intensity_histogram":
            # dependent on the base_discretisation_bin_width in feature extraction --> postfix: fbs_w6.0
            intensity_histogram = ["mean", "var", "skew", "kurt", "median", "p10", "p90",
                                    "mode", "iqr", "range", "mad", "rmad", "medad", "cov",
                                   "qcod", "entropy", "uniformity","max_grad_g", "max_grad", "min_grad_g", "min_grad", "max", "min",
                                   ]
            return intensity_histogram

        # if feature_class == "ivh":
        # dependent on the ivh_discretisation_bin_width in feature extraction
        #    intensity_volume_histogram = ["v10", "v90", "i10", "i90", "diff_v10_v90", "diff_i10_i90"]

        elif feature_class == "intensity-volume_histogram":
            intensity_volume_histogram_mirp = ["v10", "v25", "v50", "v75", "v90", "i10", "i25", "i50",
                                               "i75", "i90", "diff_v10_v90", "diff_v25_v75",
                                               "diff_i10_i90", "diff_i25_i75", "auc"]
            return intensity_volume_histogram_mirp

        elif feature_class == "grey_level_co-occurrence_matrix":
            # dependent on the base_discretisation_bin_width and on glcm_spatial_method (dimensions to extract from) --> postfix d1_2d_avg_fbs_w6.0 d1_3d_avg_fbs_w6.0 d1_2d_s_mrg_fbs_w6.0 d1_3d_v_mrg_fbs_w6.0
            glcm = ["joint_max", "joint_avg", "joint_var", "joint_entr", "diff_avg", "diff_var", "diff_entr",
                    "sum_avg", "sum_var", "sum_entr", "energy", "contrast", "dissimilarity", 
                    "inv_diff_norm", "inv_diff_mom_norm", "inv_diff_mom",  "inv_diff", "inv_var", "corr", "auto_corr", "clust_tend", "clust_shade",
                    "clust_prom", "info_corr1", "info_corr2"]
            return glcm

        elif feature_class == "grey_level_run_length_matrix":
            # dependent on the base_discretisation_bin_width and on the glrlm_spatial_method
            glrlm = ["sre", "lre", "lgre", "hgre", "srlge", "srhge", "lrlge", "lrhge", "glnu_norm","glnu", 
                     "rlnu_norm", "rlnu",  "r_perc", "gl_var", "rl_var", "rl_entr"]
            return glrlm

        elif feature_class == "grey_level_size_zone_matrix":
            # dependent on the base_discretisation_bin_width and on the glszm_spatial_method
            glszm = ["sze", "lze", "lgze", "hgze", "szlge", "szhge", "lzlge", "lzhge", "glnu_norm", "glnu",  "zsnu_norm", "zsnu",
                     "z_perc", "gl_var", "zs_var", "zs_entr"]
            return glszm

        elif feature_class == "grey_level_distance_zone_matrix":
            # dependent on the base_discretisation_bin_width and on the gldzm_spatial_method
            gldzm = ["sde", "lde", "lgze", "hgze", "sdlge", "sdhge", "ldlge", "ldhge", "glnu_norm", "glnu", "zdnu_norm", "zdnu",
                     "z_perc", "gl_var", "zd_var", "zd_entr"]
            return gldzm

        elif feature_class == "neighbourhood_grey_tone_difference_matrix":
            # dependent on the base_discretisation_bin_width and on the ngtdm_spatial_method
            ngtdm = ["coarseness", "contrast", "busyness", "complexity", "strength"]
            return ngtdm

        elif feature_class == "neighbouring_grey_level_dependence_matrix":
            # dependent on the base_discretisation_bin_width and on the ngldm_spatial_method
            ngldm = ["lde", "hde", "lgce", "hgce", "ldlge", "ldhge", "hdlge", "hdhge", "glnu_norm", "glnu", "dcnu_norm", "dcnu",
                     "dc_perc", "gl_var", "dc_var", "dc_entr", "dc_energy"]
            return ngldm
        else:
            self.logger.info("Feature class " + feature_class + " unknown!")

        # mirp_features = img_data + diagnostics + morphology + local_intensity + intensity_based_statistics + intensity_histogram + intensity_volume_histogram + intensity_volume_histogram_mirp + glcm + glrlm + glszm + gldzm + ngtdm + ngldm

        # for mirp in mirp_features:
        #    if mirp in feature:
        #        return True, feature

        return None

    def get_feature_format(self, feature_class: str, splitted_feature_str, feature_name):
        """
        Get format from MIRP features
        :param feature_class: detected class of features
        :param splitted_feature_str: fetaure name splitted by "_"
        :param feature_name complete: name of the feature in the csv
        """

        tmp = pd.DataFrame()
        
        if self.extractor == "MIRP":
            for feature in self.get_mirp_features(feature_class):
                kernel_found = False
                search_str = None

                if "_" in feature:
                    if feature == splitted_feature_str[1]:
                        search_str = splitted_feature_str[1].replace(feature, '')
                    elif splitted_feature_str[1].startswith(feature):
                        search_str = splitted_feature_str[1].replace(feature, '')
                elif feature == splitted_feature_str[1].split("_", 1):
                    search_str = splitted_feature_str[1].replace(feature, '')
                elif splitted_feature_str[1].startswith(feature):
                    search_str = splitted_feature_str[1].replace(feature, '')

                if not search_str is None:
                    for kernel in self.kernels_in_files:
                        if kernel in search_str:
                            kernel_found = True
                            kernel_specs = kernel + splitted_feature_str[1].split(kernel, 1)[1]

                            if kernel_specs.endswith("_zscore"):
                                kernel_specs = kernel_specs[:-len("_zscore")]
    
                            tmp = pd.DataFrame({"Feature": [feature],
                                                "Feature_Class": [feature_class],
                                                "Image_Kernel": [kernel_specs],
                                                "Name": [feature_name]})
                            break

                    if not kernel_found:
                        tmp = pd.DataFrame({"Feature": [feature],
                                            "Feature_Class": [feature_class],
                                            "Image_Kernel": ["Original_Image"],
                                            "Name": [feature_name]})
                    break

        elif self.extractor == "PyRadiomics":
            if self.get_pyradiomics_features(feature_class) is not None:
                for feature in self.get_pyradiomics_features(feature_class):
                    kernel_found = False
                    search_str = None
                    
                    if "_" in feature:

                        if feature == splitted_feature_str[1]:
                            search_str = splitted_feature_str[1].replace(feature, '')
                        elif splitted_feature_str[1].startswith(feature):
                            search_str = splitted_feature_str[1].replace(feature, '')
                    elif feature == splitted_feature_str[1].split("_", 1)[-1]:
                        search_str = splitted_feature_str[1].replace(feature, '')
                    elif splitted_feature_str[1].startswith(feature):
                        search_str = splitted_feature_str[1].replace(feature, '')
                    else:
                        for split in splitted_feature_str[1].split("_"):
                            if split == feature:
                                search_str = splitted_feature_str[1].replace(feature, '')
                                break

                    if not search_str is None:

                        for kernel in self.kernels_in_files:

                            if kernel in search_str:
                                kernel_found = True
                                kernel_specs = kernel + splitted_feature_str[1].split(kernel, 1)[1]

                                if kernel_specs.endswith("_zscore"):
                                    kernel_specs = kernel_specs[:-len("_zscore")]

                                tmp = pd.DataFrame({"Feature": [feature],
                                                    "Feature_Class": [feature_class],
                                                    "Image_Kernel": [kernel_specs],
                                                    "Name": [feature_name]})
                                break

                        if not kernel_found:
                            tmp = pd.DataFrame({"Feature": [feature],
                                                "Feature_Class": [feature_class],
                                                "Image_Kernel": ["Original_Image"],
                                                "Name": [feature_name]})
                        #break

        if feature_class == "Clinical_feature":
            tmp = pd.DataFrame({"Feature": [feature_name],
                                "Feature_Class": [feature_class],
                                "Image_Kernel": ["Original_Image"],
                                "Name": [feature_name]})
        return tmp

    def check_mirp_feature_format(self, input_features:list):

        first_class = []
        feature_class = []
        features = []
        morph = []
        loc = []
        stat = []
        ivh = []
        ih = []
        gcm = []
        grlm = []
        gszm = []
        gdzm = []
        ngt = []
        ngl = []
        diag = []
        img_data = []

        tmp = pd.DataFrame()
        feature_format = pd.DataFrame()
        
        for col in input_features:
            if col != "Image" and col != "Mask":
                splitted_ID = col.split("_", 1)

                if splitted_ID[0] == "morph":  # morphology
                    morph.append(splitted_ID[1:])

                    tmp = self.get_feature_format(feature_class="morphological",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "loc":  # local intensity
                    loc.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="local_intensity",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "stat":  # intensity-based statistics
                    stat.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="intensity-based_statistics",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "ivh":  # intensity-volume histogram
                    ivh.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="intensity-volume_histogram",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "ih":  # intensity histogram
                    ih.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="intensity_histogram",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "cm":  # grey level co-occurrence matrix
                    gcm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_co-occurrence_matrix",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "rlm":  # grey level run length matrix
                    grlm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_run_length_matrix",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "szm":  # grey level size zone matrix
                    gszm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_size_zone_matrix",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "dzm":  # grey level distance zone matrix
                    gdzm.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="grey_level_distance_zone_matrix",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "ngt":  # neighbourhood grey tone difference matrix
                    ngt.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="neighbourhood_grey_tone_difference_matrix",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "ngl":  # neighbouring grey level dependence matrix
                    ngl.append(splitted_ID[1:])
                    tmp = self.get_feature_format(feature_class="neighbouring_grey_level_dependence_matrix",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif len(splitted_ID) > 1 and splitted_ID[0] == "img":
                    if str(splitted_ID[0] + "_" + splitted_ID[1]) == "img_data":  # image data
                        img_data.append(splitted_ID[2])
                        tmp = self.get_feature_format(feature_class="img_data",
                                                 splitted_feature_str=splitted_ID,
                                                 feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                elif splitted_ID[0] == "diag":  # diagnostic data
                    diag.append(splitted_ID[1])
                    tmp = self.get_feature_format(feature_class="diagnostic",
                                             splitted_feature_str=splitted_ID,
                                             feature_name=col)

                    feature_format = pd.concat([feature_format, tmp], ignore_index=True)

                else:
                    self.logger.info("Unknown feature class sorting as first class feature: " + splitted_ID[0] + " " + col)
                    first_class.append(splitted_ID[0])
                    if len(splitted_ID) > 1:
                        feature_class.append(splitted_ID[1])
                    if len(splitted_ID) > 2:
                        features.append(splitted_ID[2])

        return feature_format

    def generate_feature_profile(self, profile: pd.DataFrame, title: str, path: str):
        """
        Generate Profile for Features to get an overview about Class distributions in the dataset.
        """

        data = pd.DataFrame(index=list(set(profile["Feature_Class"])))
        for kernel in tqdm.tqdm(profile["Image_Kernel"], desc="Generating Feature Profile"):
            # count number of each feature class per image kernel
            feature_classes = profile.loc[profile["Image_Kernel"] == kernel, "Feature_Class"]
            count = feature_classes.value_counts(dropna=False)
            feature_class = count.index
            tmp = pd.DataFrame(data={kernel: count.values}, index=feature_class)
            data[kernel] = tmp

        data.plot.bar(stacked=True, colormap="rainbow_r")
        plt.legend(title="Kernels", loc='center left', bbox_to_anchor=(1.0, 0.5))

        plt.xlabel("Feature Classes")
        plt.ylabel("Number of Features")
        plt.title(title, color='black')
        plt.savefig(str(path) + '/' + str(title) + '.png',
                    bbox_inches='tight',
                    dpi=200)
        plt.close()

    def exe(self, title: str):
        """
        Main class for performing feature formating to get a feature structure.
        :param title: title for the feature profile generated
        :return:
        """

        # Format title
        title = title.replace("_", " ")

        features = pd.DataFrame()
        if len(self.features) > 0:
            input_features = self.features
        else:
            self.error.warning("Can not generate Feature Profile! Please check input data!")
            input_features = []

        # clean names
        input_features = [s.replace("_zscore", "") for s in input_features]

        for roi in self.additional_ROIs:
            if roi != "":
                input_features = [s.replace("_" + roi, "") for s in input_features]

        if self.extractor == "MIRP":
            features = self.check_mirp_feature_format(input_features=input_features)
            # features.to_csv(self.output_path + "/features.csv")
            for mirp_feature_class in self.mirp_feature_class_IDs:
                if "Feature_Class" in features.columns:
                    if mirp_feature_class not in features["Feature_Class"].to_list():
                        self.error.warning("Missing Feature Class: " + mirp_feature_class)
                        print("Missing Feature Class: " + mirp_feature_class)
                else:
                    if len(features) == 0:
                        self.error.warning("No Features for stability filtering!")
                        print("No Features for stability filtering!")
                        break

        elif self.extractor == "PyRadiomics":
            features = self.check_pyradiomics_feature_format(input_features=input_features)

            for pyradiomics_feature_class in self.pyradiomics_feature_class_IDs:
                if "Feature_Class" in features.columns:
                    if pyradiomics_feature_class not in set(features["Feature_Class"]):
                        self.error.warning("Missing Feature Class: " + pyradiomics_feature_class)
                        print("Missing Feature Class: " + pyradiomics_feature_class)
                else:
                    if len(features) == 0:
                        self.error.warning("No Features for stability filtering!")
                        print("No Features for stability filtering!")
                        break

        if len(features) != 0:
            if self.generate_feature_profile_plot:
                self.generate_feature_profile(profile=features, title=self.extractor + " " + title,
                                            path=self.output_path)

        else:
            self.error.warning("Feature Profile generation Failed! No features found with correct format.")
            print("Feature Profile generation Failed! No features found with correct format.")


        #for feature in features.columns:
        #    for format in self.format_columns:
        #        if feature.startswith(format):
        #            for kernel in self.kernels_in_files:
        #                if kernel in feature:
        #                    # drop column with config feature and transformation kernel
        #                    features.drop([feature], axis=1, inplace=True)
        #                    break

        return features
