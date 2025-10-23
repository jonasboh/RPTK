import pandas as pd
import numpy as np
import logging
import tqdm
import matplotlib.pyplot as plt
import os
import time
from typing import Union

from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.feature_filtering.Feature_formater import FeatureFormatter

class IBSIFeatureFormater:
    """
    Class to format the feature names to IBSI standard.
    """

    def __init__(self, 
                extractor: str,
                features : pd.DataFrame or list,
                profile: pd.DataFrame=None,
                logger: logging.Logger=None,
                error: logging.Logger=None,
                output_path: str=None,
                RunID: str=None
                ):
        """
        Initialize the IBSI feature formater class.
        parameters:
        -----------
        extractor: str
            Extractor used to extract the features. Either MIRP or PyRadiomics.
        features: pd.DataFrame
            DataFrame containing the features.
        profile: pd.DataFrame
            DataFrame containing the feature profile.
        logger: logging.Logger
            Logger object to log the information.
        error: logging.Logger                           
            Logger object to log the errors.    
        output_path: str
            Path to save the logs.  
        RunID: str
            ID of the run.
        """ 

        self.extractor = extractor
        self.features = features
        self.logger = logger
        self.error = error
        self.profile = profile
        self.output_path = output_path
        self.RunID = RunID

        if isinstance(self.features, list): # convert to dataframe if needed
            df = pd.DataFrame(columns=self.features)
            self.features = df

        if self.output_path is None:
            self.output_path = "."

        if self.output_path.endswith("/"):
            self.output_path = self.output_path[:-1]
        
        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path, exist_ok=True)

        if self.RunID is None:
            self.RunID = time.strftime("%Y%m%d-%H%M%S")

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

        if self.extractor not in ["MIRP", "PyRadiomics"]:
            self.error.error(f"Extractor should be either MIRP or PyRadiomics")
            raise ValueError("Extractor should be either MIRP or PyRadiomics")

        self.ibsi_features = []
        self.morphological_ibsi_translation= {"MIRP":[('volume', 'Volume'),
                                            ('vol_approx', 'Approximate Volume'),
                                            ('area_mesh', 'Surface Area'),
                                            ('av', 'Surface to volume ratio'),
                                            ('comp_1', 'Compactness 1'),
                                            ('comp_2', 'Compactness 2'),
                                            ('sph_dispr', 'Spherical disproportion'),
                                            ('sphericity', 'Sphericity'),
                                            ('asphericity', 'Asphericity'),
                                            ('com', 'Centre of mass shift'),
                                            ('diam', 'Maximum 3D diameter'),
                                            ('vol_dens_aabb', 'Volume density - axis-aligned bounding box'),
                                            ('area_dens_aabb', 'Area density - axis-aligned bounding box'),
                                            ('pca_maj_axis', 'Major axis length'),
                                            ('pca_min_axis', 'Minor axis length'),
                                            ('pca_least_axis', 'Least axis length'),
                                            ('pca_elongation', 'Elongation'),
                                            ('pca_flatness', 'Flatness'),
                                            ('area_dens_aee', 'Area density - approximate enclosing ellipsoid'),
                                            ('vol_dens_aee', 'Volume density - approximate enclosing ellipsoid'),
                                            ('vol_dens_conv_hull', 'Volume density - convex hull'),
                                            ('area_dens_conv_hull', 'Area density - convex hull '),
                                            ('integ_int', 'Integrated intensity'),
                                            ('moran_i', 'Moran’s I index'),
                                            ('geary_c', 'Geary’s C measure')
                                            ],
                                    
                                    "PyRadiomics":[('Elongation', 'Elongation'),
                                                    ('Flatness', 'Flatness'),
                                                    ('LeastAxisLength', 'Least axis length'),
                                                    ('MajorAxisLength', 'Major axis length'),
                                                    ('Maximum2DDiameterColumn', np.nan),
                                                    ('Maximum2DDiameterRow', np.nan),
                                                    ('Maximum2DDiameterSlice', np.nan),
                                                    ('Maximum3DDiameter', 'Maximum 3D diameter'),
                                                    ('MeshVolume', 'Volume'),
                                                    ('MinorAxisLength', 'Minor axis length'),
                                                    ('Sphericity', ['Sphericity', 'Compactness 1', 'Compactness 2', 'Spherical disproportion']),
                                                    ('SurfaceArea', 'Surface Area'),
                                                    ('SurfaceVolumeRatio', 'Surface to volume ratio'),
                                                    ('VoxelVolume', 'Approximate Volume'),
                                                ]
                                    }



        self.glcm_ibsi_translation= {"MIRP": [
                                            ('joint_max', 'Joint maximum'),
                                            ('joint_avg', 'Joint average'),
                                            ('joint_var', 'Joint variance'),
                                            ('joint_entr', 'Joint entropy'),
                                            ('diff_avg', 'Difference average'),
                                            ('diff_var', 'Difference variance'),
                                            ('diff_entr', 'Difference entropy'),
                                            ('sum_avg', 'Sum average'),
                                            ('sum_var', 'Sum variance'),
                                            ('sum_entr', 'Sum entropy'),
                                            ('energy', 'Angular second moment'),
                                            ('contrast', 'Contrast'),
                                            ('dissimilarity', 'Dissimilarity'),
                                            ('inv_diff', 'Inverse difference'),
                                            ('inv_diff_norm', 'Normalised inverse difference'),
                                            ('inv_diff_mom', 'Inverse difference moment'),
                                            ('inv_diff_mom_norm', 'Normalised inverse difference moment'),
                                            ('inv_var', 'Inverse variance'),
                                            ('auto_corr', 'Autocorrelation'),
                                            ('clust_tend', 'Cluster tendency'),
                                            ('clust_shade', 'Cluster shade'),
                                            ('clust_prom', 'Cluster prominence'),
                                            ('info_corr1', 'First measure of information correlation'),
                                            ('info_corr2', 'Second measure of information correlation'),
                                            ('corr', 'Correlation'),
                                        ],
                                "PyRadiomics":[
                                                ('Autocorrelation', 'Autocorrelation'),
                                                ('ClusterProminence', 'Cluster prominence'),
                                                ('ClusterShade', 'Cluster shade'),
                                                ('ClusterTendency', ['Cluster tendency', 'Sum variance']),
                                                ('Contrast', 'Contrast'),
                                                ('Correlation', 'Correlation'),
                                                ('DifferenceAverage', ['Difference average','Dissimilarity']),
                                                ('DifferenceEntropy', 'Difference entropy'),
                                                ('DifferenceVariance', 'Difference variance'),
                                                ('Id', 'Inverse difference'),
                                                ('Idm', 'Inverse difference moment'),
                                                ('Idmn', 'Normalised inverse difference moment'),
                                                ('Idn', 'Normalised inverse difference'),
                                                ('Imc1', 'First measure of information correlation'),
                                                ('Imc2', 'Second measure of information correlation'),
                                                ('InverseVariance', 'Inverse variance'),
                                                ('JointAverage', 'Joint average'),
                                                ('JointEnergy', 'Angular second moment'),
                                                ('JointEntropy', 'Joint entropy'),
                                                ('MCC', np.nan),
                                                ('MaximumProbability', 'Maximum probability'),
                                                ('SumAverage', 'Sum average'),
                                                ('SumEntropy', 'Sum entropy'),
                                                ('SumSquares', 'Sum variance'),
                                                (np.nan, 'Joint maximum'),
                                                (np.nan, 'Joint variance')
                                            ]
                            }


        self.glrlm_ibsi_translation = {
                                    "MIRP": [
                                        ('sre', 'Short runs emphasis'),
                                        ('lre', 'Long runs emphasis'),
                                        ('lgre', 'Low grey level run emphasis'),
                                        ('hgre', 'High grey level run emphasis'),
                                        ('srlge', 'Short run low grey level emphasis'),
                                        ('srhge', 'Short run high grey level emphasis'),
                                        ('lrlge', 'Long run low grey level emphasis'),
                                        ('lrhge', 'Long run high grey level emphasis'),
                                        ('glnu', 'Grey level non-uniformity'),
                                        ('glnu_norm', 'Normalised grey level non-uniformity'),
                                        ('rlnu', 'Run length non-uniformity'),
                                        ('rlnu_norm', 'Normalised run length non-uniformity'),
                                        ('r_perc', 'Run percentage'),
                                        ('gl_var', 'Grey level variance'),
                                        ('rl_var', 'Run length variance'),
                                        ('rl_entr', 'Run entropy'),
                                    ],
                                    "PyRadiomics":[('GrayLevelNonUniformity', 'Grey level non-uniformity'),
                                                    ('GrayLevelNonUniformityNormalized', 'Normalised grey level non-uniformity'),
                                                    ('GrayLevelVariance', 'Grey level variance'),
                                                    ('HighGrayLevelRunEmphasis', 'High grey level run emphasis'),
                                                    ('LongRunEmphasis', 'Long runs emphasis'),
                                                    ('LongRunHighGrayLevelEmphasis', 'Long run high grey level emphasis'),
                                                    ('LongRunLowGrayLevelEmphasis', 'Long run low grey level emphasis'),
                                                    ('LowGrayLevelRunEmphasis', 'Low grey level run emphasis'),
                                                    ('RunEntropy', 'Run entropy'),
                                                    ('RunLengthNonUniformity', 'Run length non-uniformity'),
                                                    ('RunLengthNonUniformityNormalized', 'Normalised run length non-uniformity'),
                                                    ('RunPercentage', 'Run percentage'),
                                                    ('RunVariance', 'Run length variance'),
                                                    ('ShortRunEmphasis', 'Short runs emphasis'),
                                                    ('ShortRunHighGrayLevelEmphasis', 'Short run high grey level emphasis'),
                                                    ('ShortRunLowGrayLevelEmphasis', 'Short run low grey level emphasis'),
                                                ]
                                    }
                                                
                                                
        self.glszm_ibsi_translation = {
                                    "MIRP":[
                                        ('sze', 'Small zone emphasis'),
                                        ('lze', 'Large zone emphasis'),
                                        ('lgze', 'Low grey level zone emphasis'),
                                        ('hgze', 'High grey level zone emphasis'),
                                        ('szlge', 'Small zone low grey level emphasis'),
                                        ('szhge', 'Small zone high grey level emphasis'),
                                        ('lzlge', 'Large zone low grey level emphasis'),
                                        ('lzhge', 'Large zone high grey level emphasis'),
                                        ('glnu', 'Grey level non-uniformity'),
                                        ('glnu_norm', 'Normalised grey level non-uniformity'),
                                        ('zsnu', 'Zone size non-uniformity'),
                                        ('zsnu_norm', 'Normalised zone size non-uniformity'),
                                        ('z_perc', 'Zone percentage'),
                                        ('gl_var', 'Grey level variance'),
                                        ('zs_var', 'Zone size variance'),
                                        ('zs_entr', 'Zone size entropy'),
                                    ],          
                                    "PyRadiomics": [
                                        ('GrayLevelNonUniformity', 'Grey level non-uniformity'),
                                        ('GrayLevelNonUniformityNormalized', 'Normalised grey level non-uniformity'),
                                        ('GrayLevelVariance', 'Grey level variance'),
                                        ('HighGrayLevelZoneEmphasis', 'High grey level zone emphasis'),
                                        ('LargeAreaEmphasis', 'Large zone emphasis'),
                                        ('LargeAreaHighGrayLevelEmphasis', 'Large zone high grey level emphasis'),
                                        ('LargeAreaLowGrayLevelEmphasis', 'Large zone low grey level emphasis'),
                                        ('LowGrayLevelZoneEmphasis', 'Low grey level zone emphasis'),
                                        ('SizeZoneNonUniformity', 'Zone size non-uniformity'),
                                        ('SizeZoneNonUniformityNormalized', 'Normalised zone size non-uniformity'),  # Fixed duplicate
                                        ('SmallAreaEmphasis', 'Small zone emphasis'),
                                        ('SmallAreaHighGrayLevelEmphasis', 'Small zone high grey level emphasis'),
                                        ('SmallAreaLowGrayLevelEmphasis', 'Small zone low grey level emphasis'),
                                        ('ZoneEntropy', 'Zone size entropy'),
                                        ('ZonePercentage', 'Zone percentage'),
                                        ('ZoneVariance', 'Zone size variance'),
                                    ]
                                }
                                        
        self.gldm_ibsi_translation = {
                                    "MIRP": [
                                        ('sde', 'Small distance emphasis'),
                                        ('lde', 'Large distance emphasis'),
                                        ('lgze', 'Low grey level zone emphasis'),
                                        ('hgze', 'High grey level zone emphasis'),
                                        ('sdlge', 'Small distance low grey level emphasis'),
                                        ('sdhge', 'Small distance high grey level emphasis'),
                                        ('ldlge', 'Large distance low grey level emphasis'),
                                        ('ldhge', 'Large distance high grey level emphasis'),
                                        ('glnu', 'Grey level non-uniformity'),
                                        ('glnu_norm', 'Normalised grey level non-uniformity'),
                                        ('zdnu', 'Zone distance non-uniformity'),
                                        ('zdnu_norm', 'Normalised zone distance non-uniformity'),
                                        ('z_perc', 'Zone percentage'),
                                        ('gl_var', 'Grey level variance'),
                                        ('zd_var', 'Zone distance variance'),
                                        ('zd_entr', 'Zone distance entropy'),
                                    ],
                                    "PyRadiomics": [
                                        ('DependenceEntropy', 'Zone distance entropy'),
                                        ('DependenceNonUniformity', 'Zone distance non-uniformity'),
                                        ('DependenceNonUniformityNormalized', 'Normalised zone distance non-uniformity'),
                                        ('DependenceVariance', 'Zone distance variance'),
                                        ('GrayLevelNonUniformityNormalized', 'Normalised grey level non-uniformity'), # correlating with First Order - Uniformity 
                                        ('GrayLevelNonUniformity', 'Grey level non-uniformity'),
                                        ('GrayLevelVariance', 'Grey level variance'),
                                        ('HighGrayLevelEmphasis', 'High grey level zone emphasis'),
                                        ('LargeDependenceEmphasis', 'Large distance emphasis'),
                                        ('LargeDependenceHighGrayLevelEmphasis', 'Large distance high grey level emphasis'),
                                        ('LargeDependenceLowGrayLevelEmphasis', 'Large distance low grey level emphasis'),
                                        ('LowGrayLevelEmphasis', 'Low grey level zone emphasis'),
                                        ('SmallDependenceEmphasis', 'Small distance emphasis'),
                                        ('SmallDependenceHighGrayLevelEmphasis', 'Small distance high grey level emphasis'),
                                        ('SmallDependenceLowGrayLevelEmphasis', 'Small distance low grey level emphasis'),
                                        (np.nan, "Zone percentage")
                                        ]
                                    }
            
        self.nrtdm_ibsi_translation = {
                                    "MIRP": [
                                                ('coarseness', 'Coarseness'),
                                                ('contrast', 'Contrast'),
                                                ('busyness', 'Busyness'),
                                                ('complexity', 'Complexity'),
                                                ('strength', 'Strength'),
                                            ],
                                    "PyRadiomics": [
                                                        ('Busyness', 'Busyness'),
                                                        ('Coarseness', 'Coarseness'),
                                                        ('Complexity', 'Complexity'),
                                                        ('Contrast', 'Contrast'),
                                                        ('Strength', 'Strength'),
                                                    ]
                                }
                                        
        self.ngldm_ibsi_translation = {         
                                    "MIRP":[
                                                ('lde', 'Low dependence emphasis'),
                                                ('hde', 'High dependence emphasis'),
                                                ('lgce', 'Low grey level count emphasis'),
                                                ('hgce', 'High grey level count emphasis'),
                                                ('ldlge', 'Low dependence low grey level emphasis'),
                                                ('ldhge', 'Low dependence high grey level emphasis'),
                                                ('hdlge', 'High dependence low grey level emphasis'),
                                                ('hdhge', 'High dependence high grey level emphasis'),
                                                ('glnu', 'Grey level non-uniformity'),
                                                ('glnu_norm', 'Normalised grey level non-uniformity'),
                                                ('dcnu', 'Dependence count non-uniformity'),
                                                ('dcnu_norm', 'Normalised dependence count non-uniformity'),
                                                ('dc_perc', 'Dependence count percentage'),
                                                ('gl_var', 'Grey level variance'),
                                                ('dc_var', 'Dependence count variance'),
                                                ('dc_entr', 'Dependence count entropy'),
                                                ('dc_energy', 'Dependence count energy'),
                                        ]
                                }
        self.ibs_ibsi_translation = {
                                "MIRP": [
                                            ('mean', 'Mean'),
                                            ('var', 'Variance'),
                                            ('skew', 'Skewness'),
                                            ('kurt', 'Kurtosis'),
                                            ('median', 'Median'),
                                            ('min', 'Minimum'),
                                            ('p10', '10th percentile'),
                                            ('p90', '90th percentile'),
                                            ('max', 'Maximum'),
                                            ('iqr', 'Interquartile range'),
                                            ('range', 'Range'),
                                            ('mad', 'Mean absolute deviation'),
                                            ('rmad', 'Robust mean absolute deviation'),
                                            ('medad', 'Median absolute deviation'),
                                            ('cov', 'Coefficient of variation'),
                                            ('qcod', 'Quartile coefficient of dispersion'),
                                            ('energy', 'Energy'),
                                            ('rms', 'Root mean square')
                                        ],
                                "PyRadiomics": [
                                                    ('10Percentile', '10th percentile'),
                                                    ('90Percentile', '90th percentile'),
                                                    ('Energy', 'Energy'),
                                                    ('TotalEnergy', np.nan),  # No direct IBSI equivalent
                                                    ('Minimum', 'Minimum'),
                                                    ('Maximum', 'Maximum'),
                                                    ('Mean', 'Mean'),
                                                    ('Median', 'Median'),
                                                    ('Range', 'Range'),
                                                    ('MeanAbsoluteDeviation', 'Mean absolute deviation'),
                                                    ('RobustMeanAbsoluteDeviation', 'Robust mean absolute deviation'),
                                                    ('RootMeanSquared', 'Root mean square'),
                                                    ('Skewness', 'Skewness'),
                                                    ('Kurtosis', 'Kurtosis'),
                                                    ('Variance', 'Variance'),
                                                    (np.nan , 'Coefficient of variation'),
                                                    ('InterquartileRange', 'Interquartile range'),
                                                    (np.nan, 'Median absolute deviation'),
                                                ]
                                }
            
            
        self.ihs_ibsi_translation = {
                                "MIRP": [
                                            ('mean', 'Intensity histogram mean'),
                                            ('var', 'Intensity histogram variance'),
                                            ('skew', 'Intensity histogram skewness'),
                                            ('kurt', 'Intensity histogram kurtosis'),
                                            ('median', 'Intensity histogram median'),
                                            ('min', 'Intensity histogram minimum'),
                                            ('p10', 'Intensity histogram 10th percentile'),
                                            ('p90', 'Intensity histogram 90th percentile'),
                                            ('max', 'Intensity histogram maximum'),
                                            ('mode', 'Intensity histogram mode'),
                                            ('iqr', 'Intensity histogram interquartile range'),
                                            ('range', 'Intensity histogram range'),
                                            ('mad', 'Intensity histogram mean absolute deviation'),
                                            ('rmad', 'Intensity histogram robust mean absolute deviation'),
                                            ('medad', 'Intensity histogram median absolute deviation'),
                                            ('cov', 'Intensity histogram coefficient of variation'),
                                            ('qcod', 'Intensity histogram quartile coefficient of dispersion'),
                                            ('entropy', 'Intensity histogram entropy'),
                                            ('uniformity', 'Intensity histogram uniformity'),
                                            ('max_grad', 'Maximum histogram gradient'),
                                            ('max_grad_g', 'Maximum histogram gradient grey level'),
                                            ('min_grad', 'Minimum histogram gradient'),
                                            ('min_grad_g', 'Minimum histogram gradient grey level'),
                                        ],
                                "PyRadiomics": [
                                                    ('Entropy', 'Intensity histogram entropy'), 
                                                    ('Uniformity', ['Intensity histogram uniformity', 'grey_level_distance_zone_matrix_Normalised grey level non-uniformity'])
                                                ]
                                    
                                }

        self.firstorder_ibsi_translation = {"PyRadiomics": [
                                                        ('Entropy', 'Intensity histogram entropy'), 
                                                        ('Uniformity', ['Intensity histogram uniformity', 'grey_level_distance_zone_matrix_Normalised grey level non-uniformity']),
                                                        ('10Percentile', '10th percentile'),
                                                        ('90Percentile', '90th percentile'),
                                                        ('Energy', 'Energy'),
                                                        ('TotalEnergy', np.nan),  # No direct IBSI equivalent
                                                        ('Minimum', 'Minimum'),
                                                        ('Maximum', 'Maximum'),
                                                        ('Mean', 'Mean'),
                                                        ('Median', 'Median'),
                                                        ('Range', 'Range'),
                                                        ('MeanAbsoluteDeviation', 'Mean absolute deviation'),
                                                        ('RobustMeanAbsoluteDeviation', 'Robust mean absolute deviation'),
                                                        ('RootMeanSquared', 'Root mean square'),
                                                        ('Skewness', 'Skewness'),
                                                        ('Kurtosis', 'Kurtosis'),
                                                        ('Variance', 'Variance')
                                                    ]
                                    }
        self.li_ibsi_translation = {"MIRP":[('peak_loc', 'Local intensity peak'), 
                                    ('peak_glob', 'Global intensity peak')
                                    ]
                            }

        self.ivh_ibsi_translation = {
                                    "MIRP": [
                                        ('v10', 'Volume fraction at 10% intensity'),
                                        ('v90', 'Volume fraction at 90% intensity'),
                                        ('i10', 'Intensity at 10% volume'),
                                        ('i90', 'Intensity at 90% volume'),
                                        ('diff_v10_v90', 'Difference in volume fraction between 10% and 90% intensity'),
                                        ('diff_i10_i90', 'Difference in intensity between 10% and 90% volume'),
                                        # ('auc', 'Area under IVH curve'), # not IBSI compliant
                                    ]
                                }


        self.translation_dicts = {
            "morphological": self.morphological_ibsi_translation,
            "grey_level_co-occurrence_matrix": self.glcm_ibsi_translation,
            "grey_level_run_length_matrix": self.glrlm_ibsi_translation,
            "grey_level_size_zone_matrix": self.glszm_ibsi_translation,
            "grey_level_distance_zone_matrix": self.gldm_ibsi_translation,
            "neighbourhood_grey_tone_difference_matrix": self.nrtdm_ibsi_translation,
            "neighbouring_grey_level_dependence_matrix": self.ngldm_ibsi_translation, 
            "intensity_histogram": self.ihs_ibsi_translation,
            "firstorder": self.firstorder_ibsi_translation,
            "local_intensity": self.li_ibsi_translation,
            "intensity-volume_histogram": self.ivh_ibsi_translation,
            "intensity-based_statistics": self.ibs_ibsi_translation,
        }


        self.ibsi_feature_classes = ["morphological", 
                                    "grey_level_co-occurrence_matrix", 
                                    "grey_level_run_length_matrix", 
                                    "grey_level_size_zone_matrix",
                                    "grey_level_distance_zone_matrix", 
                                    "neighbourhood_grey_tone_difference_matrix",
                                    "neighbouring_grey_level_dependence_matrix",
                                    "intensity_histogram",
                                    "local_intensity",
                                    "intensity-volume_histogram",
                                    "intensity-based_statistics"]

    def create_IBSI_feature_list(self):

        print("Creating IBSI Feature list ...")
        self.logger.info("Creating IBSI Feature list ...")
        for feature_class in self.ibsi_feature_classes:
            for features in self.translation_dicts[feature_class]["MIRP"]:
                self.ibsi_features.append(feature_class + "_" + features[1].replace(" ", "_"))
        self.logger.info(f"Idenfitfied {len(self.ibsi_features)} IBSI features successfully!")

    def find_IBSI_feature(self, ibsi_feature_class_features, name_formatted):
    
        ibsi_feature = ""

        for ibsi_feature_pair in ibsi_feature_class_features:
            if name_formatted == ibsi_feature_pair[0]:
                if isinstance(ibsi_feature_pair[1], list):
                    ibsi_feature = []
                    for feature in ibsi_feature_pair[1]:
                        ibsi_feature.append(feature.replace(" ", "_"))
                    #if isinstance(ibsi_feature, str):
                    #    ibsi_feature = ibsi_feature.replace(" ", "_")
                    #else:
                    #    print(f"IBSI features not present for {feature}")
                else:
                    if isinstance(ibsi_feature_pair[1], str):
                        ibsi_feature = ibsi_feature_pair[1].replace(" ", "_")
                    else:
                        break
                break
                
        return ibsi_feature

    def convert2IBSI_FeatureName(self, features, profile= None):
        
        ibsi_feature = ""
        name_formatted = ""

        if profile is None:
            profile = self.profile

        for feature in features.copy().columns:
            for name in profile["Name"]:
                if feature == name:
                    
                    feature_class = profile.loc[profile["Name"] == name, "Feature_Class"].values
                    name_formatted = profile.loc[profile["Name"] == name, "Feature"].values
                    ibsi_feature_class_features = ""
                    
                    # PyRadiomics firstorder features are in the IBSI feature class ibs or ihs
                    if feature_class[0] == "firstorder":
                        ibsi_feature_class_features = self.translation_dicts["intensity-based_statistics"][self.extractor]
                        
                        ibsi_feature = self.find_IBSI_feature(ibsi_feature_class_features, name_formatted[0])
                        
                        if len(ibsi_feature) == 0:
                            ibsi_feature_class_features = self.translation_dicts["intensity_histogram"][self.extractor]
                            ibsi_feature = self.find_IBSI_feature(ibsi_feature_class_features, name_formatted[0])

                            if len(ibsi_feature) == 0:
                                print(f"Could not find IBSI feature for {feature}")
                                break
                    
                    else:
                        if feature_class[0] in self.translation_dicts:
                            # Try to get the IBSI formatted feature name
                            ibsi_feature_class_features = self.translation_dicts[feature_class[0]][self.extractor]
                        else:
                            print(f"Feature class {feature_class[0]} in not registered in the IBSI.")
                            self.error.warning(f"Feature class {feature_class[0]} in not registered in the IBSI.")
                            break

                        ibsi_feature = self.find_IBSI_feature(ibsi_feature_class_features, name_formatted[0])
                    
                    if len(ibsi_feature) > 0:
                        # if there are multiple IBSI names for the same feature
                        if isinstance(ibsi_feature, list):
                            tmp = features.copy()
                            for feat in ibsi_feature:
                                if feat in features.columns:
                                    if "grey_level_distance_zone_matrix" not in feat:
                                        new_feature = feature_class[0] + "_" + feat
                                    else:
                                        new_feature = feat
                                    features = features.rename(columns = {feature:new_feature.replace(" ", "_")})
                                else:
                                    features[feature] = tmp.copy()[feature]
                                    if "grey_level_distance_zone_matrix" not in feat:
                                        new_feature = feature_class[0] + "_" + feat
                                    else:
                                        new_feature = feat
                                    features = features.rename(columns = {feature:new_feature.replace(" ", "_")})
                                
                            
                        else:
                            new_feature = feature_class[0] + "_" + ibsi_feature
                            features = features.rename(columns = {feature:new_feature.replace(" ", "_")})

                    else:
                        print(f"Could not find any IBSI feature for {feature}")
                        self.error.warning(f"Could not find any IBSI feature for {feature}")


                    break
                    
        if self.extractor == "PyRadiomics":
            # rename features as the meaning and feature classea are different to IBSI standart
            
            renaming = {'firstorder_10th_percentile': 'intensity-based_statistics_10th_percentile',
                        'firstorder_90th_percentile': 'intensity-based_statistics_90th_percentile',
                        'firstorder_Energy': 'intensity-based_statistics_Energy',
                        'firstorder_Minimum': 'intensity-based_statistics_Minimum',
                        'firstorder_Maximum': 'intensity-based_statistics_Maximum',
                        'firstorder_Mean': 'intensity-based_statistics_Mean', 
                        'firstorder_Median': 'intensity-based_statistics_Median',
                        'firstorder_Range': 'intensity-based_statistics_Range',
                        'firstorder_Mean_absolute_deviation': 'intensity-based_statistics_Mean_absolute_deviation',
                        'original_firstorder_RobustMeanAbsoluteDeviation':  'intensity-based_statistics_Robust_mean_absolute_deviation',
                        'firstorder_RobustMeanAbsoluteDeviation':  'intensity-based_statistics_Robust_mean_absolute_deviation',
                        'firstorder_Root_mean_square': 'intensity-based_statistics_Root_mean_square',
                        'firstorder_RootMeanSquared': 'intensity-based_statistics_Root_mean_square',
                        'firstorder_Skewness': 'intensity-based_statistics_Skewness',
                        'firstorder_Kurtosis': 'intensity-based_statistics_Kurtosis',
                        'firstorder_Variance': 'intensity-based_statistics_Variance',
                        'original_firstorder_InterquartileRange': 'intensity-based_statistics_Interquartile_range',
                        'firstorder_InterquartileRange': 'intensity-based_statistics_Interquartile_range',
                        'firstorder_Intensity_histogram_entropy':'intensity_histogram_Intensity_histogram_entropy',
                        'firstorder_Interquartile_range': 'intensity-based_statistics_Interquartile_range',
                        'firstorder_Intensity_histogram_uniformity': 'intensity_histogram_Intensity_histogram_uniformity',
                        'shape':'morphological'
                    }
            # apply substring replacements to ALL column names
            cols = list(map(str, features.columns))

            # sort keys longest-first to avoid partial overlaps causing double-changes
            for old, new in sorted(renaming.items(), key=lambda kv: -len(kv[0])):
                cols = [c.replace(old, new) for c in cols]

            features.columns = cols
            # features.rename(columns=renaming, inplace=True)
            
        print(features.columns)    
        return features

    def plot_ibsi_feature_coverage(self, df, output_file="feature_coverage.png"):
        """
        Plots the coverage of IBSI features in a dataframe (publication-ready)
        with smart, collision-free labels for stacked bars.

        Uses:
        - self.ibsi_features            (list of IBSI feature names)
        - self.ibsi_feature_classes     (list of class prefixes, e.g. 'glcm', 'gldm', ...)
        - self.extractor                (string for figure title)
        Returns:
        dict with sets/lists of missing and non-IBSI feature names.
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # ---------- Helper: smart labels with collision avoidance ----------
        def add_smart_labels(ax, df_row, rects, small_thresh_pct=6.0, min_sep_frac=0.05):
            """
            Place labels inside segments when large enough; otherwise place them outside
            on the right with leader lines. Outside labels are vertically de-overlapped.

            ax : matplotlib Axes
            df_row : pandas Series (values correspond to the plotted columns, same order)
            rects : list[Rectangle] rectangles for this bar (same order as df_row)
            small_thresh_pct : float, % threshold below which labels go outside
            min_sep_frac : float, minimal vertical separation between outside labels
                        as a fraction of total bar height
            """
            values = df_row.values.astype(float)
            total = float(values.sum()) if values.sum() else 1.0
            if not rects:
                return

            # Bar geometry
            bar_left = rects[0].get_x()
            bar_right = rects[0].get_x() + rects[0].get_width()
            bar_y0 = rects[0].get_y()
            bar_y1 = rects[-1].get_y() + rects[-1].get_height()
            x_out = bar_right + 0.02  # small horizontal gap (axis units)

            inside_items, outside_items = [], []
            for val, rect in zip(values, rects):
                if val <= 0:
                    continue
                pct = 100.0 * val / total
                y_center = rect.get_y() + rect.get_height() / 2.0
                label = f"{pct:.1f}% ({int(val)})"
                if pct >= small_thresh_pct and rect.get_height() > 0:
                    inside_items.append((rect, y_center, label))
                else:
                    outside_items.append((rect, y_center, label))

            # Draw inside labels (centered)
            for rect, y_c, label in inside_items:
                x_c = rect.get_x() + rect.get_width() / 2.0
                ax.text(x_c, y_c, label, ha='center', va='center', fontsize=12,
                        color='black', clip_on=True)

            # De-overlap outside labels
            if outside_items:
                outside_items.sort(key=lambda t: t[1])  # by y
                min_sep = (bar_y1 - bar_y0) * float(min_sep_frac)
                placed_positions = []
                for _, y_c, _ in outside_items:
                    y_target = y_c
                    if placed_positions:
                        y_target = max(y_target, placed_positions[-1] + min_sep)
                    # keep within bar bounds
                    y_target = min(max(y_target, bar_y0 + 0.5 * min_sep), bar_y1 - 0.5 * min_sep)
                    placed_positions.append(y_target)

                for (rect, y_c, label), y_target in zip(outside_items, placed_positions):
                    ax.text(
                        x_out, y_target, label,
                        ha='left', va='center', fontsize=12,
                        color='black', clip_on=False
                    )

        # ---------- Abbreviations for long class names (legend/columns) ----------
        class_abbrev_map = {
            "grey level co-occurrence matrix": "GLCM",
            "neighbouring grey level dependence matrix": "NGLDM",
            "grey level run length matrix": "GLRLM",
            "grey level distance zone matrix": "GLDZM",
            "neighbourhood grey tone difference matrix": "NGTDM",
            "grey level size zone matrix": "GLSZM",
            # Optional shorthand for the remaining groups:
            "intensity histogram": "IH",
            "intensity-based statistics": "IS",
            "morphological": "Morphological",
            "intensity-volume histogram": "IVH",
            "local intensity": "LI",
        }

        # ---------- Extract & compute coverage ----------
        df_features = set(df.columns)
        ibsi_set = set(self.ibsi_features)

        ibsi_in_df = df_features.intersection(ibsi_set)
        ibsi_missing = ibsi_set - df_features

        ibsi_present_count = len(ibsi_in_df)
        ibsi_missing_count = len(ibsi_missing)

        non_ibsi_features = df_features - ibsi_set
        # first-order features are counted as IBSI → exclude from non-IBSI
        non_ibsi_features = [x for x in non_ibsi_features if "firstorder" not in x]
        non_ibsi_count = len(non_ibsi_features)

        # Class counts (by prefix match)
        ibsi_feature_classes_in_data = []
        for feat in ibsi_in_df:
            for feat_class in self.ibsi_feature_classes:
                if feat.startswith(feat_class):
                    ibsi_feature_classes_in_data.append(feat_class)

        feature_class_counts = (
            pd.Series(ibsi_feature_classes_in_data).value_counts().sort_index()
            if ibsi_feature_classes_in_data else pd.Series(dtype=int)
        )

        # ---------- Styling for publication ----------
        plt.rcParams.update({
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "Arial",
        })

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), dpi=300)
        fig.suptitle(f"{self.extractor} IBSI Feature Profile",
                    fontsize=18, fontweight='bold', y=1.02, ha='center')

        # ---------- Pie chart: completeness ----------
        if ibsi_missing_count > 0:
            labels = [f"Features in Data\n({ibsi_present_count})",
                    f"Features Missing\n({ibsi_missing_count})"]
            sizes = [ibsi_present_count, ibsi_missing_count]
            colors = ['#66b3ff', '#ffcc99']
            axes[0].pie(sizes, labels=labels, autopct='%1.1f%%',
                        colors=colors, startangle=180, textprops={'fontsize': 14})
        else:
            labels = [f"Features in Data\n({ibsi_present_count})"]
            sizes = [ibsi_present_count]
            colors = ['#66b3ff']
            axes[0].pie(sizes, labels=labels, autopct='%1.1f%%',
                        colors=colors, startangle=90, textprops={'fontsize': 14})
        axes[0].set_title("IBSI Feature Completeness", fontweight='bold', y=1.05)
        axes[0].axis("equal")

        # ---------- Bar data: coverage row + class row ----------
        bar_data = pd.DataFrame({
            "IBSI Features": [ibsi_present_count],
            "Non-IBSI Features": [non_ibsi_count]
        }, index=["Feature Coverage"])

        feature_class_data = pd.DataFrame(feature_class_counts).T
        feature_class_data.index = ["IBSI Feature Classes"]

        stacked_data = pd.concat([bar_data, feature_class_data], axis=0).fillna(0)
        stacked_data.columns = stacked_data.columns.str.replace(r'[_]+', ' ', regex=True)
        stacked_data.rename(columns=class_abbrev_map, inplace=True)

        # ---------- Plot stacked bar ----------
        colors = ['#66b3ff', '#ff6666'] + list(plt.cm.Paired.colors)
        ax = stacked_data.plot(
            kind='bar', stacked=True, ax=axes[1],
            color=colors, edgecolor="black", linewidth=0.5, width=0.75
        )
        axes[1].set_ylabel("Number of Features")
        axes[1].set_title("Feature Distribution in Data", fontweight='bold', y=1.05)
        axes[1].tick_params(axis='x', labelrotation=0)
        axes[1].legend(title="Feature Type", 
                       loc='upper left', 
                       bbox_to_anchor=(1.1, 1), 
                       frameon=True)
        axes[1].grid(False, axis='x')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        # ---------- Add smart, collision-free labels to each stacked bar ----------
        # Matplotlib stacks: one container per column; each container has one patch per bar (row).
        # We build the list of rects for each row by taking the ith patch from every container.
        n_rows = stacked_data.shape[0]
        for row_idx, row_name in enumerate(stacked_data.index):
            row_vals = stacked_data.loc[row_name]
            rects_for_row = [cont.patches[row_idx] for cont in ax.containers if len(cont.patches) > row_idx]
            # Slightly higher threshold for the classes bar (usually many tiny slices)
            thresh = 6.0 if row_name == "Feature Coverage" else 6.0
            add_smart_labels(ax, row_vals, rects_for_row, small_thresh_pct=thresh, min_sep_frac=0.05)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

        return {"Missing_IBSI": ibsi_missing, "Non_IBSI": non_ibsi_features}


    def format_features(self):
        """
        Format the features to IBSI standard.
        """
        missing_df = pd.DataFrame()
        non_ibsi_df = pd.DataFrame()

        if self.profile is None:
            
            # Create Feature Format
            self.profile = FeatureFormatter(
                            features = self.features,
                            extractor = self.extractor,
                            RunID=self.RunID,
                            logger=self.logger,
                            error=self.error,
                            output_path = self.output_path,
                            generate_feature_profile_plot = True).exe(title=self.extractor + " Feature Profile")

        
        self.profile.to_csv(self.output_path + "/Feature_profile_" + self.extractor + "_" + self.RunID + ".csv")

        if len(self.profile) > 0:
            # Create IBSI feature list
            self.create_IBSI_feature_list()

            # Convert the features to IBSI standard
            converted_features = self.convert2IBSI_FeatureName(self.features)

            # Plot and analyse the IBSI feature coverage
            out = self.plot_ibsi_feature_coverage(converted_features.copy(), output_file=self.output_path + "/IBSI_feature_coverage_" + self.extractor  + "_" + self.RunID + ".png")
            converted_features.to_csv(self.output_path + "/IBSI_feature_formatted_" + self.extractor  + "_" + self.RunID + ".csv")

            missing_df = pd.DataFrame({"Missing_IBSI_features":list(out["Missing_IBSI"])})
            non_ibsi_df = pd.DataFrame({"Non_IBSI_features":list(out["Non_IBSI"])})
        else:
            self.error.warning("Could not find any IBSI related features nor any radiomics features!")
            print("Could not find any IBSI related features nor any radiomics features!")

        missing_df.to_csv(self.output_path + "/" + self.extractor + "_missing_IBSI_features.csv")
        non_ibsi_df.to_csv(self.output_path + "/" + self.extractor + "_non_IBSI_features.csv")
        