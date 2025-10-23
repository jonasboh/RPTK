

import radiomics
from radiomics import *
import tqdm
import click
import SimpleITK as sitk
import six
import yaml
import os
import pandas as pd
import time

from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.feature_filtering.Feature_formater import FeatureFormatter

class RadiomicsMapGenerator:
    def __init__(self,
                 list_of_features: list,
                 path_to_img: str,
                 path_to_msk: str,
                 out_path: str,
                 extraction_yaml: str = None,
                 logger=None,
                 error=None,
                 RunID: str = None
                 ):

        self.list_of_features = list_of_features
        self.path_to_img = path_to_img
        self.path_to_msk = path_to_msk
        self.out_path = out_path
        self.extraction_yaml = extraction_yaml
        self.logger = logger
        self.error = error
        self.RunID = RunID

        if self.out_path.endswith("/"):
            self.out_path = self.out_path + "feature_maps/"
        else:
            self.out_path = self.out_path + "/feature_maps/"

        os.makedirs(self.out_path, exist_ok=True)

        if self.RunID is None:
            self.RunID = time.strftime("%Y%m%d-%H%M%S")

        self.logger = LogGenerator(
            log_file_name=self.out_path + "Feature_map_" + self.RunID + ".log",
            logger_topic="Feature Map Generation"
        ).generate_log()

        self.error = LogGenerator(
            log_file_name=self.out_path + "Feature_map_" + self.RunID + ".err",
            logger_topic="Feature Map Generation Error"
        ).generate_log()

        self.ID = os.path.basename(self.path_to_img)[:-len(".nii.gz")] + "_" + os.path.basename(self.path_to_msk)[:-len(".nii.gz")]

        # only take PyRadiomics errors:
        radiomics.logger = self.logger
        radiomics.setVerbosity(50)
        radiomics.progressReporter = tqdm.tqdm

        self.feature_class_mapping_dict = {"grey_level_co-occurrence_matrix": "glcm",
                                           "grey_level_run_length_matrix": "glrlm",
                                           "grey_level_size_zone_matrix": "glszm",
                                           "grey_level_distance_zone_matrix": "gldm",
                                           "neighbourhood_grey_tone_difference_matrix": "ngtdm"}

        formatted_features = FeatureFormatter(features=self.list_of_features,
                                              extractor="PyRadiomics",
                                              logger=self.logger,
                                              error=self.error,
                                              output_path=self.out_path).exe(title="Feature_Profile")

        formated_feature_class = []
        for feat_class in list(set(formatted_features["Feature_Class"].values)):
            if feat_class in self.feature_class_mapping_dict:
                formated_feature_class.append(self.feature_class_mapping_dict[feat_class])
            else:
                formated_feature_class.append(feat_class)

        kernels = {key: {} for key in list(
            set(formatted_features.loc[formatted_features["Image_Kernel"] != "Original_Image", "Image_Kernel"].values))}
        features = {key: [] for key in formated_feature_class}

        for feat_class in list(set(formatted_features["Feature_Class"].values)):
            for i, row in formatted_features.loc[formatted_features["Feature_Class"] == feat_class].iterrows():
                if "_" in str(row["Feature"]):
                    feature = str(row["Feature"]).split("_")[-1]
                else:
                    feature = str(row["Feature"])

                if feat_class in self.feature_class_mapping_dict:
                    features[self.feature_class_mapping_dict[feat_class]].append(feature)
                else:
                    self.logger.info("Can not generate feature map for {} of feature class {}.".format(feature, feat_class))

        if self.extraction_yaml is None:

            settings = {'binWidth': 25,
                        'resampledPixelSpacing': None,
                        'interpolator': sitk.sitkBSpline,
                        'voxelBatch': 1000}
        else:
            with open(self.extraction_yaml, 'r') as file:
                config = yaml.safe_load(file)

            settings = config["setting"]

        # print(features)
        self.logger.info(settings)

        # Config for default extraction settings
        self.py_extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        self.py_extractor.disableAllFeatures()
        # only extract for defined features
        self.py_extractor.enableFeaturesByName(**features)
        # include transformation if it has been done
        if len(kernels) > 0:
            self.py_extractor.enableImageTypes(**kernels)

    def generate_feature_maps(self):
        """
         Generate Feature maps for img/msk combination and save maps to out_path folder
         """

        self.logger.info("Calculating features ...")
        result = self.py_extractor.execute(self.path_to_img, self.path_to_msk, voxelBased=True)

        for featureName, featureValue in six.iteritems(result):
            if isinstance(featureValue, sitk.Image):
                sitk.WriteImage(featureValue, self.out_path + '%s_%s.nii.gz' % (self.ID, featureName))
                self.logger.info('Computed %s, stored as "%s_%s.nii.gz"' % (featureName, self.ID, featureName))
            else:
                self.logger.info('%s: %s' % (featureName, featureValue))




