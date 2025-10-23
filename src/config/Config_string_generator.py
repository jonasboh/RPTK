#!/usr/bin/env python

import sys
import pandas as pd
import yaml

class ConfigStringGenerator(object):
    def __init__(self,
                 feature_extractor: str,  # Pyradiomics or Mirp
                 modality: str,  # CT or MR
                 roi: str = "intra",  # e.g. intratumoral or peritumoral
                 resampling: bool = 0,  # True or False
                 resampling_method: str = None,  # thresholding or others
                 interpolator: str = None, # e.g. sitkBSpline, sitkNearestNeighbor as an interpolator
                 normalization: bool = 0,  # True or False
                 normalization_scale: int = None,  # integer
                 re_segmentation: bool = 0,  # True or False
                 re_segmentation_method: str = None,  # thresholding or kmeans
                 discretization_method: str = None,  # equal_frequency or equal_width
                 discretization_parameter: str = None,  # number of bins or bin width
                 path_to_config_file: str = None  # path to Pyradiomics yaml config file
                 ):

        # self.config = config
        self.feature_extractor = feature_extractor
        self.modality = modality
        self.re_segmentation = re_segmentation
        self.re_segmentation_method = re_segmentation_method
        self.discretization_method = discretization_method
        self.discretization_parameter = discretization_parameter
        self.path_to_config_file = path_to_config_file
        self.normalization = normalization
        self.normalization_scale = normalization_scale
        self.resampling = resampling
        self.resampling_method = resampling_method
        self.interpolator = interpolator
        self.roi = roi

    def generate(self):
        config_string = ""

        if self.path_to_config_file is not None:
            if self.path_to_config_file.endswith(".yaml"):
                with open(self.path_to_config_file, "r") as stream:
                    try:
                        config_yaml = yaml.safe_load(stream)["setting"]
                        for key, value in config_yaml.items():
                            print(key, value)
                            if "resampling" == key:
                                self.resampling = value
                            if "interpolator" == key:
                                self.interpolator = value
                            if "normalize" == key:
                                self.normalization = value
                            if "normalizeScale" == key:
                                self.normalization_scale = value
                            if "resegmentShape" == key:
                                self.re_segmentation = int(value)
                            if "resegmentRange" == key:
                                if value == None:
                                    self.re_segmentation = 0
                                    self.re_segmentation_method = None
                                else:
                                    self.re_segmentation_method = "threshold"
                            if "interpolator" == key:
                                self.interpolator = value
                            if "binwith" == key:
                                self.discretization_parameter = value
                            if "bincount" == key:
                                self.discretization_parameter = value
                    except yaml.YAMLError as exc:
                        print(exc)

                    self.config = {
                        "feature_extractor": self.feature_extractor,
                        "modality": self.modality,
                        "resampling": int(self.resampling),
                        "resampling_method": self.resampling_method,
                        "interpolator": self.interpolator,
                        "normalization": int(self.normalization),
                        "normalization_scale": self.normalization_scale,
                        "re_segmentation": int(self.re_segmentation),
                        "re_segmentation_method": self.re_segmentation_method,
                        "discretization_method": self.discretization_method,
                        "discretization_parameter": self.discretization_parameter,
                        "roi": self.roi
                    }

        else:

            self.config = {
                "feature_extractor": self.feature_extractor,
                "modality": self.modality,
                "resampling": int(self.resampling),
                "resampling_method": self.resampling_method,
                "interpolator": self.interpolator,
                "normalization": int(self.normalization),
                "normalization_scale": self.normalization_scale,
                "re_segmentation": int(self.re_segmentation),
                "re_segmentation_method": self.re_segmentation_method,
                "discretization_method": self.discretization_method,
                "discretization_parameter": self.discretization_parameter,
                "roi": self.roi
            }

        for key, value in self.config.items():
            if value is not None:
                if key == "feature_extractor":
                    config_string += "%s_" % (value)
                if key == "modality":
                    config_string += "%s_" % (value)
                if key == "resampling":
                    config_string += "%s%s_" % ("res", value)
                if key == "resampling_method":
                    config_string += "%s%s_" % ("resm", value)
                if key == "interpolator":
                    config_string += "%s%s_" % ("interp", value)
                if key == "normalization":
                    config_string += "%s%s_" % ("norm", value)
                if key == "normalization_scale":
                    config_string += "%s%s_" % ("norms", value)
                if key == "re_segmentation":
                    config_string += "%s%s_" % ("reseg", value)
                if key == "re_segmentation_method":
                    config_string += "%s%s_" % ("resegm", value)
                if key == "discretization_method":
                    config_string += "%s%s_" % ("discm", value)
                if key == "discretization_parameter":
                    config_string += "%s%s_" % ("discp", value)
                if key == "roi":
                    config_string += "%s%s_" % ("roi", value)
                #config_string += "%s%s_" % (key, value)

        config_string = config_string[:-1] # delete the last "_" character

        # write configuration into csv file
        df = pd.DataFrame.from_dict(self.config, orient="index")
        df.to_csv(config_string+".csv", header=False)

        return config_string