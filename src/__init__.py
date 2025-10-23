import sys
import os

# get current directory path for importing modules
#current_dir = os.path.abspath(os.getcwd())
#sys.path.append(current_dir)
path = os.getcwd()
sys.path.append('src')



# +
# RPTK Modlules
#from src.config.Config_string_generator import *
#from src.config.Experiment_setup import *
#from src.config.Experiment_config import *
#from src.config.Log_generator_config import *
#from src.config.Feature_sync_config import *
#from src.config.Transformation_config import *

#from src.feature_extraction.Extractor import *
#from src.feature_extraction.Outfile_checker import *
#from src.feature_extraction.mirp_feature_extraction import *
#from src.feature_extraction.mirp_pipeline_feature_extraction import *

# from src.feature_filtering.DataConfigurator import *
#from src.feature_filtering.Feature_correlator import *
#from src.feature_filtering.Feature_formater import *
#from src.feature_filtering.Feature_stability_filter import *
#from src.feature_filtering.Feature_variance_filter import *
#from src.feature_filtering.Radiomics_Filter_exe import *
#from src.feature_filtering.DataConfigurator import *

#from src.feature_selection.FeatureSelector import *
#from src.feature_selection.HyperparameterGenerator import *
#from src.feature_selection.SFSSelector import *

#from src.image_processing.DataHandler import *
#from src.image_processing.Image_transformer import *
#from src.image_processing.MR_Normalizer import *
#from src.image_processing.Pyradiomics_image_transformer import *
#from src.image_processing.Resampler import *
#from src.image_processing.Transform_Executer import *

#from src.segmentation_processing.SegProcessor import *
#from src.segmentation_processing.SegmentationFilters import *
#from src.segmentation_processing.Segmentation_perturbator import *
#from src.segmentation_processing.Perturbator import *
# from src.segmentation_processing.multi_segmentation_handler import *

#from src.feature_filtering import DataConfigurator
#from src.feature_extraction import Outfile_checker
#from src.segmentation_processing import SegmentationFilters
#from src.segmentation_processing import SegProcessor
#from src.config import Experiment_setup
# -

from multiprocessing import Pool
import multiprocessing
