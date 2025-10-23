import os
import logging
import sys
import json
from pathlib import Path


class out_dir_generator:
    def __init__(self,
                 logger: logging = logging.getLogger(__name__),
                 error=None,
                 RunID: str = "",
                 out_path: str = "",
                 extractors=None,
                 config_file=None,
                 use_previous_output: bool = False,
                 ):

        self.logger = logger
        self.error = error
        self.RunID = RunID
        self.out_path = out_path
        self.extractors = extractors
        self.config_file = config_file
        self.use_previous_output = use_previous_output

        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.out_path + "RPTK_run_" + self.RunID + ".err",
                logger_topic="RPTK Failure"
            ).generate_log()

        if self.extractors is None:
            self.error.warning("No extractors defined! Take MIRP and PyRadiomics as default!")
            self.extractors = ["MIRP", "PyRadiomics"]

    @staticmethod
    def create_folders(out_path: str = "", extractor: str = ""):
        """
        Create folders for a RPTK run for each extractor and each step of the pipeline if not existing.
        :parameter out_path: path to the output folder
        :parameter extractor: name of the extractor (e.g. MIRP, PyRadiomics)
        :return: folders
        """

        Path(out_path + extractor).mkdir(parents=True, exist_ok=True)
        Path(out_path + extractor + "/configs").mkdir(parents=True,
                                                            exist_ok=True)  # create config folder for configuration of extraction, preprocessing, and selection
        configs_dir = out_path + extractor + "/configs"

        Path(out_path + extractor + "/plots").mkdir(parents=True,
                                                          exist_ok=True)  # saving all plots from each step
        plots_dir = out_path + extractor + "/plots"
        Path(out_path + extractor + "/plots/ICC").mkdir(parents=True, exist_ok=True)  # saving ICC plots
        icc_dir = out_path + extractor + "/plots/ICC"
        Path(out_path + extractor + "/plots/SHAP").mkdir(parents=True, exist_ok=True)  # saving SHAP plots
        shap_dir = out_path + extractor + "/plots/SHAP"
        Path(out_path + extractor + "/extracted_features").mkdir(parents=True,
                                                                       exist_ok=True)  # saving extracted features
        extracted_features_dir = out_path + extractor + "/extracted_features"
        Path(out_path + extractor + "/filtered_features").mkdir(parents=True, exist_ok=True)
        filtered_features_dir = out_path + extractor + "/filtered_features"
        Path(out_path + extractor + "/extracted_features/tmp").mkdir(parents=True,
                                                                           exist_ok=True)  # saving raw extracted features
        extracted_features_tmp_dir = out_path + extractor + "/extracted_features/tmp"
        Path(out_path + extractor + "/selected_features").mkdir(parents=True,
                                                                      exist_ok=True)  # saving selected features
        selected_features_dir = out_path + extractor + "/selected_features"
        Path(out_path + "preprocessed_data").mkdir(parents=True, exist_ok=True)  # saving preprocessed data
        preprocessed_data_dir = out_path + "preprocessed_data"

        Path(out_path + "preprocessed_data/perturbed_seg").mkdir(parents=True,
                                                                  exist_ok=True)  # saving perturbed segmentations

        Path(out_path + "preprocessed_data/plots").mkdir(parents=True, exist_ok=True)  # saving Data plots
        data_dir = out_path + "preprocessed_data/plots"

        perturbed_seg_dir = out_path + "preprocessed_data/perturbed_seg"
        Path(out_path + "preprocessed_data/not_accepted_perturbed_seg").mkdir(parents=True,
                                                                          exist_ok=True)

        Path(out_path + "preprocessed_data/accepted_perturbed_seg").mkdir(parents=True,
                                                                           exist_ok=True)  # saving accepted perturbed segmentations

        accepted_perturbed_seg_dir = out_path + "preprocessed_data/accepted_perturbed_seg"
        # Path(out_path + "preprocessed_data/processed_seg").mkdir(parents=True,
        #                                                           exist_ok=True)  # saving processed segmentations
        processed_seg_dir = out_path + "preprocessed_data/processed_seg"
        #Path(out_path + "preprocessed_data/transformed_images").mkdir(parents=True,
        #                                                               exist_ok=True)  # saving transformed images
        transformed_images_dir = out_path + "preprocessed_data/transformed_images"

        resampled_images_dir = out_path + "preprocessed_data/resampled/img"
        #Path(resampled_images_dir).mkdir(parents=True, exist_ok=True)

        resampled_seg_dir = out_path + "preprocessed_data/resampled/seg"
        #Path(resampled_seg_dir).mkdir(parents=True, exist_ok=True)

        return configs_dir, \
               plots_dir, \
               icc_dir, \
               shap_dir, \
               extracted_features_dir, \
               filtered_features_dir, \
               extracted_features_tmp_dir, \
               selected_features_dir, \
               preprocessed_data_dir, \
               perturbed_seg_dir, \
               accepted_perturbed_seg_dir, \
               processed_seg_dir, \
               transformed_images_dir, \
               resampled_images_dir, \
               resampled_seg_dir

    def create_out_dir(self, RunID: str = "", out_path: str = ""):
        """
        Create output directory for the current run
        """

        configs_dir = ""
        plots_dir = ""
        icc_dir = ""
        shap_dir = ""
        extracted_features_dir = ""
        filtered_features_dir = ""
        extracted_features_tmp_dir = ""
        selected_features_dir = ""
        preprocessed_data_dir = ""
        perturbed_seg_dir = ""
        accepted_perturbed_seg_dir = ""
        processed_seg_dir = ""
        transformed_images_dir = ""
        resampled_images_dir = ""
        resampled_seg_dir = ""

        if RunID == "":
            RunID = self.RunID
        if out_path == "":
            out_path = self.out_path

        if self.use_previous_output:
            print("Use structure from previous RPTK run ...")

        out_path_sepperated = out_path.split("/")

        # if the output folder does not exist, create it --> there is no previous run
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            self.logger.info("Output directory: " + str(out_path))
            print("Output directory: " + str(out_path))
        else:
            # check if the outfolder contains the runID --> use previous output
            if out_path_sepperated[-2] == RunID and self.use_previous_output:
                self.logger.info("Output directory: " + str(out_path))
                print("Output directory: " + str(out_path))
            elif out_path_sepperated[-2] == RunID and not self.use_previous_output:
                if RunID != out_path_sepperated[-2]:
                    self.error.error("Output folder contains a folder with the a different runID! " +
                                     "Please insert the correct runID or set use_previous_output = True!")
                    RunID = out_path_sepperated[-2]
            elif out_path_sepperated[-2] != RunID and self.use_previous_output:
                self.error.error("Output folder does not contain a folder with the same runID! " +
                                 "Please insert a runID of a previous run in out_path or set use_previous_output = False!")
                sys.exit("ERROR: Output folder does not contain a folder with the same runID! " +
                         "Please  insert a runID of a previous run in out_path or set use_previous_output = False!")
            else:
                self.logger.info("Output directory: " + str(out_path + RunID))
                print("Output directory: " + str(out_path + RunID))

        # check if there is no folder containing the runID in the out_path
        if not os.path.exists(out_path + RunID) and not self.use_previous_output and out_path_sepperated[-2] != RunID:
            os.makedirs(out_path + RunID)
            self.logger.info("Output directory: " + str(out_path + RunID))
            print("Output directory: " + str(out_path + RunID))

        if not self.use_previous_output and out_path_sepperated[-2] != RunID:
            # append the runID to the out_path
            out_path = out_path + RunID

        if self.extractors is None:
            self.error.error("No extractors defined! Can not process! Review the RPTK config!")
            sys.exit("ERROR: No extractors defined! Can not process! Review the RPTK config!")
        else:
            # create files for each extractor if folders are not existing yet
            for extractor in self.extractors:
                configs_dir, \
                plots_dir, \
                icc_dir, \
                shap_dir, \
                extracted_features_dir, \
                filtered_features_dir, \
                extracted_features_tmp_dir, \
                selected_features_dir, \
                preprocessed_data_dir, \
                perturbed_seg_dir, \
                accepted_perturbed_seg_dir, \
                processed_seg_dir, \
                transformed_images_dir, \
                resampled_images_dir, \
                resampled_seg_dir  = self.create_folders(out_path=out_path, extractor=extractor)

                # write RPTK config file into config folder
                if self.config_file is not None:
                    with open(configs_dir + "/RPTK_config.json", 'w') as f:
                        json.dump(self.config_file, f)

        return configs_dir, \
               plots_dir, \
               icc_dir, \
               shap_dir, \
               extracted_features_dir, \
               filtered_features_dir, \
               extracted_features_tmp_dir, \
               selected_features_dir, \
               preprocessed_data_dir, \
               perturbed_seg_dir, \
               accepted_perturbed_seg_dir, \
               processed_seg_dir, \
               transformed_images_dir, \
               resampled_images_dir, \
               resampled_seg_dir