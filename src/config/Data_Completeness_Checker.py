import pandas as pd
import os


class DataCompletenessChecker:
    """
    Checking completeness of the results in input csv and output folder
    """

    def __init__(self,
                 input_csv: pd.DataFrame,
                 output_path: str,
                 logger = None,
                 error = None
                 ):

        self.input_csv = input_csv
        self.output_path = output_path
        self.logger = logger
        self.error = error

    def csv_image_transformation(self, input_csv, kernels_in_files, transformation_out_path):

        # 1. Check for image transformation in the output
        # 1.1 Get all Images with transformation

        results = []
        if len(os.listdir(transformation_out_path)) > 0:

            # 1.2 check for image transformation in the output folder
            out_files = glob.glob(transformation_out_path + '*.nii.gz')

            partial_function = partial(self.scan_processed_image_transformations,
                                       input_csv=input_csv,
                                       kernels_in_files=kernels_in_files,
                                       )

            # Process files using multiple CPUs
            with cf.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
                # Use tqdm with map to show progress
                for result in tqdm.tqdm(executor.map(partial_function, list(set(out_files))), total=len(list(set(out_files))), desc="Checking transformed Image Quality"):
                    if not result is None:
                        results.append(result)

            if len(results) > 0:
                self.logger.info("Found {} processed Image Transformation files not included in the output.".format(str(len(results))))
                print("Found {} processed Image Transformation files not included in the output.".format(str(len(results))))

                for entry in results:
                    input_csv = pd.concat([input_csv, entry], ignore_index=True)

        return input_csv


    def scan_processed_image_transformations(self,
                                             trans_file: str,
                                             input_csv: pd.DataFrame,
                                             kernels_in_files: dict,):
        """
        Scan output dir for transformed images and add them to the input if existing
        :param input_csv: pd.DataFrame with input parameters
        :param trans_file: output file where transformed images
        :param kernels_in_files: dict with the formatted kernel: kernel pattern in the file
        :return: input_csv with transformed images included from the output folder
        """

        transformaed_images = input_csv[~input_csv["Image_Transformation"].isna()]
        df = None

        # Transformation not in the output
        if trans_file in transformaed_images["Image"]:
            return None

        else:
            # need to add the missing transformation to input pd.DataFrame
            for ID in set(input_csv["ID"]):

                # get a unique ID if the ID could be a subset of another ID
                if str(ID + "_") in os.path.basename(trans_file):

                    # get the right kernel and add it to
                    for kernel, file_kernel in kernels_in_files.items():
                        if file_kernel in os.path.basename(trans_file):
                            # Add all combinations of MaskTransformation
                            for mask in input_csv.loc[input_csv["ID"] == ID, "Mask"]:
                                # add file to input pd.DataFrame
                                df = self.add_new_csv_entry(
                                                ID=ID,
                                                img_path=trans_file,
                                                seg_path=mask,
                                                modality=input_csv.loc[input_csv["Mask"] == mask, "Modality"].to_list(),
                                                roi_label=input_csv.loc[input_csv["Mask"] == mask, "ROI_Label"].to_list(),
                                                image_transformation=kernel,
                                                mask_transformation=input_csv.loc[
                                                    input_csv["Mask"] == mask, "Mask_Transformation"].to_list(),
                                                timepoint=input_csv.loc[input_csv["Mask"] == mask, "Timepoint"].to_list(),
                                                rater=input_csv.loc[input_csv["Mask"] == mask, "Rater"].to_list(),
                                                prediction_label=input_csv.loc[
                                                    input_csv["Mask"] == mask, "Prediction_Label"].to_list(),
                                                df=pd.DataFrame())
                            break

        return df

    def add_missing_image_transformations(self, input_csv: pd.DataFrame, kernels_in_files:dict):
        """
        Search for missing or failed Image transformations and add them to a dict to process those image transformations
        :param input_csv: pd.DataFrame input csv
        :param kernels_in_files: dict with formatted kernel names: name patterns in file
        :return trans_to_process: dict with formatted kernel names: list of files
        """

        trans_to_process = {}

        for ID in tqdm.tqdm(set(input_csv["ID"]), total=len(set(input_csv["ID"])),
                            desc="Scanning for Missing Transformations"):

            # 1.3 check each ID for missing kernel transformation:
            SamplesOI = input_csv[input_csv["ID"] == ID]

            # check for non processed image transformations
            for formatted_kernel in kernels_in_files:

                # image transformation not performed for this sample
                if formatted_kernel not in SamplesOI["Image_Transformation"]:
                    if formatted_kernel in trans_to_process:
                        trans_to_process[formatted_kernel] += SamplesOI.loc[
                            SamplesOI["Image_Transformation"].isna(), "Image"].tolist()
                    else:
                        trans_to_process[formatted_kernel] = SamplesOI.loc[
                            SamplesOI["Image_Transformation"].isna(), "Image"].tolist()

            # check if image transformation files exist
            for trans_file in SamplesOI.loc[~SamplesOI["Image_Transformation"].isna(), "Image"].tolist():
                # check if the transformation file does not exist
                if not os.path.isfile(trans_file):
                    self.error.warning("File {} does not exist, need to redo transformation!".format(trans_file))
                    kernel = SamplesOI.loc[SamplesOI["Image"] == trans_file, "Image_Transformation"].tolist()
                    for k in kernel:
                        if k in trans_to_process:
                            for img in list(set(SamplesOI.loc[SamplesOI["Image_Transformation"].isna(), "Image"].to_list())):
                                if img not in trans_to_process[k]:
                                    trans_to_process[k].append(img)
                        else:
                            trans_to_process[k] = list(set(SamplesOI.loc[
                                SamplesOI["Image_Transformation"].isna(), "Image"].to_list()))

        return trans_to_process