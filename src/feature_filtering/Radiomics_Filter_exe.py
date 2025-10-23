import glob
import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from tqdm import tqdm
import re
from math import isnan
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from detect_delimiter import detect
import datetime
import warnings
import multiprocessing
from multiprocessing import Pool
from pandas.api.types import is_string_dtype
import sys

from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.feature_filtering.ibsi_feature_formater import IBSIFeatureFormater

warnings.filterwarnings("ignore")


class RadiomicsFilter:
    def __init__(self,
                 data,  # DataFrame or Path to feature files or folder containing multiple feature extraction files
                 path_to_img_seg_csv: str = "",
                 out_path: str = "",  # Path to output folder
                 # Path to CSV file which includes the path to imgs and segs for feature extraction
                 variance_threshold: float = 0.1,  # Threshold for filtering features by variance
                 correlation_threshold: float = 0.90,  # Threshold for filtering of correlating features
                 extractor: str = "PyRadiomics",  # PyRadiomics or MIRP
                 # data: pd.DataFrame = pd.DataFrame(),  # Data with with extracted radiomics features
                 longitudinal_data: bool = False,  # Working with different time points
                 multiple_ROIs: bool = False,  # Working with different ROIs (e.g. Intratumoral & Peritumoral)
                 logger: logging = None,  # Logger for logging
                 RunID: str = "",  # RunID for logging
                 n_cpu: int = 3,
                 stability_filtering: bool = True,  # Filtering features with perturbed segmentations to check for high segmentation sensitivity
                 ICC_threshold: float = 0.9,  # threshold for ICC calculations
                 log_folder=None,  # Folder to store log files in
                 error=None,
                 format_check=True,
                 peritumoral: bool = True,
                 additional_rois_to_features: bool = False, # If features should be calculated for additional rois sepperately
                 delta: bool = False,):

        self.longitudinal_data = longitudinal_data
        self.multiple_ROIs = multiple_ROIs
        self.data = data
        self.path_to_img_seg_csv = path_to_img_seg_csv
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.extractor = extractor
        self.logger = logger
        self.RunID = RunID
        self.n_cpu = n_cpu
        self.stability_filtering =  stability_filtering
        self.ICC_threshold = ICC_threshold
        self.log_folder = log_folder
        self.out_path = out_path
        self.error = error
        self.format_check = format_check
        self.peritumoral = peritumoral
        self.additional_rois_to_features = additional_rois_to_features
        self.delta = delta

        if not self.out_path.endswith("/"):
            self.out_path += "/"

        self.out_path = os.path.join(self.out_path, "filtered_features/")

        # Check if logger is provided
        # self.df_mask_transform = pd.DataFrame()
        #if self.log_folder is None:
        #    self.log_folder = os.path.join(self.out_path, "logs/")

        # Config Logger #
        if self.log_folder is not None:
            # folder to store log files in
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)

            self.logger = LogGenerator(
                log_file_name=self.log_folder + "/RPTK_feature_filtering_" + self.RunID + ".log",
                logger_topic="RPTK Feature Filtering"
            ).generate_log()

            # if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.log_folder + "/RPTK_feature_filtering_" + self.RunID + ".err",
                logger_topic="RPTK Feature Filtering error"
            ).generate_log()
        else:
            #print(self.out_path + "RPTK_feature_filtering_" + self.RunID + ".log")
            #print(self.out_path)
            self.logger = LogGenerator(
                log_file_name=self.out_path + "RPTK_feature_filtering_" + self.RunID + ".log",
                logger_topic="RPTK Feature Filtering"
            ).generate_log()

            # if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.out_path + "RPTK_feature_filtering_" + self.RunID + ".err",
                logger_topic="RPTK Feature Filtering error"
            ).generate_log()




        # check if input data is path to files/folders or DataFrame
        if isinstance(self.data, str):
            if len(self.data) > 0:
                if self.input_is_file(self.data):
                    self.logger.info("Found Feature file: " + self.data)
                    df = pd.read_csv(self.data, sep=",", index_col=0)
                    if len(df.columns) < 3:
                        df = pd.read_csv(self.data, sep=";", index_col=0)

                    self.data = df

                # elif self.input_is_folder(path_to_data):
                #     self.path_to_folder = path_to_data
                # else:
                #     if self.data.empty:
                #         self.error.warning(path_to_data + " not found!")
            else:
                self.error.error("Formatting error in path to data provided for feature filtering!")
                raise ValueError("Formatting error in path to data provided for feature filtering!")

        self.string_df = pd.DataFrame()
        self.string_parameter = pd.DataFrame()

        # Check if the path goes to a file or to the folder with multiple files

        # List to get configuration of transformation kernels with regex

    def input_is_folder(self, path):

        is_folder = os.path.isdir(path)

        return is_folder

    def input_is_file(self, path):

        is_file = os.path.isfile(path)

        return is_file

    def read_data(self):
        """
        Read the input and put it into a data frame.
        Possible accepted input formats:
        1. Single CSV File with all the features for all samples. (PyRadiomics)
        2. Multiple CSV Files with all features from each Sample. (MIRP)
        """

        if isinstance(self.data, pd.DataFrame):
            if self.data.empty:
                self.error.warning("No data provided for feature filtering!")
                raise ValueError("No data provided for feature filtering!")

            self.logger.info("### Unfiltered Radiomics Data ###")
            self.logger.info(str(self.data.shape[0]) + " Samples")
            self.logger.info(str(self.data.shape[1]) + " Features")

        # if no data is given, read it from the file
        elif isinstance(self.data, str):

            # read files and check for separator

            # 1. Check if it is only 1 csv File
            if self.input_is_file(self.data):

                if not self.data.endswith(".csv"):
                    self.logger.error("Input File {} is not a csv file!".format(self.data))
                    raise ValueError("No CSV File provided for Feature Filtering!")

                self.logger.info("Reading CSV File: " + self.data)

                df = pd.read_csv(self.data, sep=",")

                if len(df.columns) < 3:
                    self.logger.info("Separator is not \",\", trying \";\"")
                    df = pd.read_csv(self.data, sep=";")

                self.data = df

            # if there are multiple csv files (MIRP)
            elif self.input_is_folder(self.data):

                self.logger.info("Reading CSV Files ...")
                files = pd.DataFrame(columns=["File_path", "IDs"])

                if len(self.path_to_img_seg_csv) > 0:

                    img_seg_df = pd.read_csv(self.path_to_img_seg_csv)
                    feature_files = os.listdir(self.path_to_folder)

                    for index, row in tqdm(img_seg_df.iterrows(), desc="Reading Img/Seg Path File",
                                           total=img_seg_df.shape[0]):
                        for path in feature_files:
                            if path.endswith(".csv"):
                                if os.path.basename(row["Image"]).endswith(".nii.gz"):
                                    if os.path.basename(path).startswith(
                                            os.path.basename(row["Image"])[:-len(".nii.gz")]):
                                        files = files.append(
                                            {'File_path': path,
                                             'IDs': os.path.basename(row["Image"])[:-len(".nii.gz")]},
                                            ignore_index=True)
                                if os.path.basename(row["Image"]).endswith(".nii"):
                                    if os.path.basename(path).startswith(os.path.basename(row["Image"])[:-len(".nii")]):
                                        files = files.append(
                                            {'File_path': path, 'IDs': os.path.basename(row["Image"])[:-len(".nii")]},
                                            ignore_index=True)

                    files = files.sort_values(by=["File_path"])
                    if self.path_to_folder.endswith("/"):
                        files['File_path'] = self.path_to_folder + files['File_path'].astype(str)
                    else:
                        files['File_path'] = self.path_to_folder + "/" + files['File_path'].astype(str)

                    feature_files = files["File_path"].tolist()

                    li = []

                    for csv in tqdm(feature_files, desc="Reading Feature Files"):
                        df = pd.read_csv(csv, index_col=None, header=0, sep=";")
                        if len(df.columns) < 3:
                            self.logger.info("Separator is not \";\", trying \",\"")
                            df = pd.read_csv(csv, index_col=None, header=0, sep=",")

                        li.append(df)

                    self.data = pd.concat(li, axis=0, ignore_index=True)

                else:
                    self.logger.error("You provided a folder as input. "
                                      "For multiple input files please provide file paths in a CSV "
                                      "with Image and Mask Column!")
                    raise ValueError("You provided a folder as input. "
                                     "For multiple input files please provide file paths in a CSV "
                                     "with Image and Mask Column!")

            self.logger.info("### Raw Radiomics Data ###")
            self.logger.info(str(self.data.shape[0]) + " Samples")
            self.logger.info(str(self.data.shape[1]) + " Features" + "\n")

        else:
            self.logger.error("Data loading failed! Please Provide a valid input format: pd.DataFrame(), "
                              "path to file or folder!")
            raise ValueError("Data loading failed! Please Provide a valid input format: pd.DataFrame(), "
                             "path to file or folder!")

        # check for failed feature extractions
        nan_features = [i for i in self.data.columns if self.data[i].isnull().any()]

        if ("Image" in nan_features) or ("Mask" in nan_features):
            samples_to_drop = self.data[(self.data['Image'].isna()) or (self.data['Mask'].isna())].shape[0]
            print(f"Number of samples to be dropped as they do not have Image or Mask assigned: {samples_to_drop}")
            self.error.warning(f"Number of samples to be dropped as they do not have Image or Mask assigned: {samples_to_drop}")
            self.data = self.data.dropna(subset=['Image', 'Mask'])
            nan_features = [i for i in self.data.columns if self.data[i].isnull().any()]

        nan_features = list(filter(lambda x: x != "Mask_Transformation", nan_features))
        nan_features = list(filter(lambda x: x != "Image_Transformation", nan_features))
        nan_features = list(filter(lambda x: x != "Rater", nan_features))

        if len(nan_features) > 0:
            
            print("Failed feature extraction for {} Features: {}".format(str(len(nan_features)), str(nan_features)))
            self.error.warning("Failed feature extraction for {} Features: {}".format(str(len(nan_features)), str(nan_features)))
            self.data.drop(nan_features, axis = 1, inplace = True) 


    ## 2
    def remove_duplicated_feature(self, df: pd.DataFrame):
        """
        Drops duplicated value columns of pandas dataframe.
        """
        dataframe_T = df.T
        df = dataframe_T.drop_duplicates(keep='first').T

        self.logger.info("Duplicated Features: " + str(dataframe_T.duplicated().sum()) + "\n")

        self.logger.info("Corrected Sample Size:")
        self.logger.info(str(df.shape[0]) + " Samples")
        self.logger.info(str(df.shape[1]) + " Features\n")

        return df

    def nan_in_columns(self, df):
        """
        Filter for columns only containing NaN values.
        :param df: pd.DataFrame to be filtered
        :return: list of nan features
        """
        nan_columns = df.columns[df.isna().all()].tolist()

        if len(nan_columns) == 0:
            self.logger.info("Found No Features with NaN")
        else:
            self.error.warning("Found Features with NaN: " + str(len(nan_columns)))

        for column in nan_columns:
            self.error.warning("Feature only containing NaN: " + column)

        return nan_columns

    def get_samples_with_nan(self, df):
        """
        extract all samples containing nan values to another dataframe
        :return: dataframe with all samples containing nan values
        """

        before = df.shape[0]
        df = df.dropna(how='all')
        after = df.shape[0]

        if before != after:
            self.logger.info("Drop Samples " + str(before - after) + " with only NaN!")

        # select all rows with nan
        nan_df = df[df.isna().any(axis=1)]
        nan_df = nan_df.loc[:, nan_df.isnull().any()]

        return nan_df

    ## 3
    def normalize(self, df: pd.DataFrame):
        """
        Perform Z-score normalization for each feature (column) in the dataframe and rename it with adding the suffix _zscore
        :param df: Radiomics features without any sting features
        :return: normalized features
        """
        # clean from strings or file path and take ID as an index
        # self.data = self.data.groupby(level=0).agg(lambda x: ','.join(x))
        # self.data = self.data.join(df)

        # TODO:
        # check for nans in columns and rows and give error message with names of features
        # drop columns where all values are nan
        # df = df.dropna(how='all').dropna(how='all', axis=1) # was on before

        normalized = False

        self.logger.info("Normalizing Data ...")
        for col in df.columns:
            if "_zscore" in col:
                normalized = True
            if df[col].isnull().values.any():
                self.error.warning("NaN in Feature: " + col)
        if not normalized:
            df = df.astype('float64')

            # Normalization
            df_zscore = pd.DataFrame()

            cols = list(df.columns)

            # Normalize on z-scores for each column
            for col in cols:
                if '_zscore' not in col:
                    col_zscore = col + '_zscore'
                    df_zscore[col_zscore] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

            df = df_zscore
        else:
            self.logger.info("Data already normalized!")

        return df

    ## 4
    def variance_filter(self, df: pd.DataFrame):

        constant_filter = VarianceThreshold(threshold=self.variance_threshold)
        constant_filter.fit(df)
        constant_columns = [column for column in df.columns
                            if column not in df.columns[constant_filter.get_support()]]

        df.drop(columns=constant_columns, inplace=True)

        self.logger.info("Quasi-Constant features: " + str(len(constant_columns)) + "\n")

        return df

    ## 5
    #def corr_filter(self, df):

    #    # removes correlating features with grater then threshold from df
    #    correlation_matrix = df.corr().abs()

    #    # go through symmetric matrix
    #    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))

    #    # filter of columns bigger then threshold
    #    corr_feat = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]

    #    csv2 = df[corr_feat].copy()

    #    # drop correlating features
    #    df = df.drop(columns=corr_feat)

    #    return df, csv2

    def corr_filter(self, df: pd.DataFrame):
        """
        Remove highly correlated columns (|r| > self.correlation_threshold).
        Keeps the first column (by current column order) and drops later ones.

        Returns
        -------
        df_reduced : pd.DataFrame
            DataFrame with correlated features removed
        df_dropped : pd.DataFrame
            DataFrame containing only the dropped (highly correlated) features
        """
        df_in = df.copy()

        # Correlate numeric columns only
        num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            # Nothing numeric: nothing to drop
            return df_in, df_in.iloc[:, 0:0].copy()

        # Absolute Pearson correlation
        corr = df_in[num_cols].corr().abs()

        # Upper triangle (exclude diagonal) to avoid double counting
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        thr = float(self.correlation_threshold)

        # Drop any column that has at least one correlation > threshold to an earlier column
        to_drop = [col for col in upper.columns if (upper[col] > thr).any()]

        df_dropped = df_in[to_drop].copy()
        df_reduced = df_in.drop(columns=to_drop)

        return df_reduced, df_dropped

    ## 7
    def split_by_ROI(self):
        # Split df depending on ROI
        intra = self.data.loc[self.data["ROI"] == 1]  # "Intratumoral"
        peri = self.data.loc[self.data["ROI"] == 0]  # "Peritumoral"

        return intra, peri

    def correlation_surrogates_detection(self, df: pd.DataFrame):
        """
        Detect surrogate features and filter by correlation (NO ROI grouping).

        Returns
        -------
        df_filtered : pd.DataFrame
            Original df with highly correlated features removed (keeps first-in-order).
        corr_simple_count : pd.Series
            For ALL numeric features, the count of |corr| > threshold (descending).
        surrogates : list[str]
            KEPT features that serve as representatives for the dropped ones.
        """
        # 1) Filter correlated features (pairwise threshold, keep-first rule)
        df_kept, df_dropped = self.corr_filter(df)

        self.logger.info("### 10.1 Kept (Non-Correlating by rule) ###")
        self.logger.info(f"{df_kept.shape[0]} Samples")
        self.logger.info(f"{df_kept.shape[1]} Features\n")

        self.logger.info("### 10.2 Dropped (Highly Correlating) ###")
        self.logger.info(f"{df_dropped.shape[0]} Samples")
        self.logger.info(f"{df_dropped.shape[1]} Features\n")

        # 2) Correlation summary over ALL numeric features (informative counts)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        thr = float(self.correlation_threshold)

        if num_cols:
            C_all = df[num_cols].corr().abs()
            corr_simple_count = (C_all.where(np.triu(np.ones(C_all.shape), k=1).astype(bool)) > thr).sum()
            corr_simple_count = corr_simple_count.sort_values(ascending=False)
        else:
            corr_simple_count = pd.Series(dtype=float)

        # 3) Surrogates: for each dropped feature, find its best kept counterpart
        #    (the kept feature with the strongest |corr| > threshold)
        surrogates = []
        if num_cols and df_dropped.shape[1] > 0 and df_kept.shape[1] > 0:
            C_abs = df[num_cols].corr().abs()

            # Preserve original column order for deterministic tie-breaking
            col_index = {c: i for i, c in enumerate(df.columns)}
            kept_set = set(df_kept.columns)
            dropped_cols = [c for c in df_dropped.columns if c in C_abs.columns]

            surrogate_set = set()
            for col in dropped_cols:
                if col not in C_abs.columns:
                    continue
                # candidates: KEPT features with |r| > threshold
                candidates = [k for k in kept_set if k in C_abs.columns and C_abs.loc[k, col] > thr]
                if candidates:
                    # pick the strongest correlation; break ties by original order
                    candidates.sort(key=lambda k: (C_abs.loc[k, col], - (len(df.columns) - col_index[k])), reverse=True)
                    surrogate_set.add(candidates[0])

            # Sort surrogates by original column order
            surrogates = sorted(list(surrogate_set), key=lambda x: col_index[x])

        self.logger.info("### 10.3 Identified Surrogate (Kept) Features ###")
        self.logger.info(f"{len(surrogates)} Features\n")

        self.logger.info("### 10.4 Removed Correlating Features ###")
        self.logger.info(f"{df_dropped.shape[1]} Features\n")

        # 4) Final filtered df = ONLY the kept set (do NOT re-add dropped)
        df_filtered = df_kept.copy()

        self.logger.info("Corrected Sample Size:")
        self.logger.info(f"Samples: {df_filtered.shape[0]}")
        self.logger.info(f"Features: {df_filtered.shape[1]}\n")

        return df_filtered, corr_simple_count, surrogates

        """ def correlation_surrogates_detection(self, df: pd.DataFrame):

            non_corr_df, corr_df = self.corr_filter(df)

            self.logger.info("### 10.1 Non-Correlating Features ###")
            self.logger.info(str(non_corr_df.shape[0]) + " Samples")
            self.logger.info(str(non_corr_df.shape[1]) + " Features\n")

            self.logger.info("### 10.2 Correlating Features ###")
            self.logger.info(str(corr_df.shape[0]) + " Samples")
            self.logger.info(str(corr_df.shape[1]) + " Features\n")

            corr = corr_df.corr()
            corr_simple_count = corr.copy()[corr > self.correlation_threshold].count().sort_values(ascending=False)
            corr_count = corr[corr > self.correlation_threshold].copy()

            # plot corr matrix
            # plt.figure(figsize=(11, 8))
            # sns.heatmap(corr_count, cmap="Greens", annot=False)
            # plt.show()
            corr = corr[corr > self.correlation_threshold].count()

            # corr_simple_count = corr_count[corr > self.correlation_threshold].count().sort_values(ascending=False)

            # Sort values (most counted comes first)
            corr_count["count"] = corr_count[corr > self.correlation_threshold].count()
            corr_count = corr_count.sort_values(by=['count'], ascending=False)
            corr_count.drop(['count'], axis=1, inplace=True)

            corr = corr.sort_values(ascending=False)

            corr_features = []
            for i in corr.keys().values:
                corr_features.append(i)

            df_corr = df[corr_features]

            corr = df_corr.corr()

            # Get surrogate features for feature reduction
            unique_features = []
            columns = corr.columns

            for col in columns:
                if col in corr.columns:
                    # 1. extract features which are highly correlating with other features for a surrogate
                    # hcorr = corr.loc[corr[col] > 0.9, corr[col] > 0.9]
                    unique_features.append(col)
                    # hcorr.drop(hcorr[hcorr["cm_auto_corr_d1_3d_v_mrg_fbs_w6.0_zscore"]>0.9],inplace=True)

                    # 2. update df for columns of interest
                    corr = corr.loc[corr[col] < self.correlation_threshold, corr[col] < self.correlation_threshold]

            surrogates = unique_features

            self.logger.info("### 10.3 Identified Surrogate Features ###")
            self.logger.info(str(len(surrogates)) + " Features\n")

            self.logger.info("### 10.4 Removed Correlating Features ###")
            self.logger.info(str(corr_df.shape[1] - len(surrogates)) + " Features\n")

            # add non correlating features to get final dataset
            for i in non_corr_df.columns:
                unique_features.append(i)

            df_filtered = df[unique_features]
            # self.data = df_filtered
            # 6 Config df
            # df_filtered = config_df(df_filtered)

            self.logger.info("Corrected Sample Size:")
            self.logger.info("Samples: " + str(self.data.shape[0]))
            self.logger.info("Features: " + str(self.data.shape[1]) + "\n")

            return df_filtered, corr_simple_count, surrogates
        """
    def get_data(self):
        return self.data

    @staticmethod
    def add_mulitple_Mask_to_sample(df):
        """
        This function checks for duplicate indices in a pandas DataFrame and consolidates the
        'Mask' values for rows with the same 'Image' value into a list in the 'Mask' column.
        
        Args:
        df (pd.DataFrame): Input DataFrame with potential duplicate indices.

        Returns:
        pd.DataFrame: Squeezed DataFrame with unique indices and consolidated 'Mask' values.
        """
        before = df.copy()

        index_name = before.index.name
        # remove multile ROI
        before = before[~df.index.duplicated()].copy()
        # remove Mask as it only shows fist ROI
        before.drop(['Mask'], axis = 1, inplace = True) 

        before[index_name] = before.index

        # before.reset_index(drop=False)
        before.set_index("Image",drop=False, inplace=True)

        # Group by 'Image' and consolidate 'Mask' values into lists
        squeezed_df = (
            df.groupby("Image", group_keys=False)
            .agg({"Mask": lambda x: list(x)})
        )

        final = pd.concat([before, squeezed_df], axis=1)

        final.set_index(index_name, drop=False, inplace=True)
        
        return final

    def update_df(self, df):
        """
        Sync the dataframe with the filtered data by sync index
        :parameter: df: dataframe to be updated by index
        """

        if not self.data.index[0] in df.index:
            if "config" in df.columns:
                if self.data.index[0] in df["config"].values:
                    df.set_index("config", drop=True, inplace=True)
                else:
                    self.error.error("Dataframe index does not match with the data index")
                    self.error.error("Dataframe index: " + str(df.index))
                    self.error.error("Data index: " + str(self.data.index))
                    print("Dataframe index {} does not match with the data index {}".format(str(df.index[0]),str(self.data.index[0])))
            else:
                self.error.error("Dataframe index does not match with the data index")
                self.error.error("Dataframe index: " + str(df.index))
                self.error.error("Data index: " + str(self.data.index))
                print("Dataframe index {} does not match with the data index {}".format(str(df.index[0]),str(self.data.index[0])))

        # check if _ is in the index of the data
        has_underscore = any('_' in idx for idx in df.copy().index)
        corrected_index = []
        for i in df.copy().index:
            
             # check if _ is in the index of the data
            if has_underscore:
                
                data_index = ""

                if i not in self.data.copy().index:
                    ID = str(i).split("_")[0]
                    

                    # TODO: risky as there migth be wrong string parameters
                    for j in self.data.copy().index:
                        if str(j).startswith(ID):
                            data_index = j
                            break

                    # sample not found in data
                    if data_index == "":
                        df = df.drop(i)
                        print("Sample " + str(i) + " not in data index. Dropped from dataframe!")
                        self.error.warning("Sample " + str(i) + " not in data index. Dropped from dataframe!")
                    else:
                        if data_index != "":
                            corrected_index.append(data_index)
                else:
                    if data_index != "":
                        corrected_index.append(data_index)
            
        if len(corrected_index) > 0:
            df.index = corrected_index
        
        if len(df) > 0:
            # if there are multiple Mask you add the masks to the same row assigned to the same image
            if len(df) != len(self.data):
                if "Mask" in df.columns:
                    if "Image" in df.columns:
                        df = RadiomicsFilter.add_mulitple_Mask_to_sample(df)
                    else:
                        print("Image is missing in string data.")
                        self.error.warning("Image is missing in string data.")
                else:
                    print("Mask is missing in string data.")
                    self.error.warning("Mask is missing in string data.")

            data_index = self.data.copy().index
            df_index = df.index
            mask = data_index.isin(df_index)
            result = df.loc[mask]
        else:
            result = pd.DataFrame()

        return result

    # def config_mirp_features(self):

    def searching_for_nan(self, df: pd.DataFrame, out_file_name: str):

        self.logger.info("### Features with NaN ###")
        # Filter for columns only containing nan
        nan_features_list = self.nan_in_columns(df=df)
        # nan_features = self.data[nan_features_list]
        df = df.drop(nan_features_list, axis=1)

        nan_df = self.get_samples_with_nan(df=df)
        self.logger.info("### Samples with NaN ###")
        # TODO: filter and write seg changed samples to csv

        if nan_df.shape[0] > 0:
            self.error.warning("Found Samples and Features with NaN:" +
            "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(nan_df.shape[0]) + " Samples" +
            "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(nan_df.shape[1]) + " Features")

            # if there are just a few features missing
            if nan_df.shape[1] < 10:
                df = df.drop(columns=nan_df.columns)
            else:
                df = df.drop(nan_df.index)

            self.logger.info("Corrected Feature Space:" +
                               "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(df.shape[0]) + " Samples"+
                               "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(df.shape[1]) + " Features")

            # write NaN samples to csv
            self.logger.info("Write NaN samples to csv: " + str(os.path.dirname(self.out_path)) + "/" +
                             out_file_name)

            if not os.path.isdir(os.path.dirname(self.out_path) + "/tmp/"):
                Path(self.out_path + "/tmp/").mkdir(parents=True, exist_ok=True)

            nan_df.to_csv(os.path.dirname(self.out_path) + "/tmp/" + out_file_name)

                # print("ERROR:" + os.path.dirname(self.out_path) + "/tmp/" + " does not exist!" + "Check your environment!")
                #raise Exception("ERROR:" + os.path.dirname(self.out_path) + "/tmp/" + " does not exist!" +
                #                "Check your environment!")
        else:
            self.logger.info("Found no Samples with NaN \n")

        return df

    def config_mirp_features(self):
        """
        Config MIRP features
        :return:
        """
        ### Configured

        img_seg_df = pd.read_csv(self.path_to_img_seg_csv)
        # Add data configs to mirp csv
        image = img_seg_df["Image"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])
        mask = img_seg_df["Mask"].apply(lambda x: os.path.basename(x)[:-len(".nii.gz")])

        # generate Index with Image name and Mask name for config string
        id = []
        for x, y in zip(image, mask):
            id.append(x + "_" + y)

        ID = pd.Series(index=img_seg_df.index, data=id)
        img_seg_df.index = ID

        self.data = pd.concat([img_seg_df, self.data], axis=1, join='outer')

        self.logger.info("### 1. Config MIRP features")
        self.logger.info("Corrected Sample Size:" +
        "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(self.data.shape[0]) + " Samples" +
        "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(self.data.shape[1]) + " Features\n")

    def get_DataConfigurator(self):
        """
        create Data configurator
        :return: DataConfigurator object
        """

        from rptk.src.feature_filtering.DataConfigurator import DataConfigurator

        configurator = DataConfigurator(
                                        out_path=self.out_path,
                                        logger=self.logger,
                                        error=self.error,
                                        data=self.data,
                                        multiple_ROIs=self.multiple_ROIs,
                                        longitudinal_data=self.longitudinal_data,
                                        extractor=self.extractor,
                                        path_to_img_seg_csv=self.path_to_img_seg_csv,
                                        n_cpu=self.n_cpu,
                                        RunID=self.RunID,
                                        stability_filtering=self.stability_filtering,
                                        ICC_threshold=self.ICC_threshold,
                                        format_check=self.format_check,
                                        additional_rois_to_features=self.additional_rois_to_features,
                                        peritumoral=self.peritumoral,
                                        delta=self.delta)
        return configurator

    def filter_profile_plotter(self, df):
        """
        Plot feature profile alteration
        """
        
        plot_df = df.T
        plot_df["RPTK Filtering Steps"] = plot_df.index
        plot_df = plot_df.reset_index(drop=True)

        x = plot_df.index.to_list()
        y = plot_df[self.extractor].to_list()
        labels = plot_df["RPTK Filtering Steps"].to_list()
        
        plt.xticks(x, labels, rotation='vertical')
        plt.title('Feature Filtering Profile', fontsize=16)
        plt.plot(x, y)
        plt.tight_layout(pad=4)
        plt.xlabel('RPTK Filtering Steps', fontsize=12)
        plt.ylabel('Number of Features', fontsize=12)
        
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.dirname(self.out_path) + "/feature_alteration_profile.png",
                    format='png',
                    dpi=200,
                    bbox_inches='tight')
        plt.close()
    
    def calc_mean_for_multi_ROI(self, df:pd.DataFrame, ID:str):
        """
        If there are multiple ROIs for the same sample and they do have the same label --> make mean of the samples
        :param df: Data for calculating means fro multiple ROIS sharing the same prediction label and the same timepoint
        :param ID: The ID on which duplicates are detected
        """
        mean_df = pd.DataFrame()
        pbar = tqdm(set(df[ID].copy().to_list()), desc="calculate mean for multiple ROIs")
        for id in pbar:
            pbar.set_description("calculate mean for multiple ROIs")
            tmp = df.copy()[df.copy()[ID] == id]
            if len(tmp) > 1:
                self.logger.info("Calculate mean for sample {}".format(str(id)))
                label = set(tmp.copy()["Prediction_Label"].to_list())
                timepoint_available = "Timepoint" in tmp.columns
                
                if timepoint_available:
                    if not tmp["Timepoint"].isnull().all():
                        timepoints = set(tmp["Timepoint"].to_list())
                        if len(timepoints) > 1:
                            # if there are multiple lesions for timepoints
                            for time in timepoints:
                                pbar.set_description("calculate mean for timepoint ".format(str(time)))
                                time_tmp = tmp.copy()[tmp.copy()["Timepoint"] == time]
                                time_tmp_label = set(time_tmp.copy()["Prediction_Label"].to_list())
                                if len(time_tmp_label) == 1:
                                    # lesions have the same label
                                    tmp_mean = time_tmp.groupby(by=[ID], dropna=False).mean()
                                    mean_df = pd.concat([mean_df,tmp_mean])
                                else:
                                    for lab in time_tmp_label:
                                        tmp_lab = time_tmp.copy()[time_tmp.copy()["Prediction_Label"] == lab]
                                        if len(tmp_lab) > 1:
                                            tmp_mean = tmp_lab.groupby(by=[ID], dropna=False).mean()
                                        else:
                                            tmp_mean = time_tmp.copy()
                                        mean_df = pd.concat([mean_df,tmp_mean])

                        else:
                            # if there is one timepoint
                            if len(label) == 1:
                                # lesions have the same label
                                tmp_mean = tmp.groupby(by=[ID], dropna=False).mean()
                                mean_df = pd.concat([mean_df,tmp_mean])
                            else:
                                for lab in label:
                                    tmp_lab = tmp.copy()[tmp.copy()["Prediction_Label"] == lab]
                                    if len(tmp_lab) > 1:
                                        tmp_mean = tmp_lab.groupby(by=[ID], dropna=False).mean()
                                    else:
                                        tmp_mean = tmp.copy()

                                    mean_df = pd.concat([mean_df,tmp_mean])
                # no timepoints
                else:
                    # lesions have the same label
                    if len(label) == 1:
                        tmp_mean = tmp.groupby(by=[ID], dropna=False).mean()
                        mean_df = pd.concat([mean_df,tmp_mean])
                    else:
                        # lesions have different labels
                        for lab in label:
                            pbar.set_description("calculate mean for multiple ROIs with label {}".format(str(lab)))
                            tmp_lab = tmp.copy()[tmp.copy()["Prediction_Label"] == lab]
                            if len(tmp_lab) > 1:
                                tmp_mean = tmp_lab.groupby(by=[ID], dropna=False).mean()
                            else:
                                tmp_mean = tmp.copy()
                            mean_df = pd.concat([mean_df,tmp_mean])

            else:
                # no duplications
                mean_df = pd.concat([mean_df,tmp])

        return mean_df

    def process_dataframe(self, df):
        """
        Check if string dataframe has index only from peritumoral 
        """

        # make unique columns if duplicated
        def make_unique(column_names):
            counts = {}
            unique_columns = []
            for col in column_names:
                if col in counts:
                    counts[col] += 1
                    new_col_name = f"{col}_{counts[col]}"
                    unique_columns.append(new_col_name)
                else:
                    counts[col] = 0
                    unique_columns.append(col)
            return unique_columns
        
        df.columns = make_unique(df.columns)

        # make duplicated columns unique
        # df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)

        # drop columns which are all having nan
        df = df.dropna(axis=1, how='all')
        
        if "Image" not in df.columns:
            # rename columns if Image or Mask are included in column name
            df = df.rename(columns=lambda x: 'Image' if 'Image.' in x else ('Mask' if 'Mask.' in x else x))

        if "Image" in df.columns:
            # Drop samples with NaN in Image or Mask
            samples_to_drop = df[df['Image'].isna() | df['Mask'].isna()].shape[0]
            df = df.dropna(subset=['Image', 'Mask'])
            print(f"Number of samples dropped: {samples_to_drop}")
        
        # Remove '_peritumoral' from index if present
        df.index = df.index.str.replace(r'_[0-9]*_peritumoral', '', regex=True)
        
        df = df.rename(columns=lambda x: 'ID' if 'ID' in x else x)
        
        return df

    def remove_uninformative_features(self):
        """
        Remove feature which are non informative and should not be considered for feature selection. This is a sanity check.
        """
        
        # drop morphological peritumoral features
        if self.extractor == "MIRP":
            self.data = self.data[self.data.columns.drop(list(self.data.filter(regex='peritumoral').filter(regex='morph')))]

        if self.extractor == "PyRadiomics":
            self.data = self.data[self.data.columns.drop(list(self.data.filter(regex='peritumoral').filter(regex='shape')))]

        # filtering for index features
        index_cols = self.data.copy().loc[:,self.data.columns[self.data.columns.str.contains(pat = 'index_')]].columns
        if len(index_cols) > 0:
            print(f"Found {len(index_cols)} features with index included. Dropping ... ")
            self.error.warning(f"Found {len(index_cols)} features with index included. Dropping ... ")

            # drop index features for sanity check
            self.data = self.data[self.data.columns.drop(list(self.data.filter(regex='index_')))]

        # remove ROI_Label
        if "ROI_Label" in self.data.columns:
            self.data.drop(["ROI_Label"], inplace=True, axis=1)

        # remove duplicated ID
        if self.data.index.name == "ID":
            id_cols = self.data.copy().loc[:,self.data.columns[self.data.columns.str.contains(pat = 'ID.')]].columns
            if len(id_cols) > 0:
                print(f"Found {len(id_cols)} features with ID included. Dropping ... ")
                self.error.warning(f"Found {len(id_cols)} features with ID included. Dropping ... ")

                # drop index features for sanity check
                self.data = self.data[self.data.columns.drop(list(self.data.filter(regex='ID.')))]
        else:
            # make ID index
            if "ID" in self.data.columns:
                if len(self.data[self.data["ID"].duplicated()]) == 0:
                    self.data = self.data.set_index("ID")
            else:
                id_cols = self.data.copy().loc[:,self.data.columns[self.data.columns.str.contains(pat = 'ID.')]].columns
                if len(id_cols) > 0:
                    self.data.rename(columns={id_cols[0]: "ID"})
                    self.data = self.data.set_index("ID")

                id_cols = self.data.copy().loc[:,self.data.columns[self.data.columns.str.contains(pat = 'ID.')]].columns
                if len(id_cols) > 0:
                    print(f"Found {len(id_cols)} features with ID included. Dropping ... ")
                    self.error.warning(f"Found {len(id_cols)} features with ID included. Dropping ... ")
                    self.data = self.data[self.data.columns.drop(list(self.data.filter(regex='ID.')))]



        # remove config
        if "config" in self.data.columns:
            self.data.drop(["config"], inplace=True, axis=1)

    def run(self):
        """
        -- Main Function to run the Feature Filter --
        """
    
        feature_alteration = pd.DataFrame({"Feature Extraction": [np.nan],
                                           "Feature Configuration": [np.nan],
                                           "Feature Instability": [np.nan],
                                           "Constant Feature Filtering": [np.nan],
                                           "Numeric Feature Filtering": [np.nan],
                                           "Dropping Failed Extractions": [np.nan],
                                           "Dropping Duplicated Features": [np.nan],
                                           "Filter by Variance": [np.nan],
                                           "Filter by Correlation": [np.nan],
                                           "Adding Configuration": [np.nan],
                                           "Remove NaN Features": [np.nan]},
                                          index=[self.extractor])
        
        if not os.path.isfile(os.path.dirname(self.out_path) + "/filtered_features.csv"):
            self.read_data()

            # if index name is ID but wrong index has been given
            if "ID" not in self.data.columns:
                if self.data.index.name == "ID":
                    self.data["ID"] = self.data.index
                    if isinstance(self.data.index.values[0], (int, np.integer)):
                        self.error.warning("Index confguration is not configured by RPTK! Index should be string format!")
                        # if self.data.index.to_list() != self.data["ID"].to_list():
                    
            
            if "ID" in self.data.columns:
                self.data.index = self.data["ID"]

            feature_alteration.loc[feature_alteration.index == self.extractor, "Feature Extraction"] = self.data.shape[1]

            if self.path_to_img_seg_csv != "":
                self.logger.info("### Feature filtering Configuration:" +
                                 "\n\t\t\t\t\t\t\t\t\t\tInput CSV: " + str(self.path_to_img_seg_csv) +
                                 "\n\t\t\t\t\t\t\t\t\t\tInput Size: " + str(self.data.shape) +
                                 "\n\t\t\t\t\t\t\t\t\t\tOutput directory: " + self.out_path +
                                 "\n\t\t\t\t\t\t\t\t\t\tVariance threshold: " + str(self.variance_threshold) +
                                 "\n\t\t\t\t\t\t\t\t\t\tCorrelation threshold: " + str(self.correlation_threshold) +
                                 "\n\t\t\t\t\t\t\t\t\t\tRunID: " + self.RunID +
                                 "\n\t\t\t\t\t\t\t\t\t\tNumber of CPU: " + str(self.n_cpu) +
                                 "\n\t\t\t\t\t\t\t\t\t\tExtractor: " + self.extractor
                                 )
            else:
                self.logger.info("### Feature filtering Configuration:" +
                                 "\n\t\t\t\t\t\t\t\t\t\tInput Size: " + str(self.data.shape) +
                                 "\n\t\t\t\t\t\t\t\t\t\tOutput directory: " + self.out_path +
                                 "\n\t\t\t\t\t\t\t\t\t\tVariance threshold: " + str(self.variance_threshold) +
                                 "\n\t\t\t\t\t\t\t\t\t\tCorrelation threshold: " + str(self.correlation_threshold) +
                                 "\n\t\t\t\t\t\t\t\t\t\tRunID: " + self.RunID +
                                 "\n\t\t\t\t\t\t\t\t\t\tNumber of CPU: " + str(self.n_cpu) +
                                 "\n\t\t\t\t\t\t\t\t\t\tExtractor: " + self.extractor
                                 )


            self.logger.info("### Starting Feature filtering ###")
            print("### Starting Feature filtering ###")

            #if self.extractor == "MIRP":
            #    self.data.set_index("img_data_roi")
            # ADD here configuration

            self.logger.info("Features: " + str(self.data.shape[1]))
            self.logger.info("Samples: " + str(self.data.shape[0]))

            print("Features: " + str(self.data.shape[1]))
            print("Samples: " + str(self.data.shape[0]))

            # check if ID is correctly assigned to Image and Mask

            # Iterate through the rows and fix the IDs
            for index, row in self.data.copy().iterrows():
                #extract ID from filename
                image_id = os.path.basename(row['Image']).split('_')[0]
                mask_id = os.path.basename(row['Mask']).split('_')[0]
                
                id_mismatch = False

                # Ensure the IDs match; if not, take the first one as the correct ID
                if image_id != mask_id:
                    print(f"Mismatch found in row {index}: Image ID = {image_id}, Mask ID = {mask_id}")

                ID = []
                if isinstance(row["ID"], int):
                    ID = [row["ID"]]

                if isinstance(row["ID"], list):
                    ID = row["ID"]

                for id_ in ID:
                    
                    if str(image_id) != str(id_):

                        # if it turns out that the ID is a number adapt the ID to the real ID
                        if isinstance(id_, int) or isinstance(id_, float):
                            print("ID", id_ , "is a number not a sting! Trying to convert ...")
                            # this is not really a mismatch but just a problem of the number in the ID
                            if str(mask_id).endswith(str(id_)):
                                print("Correcting ID as a number for the sample", id_, "to Mask ID", str(mask_id))
                                # id_ = str(mask_id)
                                if str(image_id) == str(mask_id):
                                    self.data.at[index, 'ID'] = str(mask_id)
                                    continue
                                else:
                                    print("Found a mismatch between Mask and Image!", "Image ID:",str(image_id),"Mask ID:",str(mask_id))
                                    self.error.warning("Found a mismatch between Mask and Image! Image ID: " + str(image_id) + " Mask ID: " + str(mask_id))
                            
                            elif str(image_id).endswith(str(id_)):
                                print("Correcting ID as a number for the sample", id_, "to Image ID", str(image_id))
                                # id_ = str(mask_id)
                                if str(image_id) == str(mask_id):
                                    self.data.at[index, 'ID'] = str(image_id)
                                    continue
                                else:
                                    print("Found a mismatch between Mask and Image!", "Image ID:",str(image_id),"Mask ID:",str(mask_id))
                                    self.error.warning("Found a mismatch between Mask and Image! Image ID: " + str(image_id) + " Mask ID: " + str(mask_id))
                            else:
                                # wrong ID for this case
                                if str(image_id) == str(mask_id):
                                    self.data.at[index, 'ID'] = str(image_id)
                                    continue
                                else:
                                    print("Found a mismatch between Mask and Image!", "Image ID:",str(image_id),"Mask ID:",str(mask_id))
                                    self.error.warning("Found a mismatch between Mask and Image! Image ID: " + str(image_id) + " Mask ID: " + str(mask_id))



                        # if the Mask ID is the correct ID
                        elif str(mask_id) == str(id_):
                            print(f"Mismatch found in row {str(index)}: Image ID = {str(image_id)}, ID = {str(id_)}")
                            id_mismatch = True

                            # get right image matching to this sample
                            sample = self.data.copy().loc[(self.data["Image_Transformation"] == row['Image_Transformation']) and (self.data["ID"] == row['ID']), "Image"].to_list()

                            for img in sample:
                                real_image_id =  os.path.basename(img).split('_')[0]
                                if str(real_image_id) == str(id_):
                                    self.data.at[index, 'Image'] = img
                                    break

                            # Replace the ID column with the correct ID
                            self.data.at[index, 'ID'] = mask_id
                        
                        # if Image ID is the correct ID
                        else:
                            print(f"Mismatch found in row {str(index)}: Mask ID = {str(mask_id)}, ID = {str(id_)}")
                            id_mismatch = True

                            # get right msk matching to this sample
                            sample = self.data.copy().loc[(self.data["Mask_Transformation"] == row['Mask_Transformation']) and (self.data["ID"] == row['ID']), "Mask"].to_list()

                            for msk in sample:
                                real_msk_id =  os.path.basename(msk).split('_')[0]
                                if str(real_msk_id) == str(id_):
                                    self.data.at[index, 'Mask'] = msk
                                    break


                            # Replace the ID column with the correct ID
                            self.data.at[index, 'ID'] = image_id

                    if str(mask_id) != str(id_):
                        if id_mismatch:
                            print(f"Mismatch found in row {str(index)}: Mask ID = {str(image_id)}, ID = {str(id_)}")
                        else:
                            self.error.error(f"Mismatch found in row {str(index)}: Mask ID = {str(image_id)}, ID = {str(id_)} but Image ID seems to match! please check if the Image and Mask are from the same sample.")
                            raise ValueError(f"Mismatch found in row {str(index)}: Mask ID = {str(image_id)}, ID = {str(id_)} but Image ID seems to match! please check if the Image and Mask are from the same sample.")

            # self.data.to_csv(self.out_path + "DataConfigurator_start_" + self.RunID + "_data.csv")

            self.configurator = self.get_DataConfigurator()
            self.data, self.string_parameter = self.configurator.config()

            # remove duplicated columns in the dataframe
            self.string_parameter = self.string_parameter.loc[:,~self.string_parameter.columns.duplicated()].copy()
            # self.string_parameter.to_csv(str(os.path.dirname(self.out_path)) + "/string_parameters_0.csv")
            print("Features after config: " + str(self.data.shape[1]))
            print("Samples after config: " + str(self.data.shape[0]))

            self.string_parameter = self.process_dataframe(self.string_parameter)

            feature_alteration.loc[feature_alteration.index == self.extractor, "Feature Configuration"] = self.configurator.config_feature_size
            feature_alteration.loc[feature_alteration.index == self.extractor, "Feature Instability"] = self.configurator.stable_feature_size
            feature_alteration.loc[feature_alteration.index == self.extractor, "Constant Feature Filtering"] = self.configurator.non_constant_feature_size
            feature_alteration.loc[feature_alteration.index == self.extractor, "Numeric Feature Filtering"] = self.data.shape[1]

            # Synchronize feature space - no need as we are searching for instable features
            # self.df_mask_transform = self.df_mask_transform[self.data.columns]

            self.logger.info("### 8. Calculate Mean for Samples with multiple ROI ###")
            print("### 8. Calculate Mean for Samples with multiple ROI ###")

            #if self.data.index.name == "config":
            #    if "config" in self.data.columns:
            #        self.data["config"] = self.data.index

            #if self.string_parameter.index.name == "config":
            #    if "config" in self.string_parameter.columns:
            #        self.string_parameter["config"] = self.string_parameter.index

            if "ID" in self.string_parameter.columns:
                wrong_ID_format = False
                for i in self.string_parameter["ID"]:
                    if "_" in str(i):
                        wrong_ID_format = True
                        break
                        
                if wrong_ID_format:
                    self.string_parameter['real_ID'] = self.string_parameter["ID"].str.split("_", expand=True)[0]

                    if "ID.1" in self.string_parameter.columns:
                        self.string_parameter.drop(['ID', 'ID.1'], axis = 1, inplace = True) 
                    else:
                        self.string_parameter.drop(['ID'], axis = 1, inplace = True)
                        
                    self.string_parameter = self.string_parameter.rename(columns={"real_ID": "ID"})

                else:
                    if "ID.1" in self.string_parameter.columns:
                        self.string_parameter.drop(['ID.1'], axis = 1, inplace = True) 

            # convert the cropped configuration
            if "config" in self.data.columns:
                if "_cropped_" in self.data["config"].to_list()[0]:
                    confs = self.data["config"].to_list()
                    for conf in confs:
                        x = re.search("([A-Za-z0-9-._+/]*_image_)[A-Za-z0-9-._+/]*_cropped_resample_([A-Za-z0-9-._]*)", conf)
                        if not x is None: 
                            real_config = x.group(1) + x.group(2)
                            # print("Change Config", conf, "to", real_config)
                            self.data.loc[self.data["config"] == conf, "config"] = real_config
                            self.string_parameter.loc[self.string_parameter["config"] == conf, "config"] = real_config
                            # self.data.rename(index={idx:real_config}, inplace=True)
                        else:
                            print("Could not get the ID out of configuration. Please check that special characters like \"_\" are not in your ID.") 

            # check configuration of config (Mask should not be included!)

            # Got two configs 1 for all samples to be unique and 1 for detecting multiple ROIs for each sample
            # Need to set the config for multiple ROIs as index to detect them
            if self.data.index.name == "config":
                if "config" in self.data.columns:
                    self.data.reset_index(drop=True, inplace=True)
                else:
                    self.data.reset_index(drop=False, inplace=True)
            
            # self.string_parameter.to_csv(str(os.path.dirname(self.out_path)) + "/string_parameters_1.csv")
            if self.string_parameter.index.name == "config":
                self.string_parameter["unique_config"] = self.string_parameter.index.copy()

            if self.data["config"].duplicated().sum() > 0:
                self.logger.info("Calculate mean for " + str(self.data["config"].duplicated().sum()) + " ROIs")
                print("Calculate mean for " + str(self.data["config"].duplicated().sum()) + " ROIs")

                print("Features before mean: " + str(self.data.shape[1]))
                print("Samples before mean: " + str(self.data.shape[0]))
                # self.data.to_csv(self.out_path + "/data_before_mean.csv")
                # self.string_parameter.to_csv(self.out_path + "/string_parameter_before_mean.csv")

                if "Prediction_Label" not in self.data.columns:
                    self.data = self.data.set_index("config")
                    self.data["Prediction_Label"] = self.string_parameter["Prediction_Label"]
                    self.data.reset_index(drop=False, inplace=True)

                # self.data.to_csv(self.out_path + "/before_mean.csv")
                
                if "Timepoint" in self.data.columns:
                    # Calculate mean for all samples with the same config       
                    self.data = self.data.groupby(by=["config","Prediction_Label","Timepoint"], dropna=False).mean()
                else:
                    # Calculate mean for all samples with the same config
                    self.data = self.data.groupby(by=["config","Prediction_Label"], dropna=False).mean()

                index_level_to_get = 0
                self.data["config"] = self.data.index.get_level_values(index_level_to_get)
                self.data = self.data.set_index("config")
                self.data_mean = self.data.copy()
                # self.data = self.calc_mean_for_multi_ROI(df=self.data.copy(), ID="config")
                #self.data.index = self.data["config"]
                # --> config gets index!
                # self.data = self.data.groupby(by=["config"], dropna=False).mean()
            
            else:
                self.data = self.data.set_index("config")
                self.logger.info("No multiple ROIs found. Processing each sample without calculate the mean.")
                print("No multiple ROIs found. Processing each sample without calculate the mean.")

            # self.string_parameter.to_csv(str(os.path.dirname(self.out_path)) + "/string_parameters_2.csv")
            # correct features and remove string features if they are not outsourced already
            for feature in self.data.copy().columns:
                try:
                    self.data[feature] = self.data[feature].astype(float)
                except:
                    self.logger.info(feature + " not convertable to number. Saving it for later ...")
                    print(feature + " not convertable to number. Saving it for later ...")

                    if feature not in self.string_parameter.columns:
                        self.string_parameter[feature] = self.data[feature]
                        
                    self.data.drop(columns=[feature], inplace=True)

            self.logger.info("Corrected Sample Size: " +
            "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + "Samples: " + str(self.data.shape[0]) +
            "\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + "Features: " + str(self.data.shape[1]) + "\n")

            self.logger.info("### 9. Searching for NaN Values ###")
            print("### 9. Searching for NaN Values ###")
            self.data = self.searching_for_nan(df=self.data,
                                               out_file_name=self.extractor + "_" + self.RunID + "_nan_samples.csv")

            feature_alteration.loc[feature_alteration.index == self.extractor, "Dropping Failed Extractions"] = \
            self.data.shape[1]

            print("Corrected Sample Size:")
            print(str(self.data.shape[0]) + " Samples")
            print(str(self.data.shape[1]) + " Features\n")

            self.logger.info("### 10. Duplicated Feature Searching ###")
            print("### 10. Duplicated Feature Searching ###")
            self.data = self.remove_duplicated_feature(df=self.data.copy())

            feature_alteration.loc[feature_alteration.index == self.extractor, "Dropping Duplicated Features"] = \
                self.data.shape[1]

            print("Corrected Sample Size:")
            print(str(self.data.shape[0]) + " Samples")
            print(str(self.data.shape[1]) + " Features\n")

            self.logger.info("### 11. Variance Feature Searching ###")
            print("### 11. Variance Feature Searching ###")
            self.data = self.variance_filter(df=self.data.copy())

            feature_alteration.loc[feature_alteration.index == self.extractor, "Filter by Variance"] = \
                self.data.shape[
                    1]

            self.logger.info("Corrected Sample Size:")
            self.logger.info(str(self.data.shape[0]) + " Samples")
            self.logger.info(str(self.data.shape[1]) + " Features\n")

            print("Corrected Sample Size:")
            print(str(self.data.shape[0]) + " Samples")
            print(str(self.data.shape[1]) + " Features\n")

            self.logger.info("### 12. Correlation Feature Searching ###")
            print("### 12. Correlation Feature Searching ###")
            self.data, corr_simple_count, surrogates = self.correlation_surrogates_detection(df=self.data.copy())

            feature_alteration.loc[feature_alteration.index == self.extractor, "Filter by Correlation"] = \
                self.data.shape[1]

            self.logger.info("Corrected Sample Size:")
            self.logger.info(str(self.data.shape[0]) + " Samples")
            self.logger.info(str(self.data.shape[1]) + " Features\n")

            print("Corrected Sample Size:")
            print(str(self.data.shape[0]) + " Samples")
            print(str(self.data.shape[1]) + " Features\n")

            self.logger.info("### 13. Add configuration Data ###")
            print("### 13. Add configuration Data ###")
            
            # sync the index
            if self.string_parameter.index.name != "config":
                if "config" in self.string_parameter.columns:
                    self.string_parameter = self.string_parameter.drop_duplicates(subset='config', keep="first")
                    self.string_parameter = self.string_parameter.set_index('config')

            if len(self.string_parameter) > 1:
                self.string_parameter = self.update_df(self.string_parameter.copy())
            else:
                self.string_parameter =  pd.DataFrame()

            self.string_parameter = self.string_parameter.loc[:,~self.string_parameter.columns.duplicated()].copy()

            # check ID configuration
            if "ID" in self.string_parameter.columns:
                if "_cropped_" in self.string_parameter["ID"][0]:
                    confs = self.string_parameter["ID"].to_list()
                    for conf in confs:
                        print(conf)
                        if not pd.isnull(conf):
                            x = re.search("([A-Za-z0-9-._+/]*)_image_", conf)
                            if not x is None: 
                                real_config = str(conf).split("_")[0]
                                # real_config = x.group(1)
                                # print("Change Config", conf, "to", real_config)
                                self.string_parameter.loc[self.string_parameter["ID"] == conf, "ID"] = real_config
                                # self.data.rename(index={idx:real_config}, inplace=True)
                            else:
                                self.error.warning("Could not get the ID out of configuration. Please check file string_parameters.csv for ID config.") 
                                # self.string_parameter.to_csv(os.path.dirname(self.path_to_data) + "/string_parameters.csv")


            # self.string_parameter.to_csv(os.path.dirname(self.path_to_data) + "/string_parameter.csv")
            # self.data.to_csv(os.path.dirname(self.path_to_data) + "/preselected_relevant_parameter.csv")

            # Update the index of the string data
            # if "config" in self.string_df.columns:
            #    self.string_df = self.string_df.drop_duplicates(subset='config', keep="first")
            #    self.string_df = self.string_df.set_index('config')
            #    # self.string_df.index = self.string_df["config"]
            #    self.string_df = self.update_df(self.string_df)
            # self.string_df.to_csv(os.path.dirname(self.path_to_data) + "/string_containing_features.csv")

            # for col in self.string_parameter.columns:
            #    if col in self.string_df.columns:
            #        self.string_df = self.string_df.drop([col], axis=1)
            if self.data.index.name != "config":
                if "config" in self.data.columns:
                    self.data.set_index('config', inplace=True)
                else:
                    self.error.warning("No configuration data found in the data frame. Please check the data.")
                    self.error.warning("Check File: " + os.path.dirname(self.out_path) + "/tmp/config_failure.csv")
                    self.data.to_csv(os.path.dirname(self.out_path) + "/tmp/config_failure.csv")

            # self.string_parameter.set_index('config', inplace=True)

            # merge string data with configured data
            # if self.string_df.shape[0] > 0:
            # self.data = pd.concat([self.data, self.string_df], axis=1, join='outer'ƒ)

            # drop duplicates which are completely duplicated in all columns
            self.data = self.data.drop_duplicates()
            # self.string_parameter = self.string_parameter.drop_duplicates()
            # self.string_parameter.to_csv(str(os.path.dirname(self.out_path)) + "/string_parameters_final.csv")
            # self.data.to_csv(str(os.path.dirname(self.out_path)) + "/data_final.csv")

            self.data = pd.concat([self.data, self.string_parameter], axis=1)
            self.data = self.data.loc[:,~self.data.columns.duplicated()]

            feature_alteration.loc[feature_alteration.index == self.extractor, "Adding Configuration"] = \
                self.data.shape[1]

            # drop features with only nan:
            self.data.dropna(axis=1, how='all', inplace=True)
            
            feature_alteration.loc[feature_alteration.index == self.extractor, "Remove NaN Features"] = \
                self.data.shape[1]

            self.logger.info("Filtered Feature Space:")
            self.logger.info(str(self.data.shape[0]) + " Samples")
            self.logger.info(str(self.data.shape[1]) + " Features\n")

            print("Filtered Feature Space:")
            print(str(self.data.shape[0]) + " Samples")
            print(str(self.data.shape[1]) + " Features\n")

            if "ID" in self.data.columns:
                if "_" in self.data["ID"][0]:
                    print("Found illigal character _ in ID. Try to adjust ID ...")
                    confs = self.data["ID"].to_list()
                    for conf in confs:
                        real_config = str(conf).split("_")[0]
                        self.data.loc[self.data["ID"] == conf, "ID"] = real_config
            
            # if column replacing failed - repair
            if None in self.data.columns:
                self.data = self.data.drop([None], axis=1)

            if sum(self.data.columns.str.startswith('index')) > 0:
                self.logger.info("Excluding non reproducible features: " + str(self.data.loc[:,self.data.columns.str.startswith('index')].columns.to_list()))
                print("Excluding non reproducible features: " + str(self.data.loc[:,self.data.columns.str.startswith('index')].columns.to_list()))
                self.data = self.data.loc[:,~self.data.columns.str.startswith('index')]

            self.remove_uninformative_features()

            self.data.to_csv(os.path.dirname(self.out_path) + "/filtered_features.csv")

            if not os.path.exists(os.path.dirname(self.out_path) + "/feature_alteration_profile.csv"):
                feature_alteration.to_csv(os.path.dirname(self.out_path) + "/feature_alteration_profile.csv")
                
                # plot feature space alteration during filtering
                try:
                    self.filter_profile_plotter(df=feature_alteration)
                except Exception as ex:
                    print(f"Could not plot feature space alteration: {ex}")
                    self.error.warning(f"Could not plot feature space alteration: {ex}")

            # if there are more then one ROI
            if self.multiple_ROIs:
                intra_df, peri_df = self.split_by_ROI()
                return intra_df, peri_df
        else:
            self.data = pd.read_csv(os.path.dirname(self.out_path) + "/filtered_features.csv")

            self.remove_uninformative_features()

            if "ID" in self.data.columns:
                if "_" in self.data["ID"][0]:
                    print("Found illigal character _ in ID. Try to adjust ID ...")
                    confs = self.data["ID"].to_list()
                    for conf in confs:
                        real_config = str(conf).split("_")[0]
                        self.data.loc[self.data["ID"] == conf, "ID"] = real_config
        

        # check for IBSI coverage
        #IBSIFeatureFormater(extractor=self.extractor,
        #                    features=self.data,
        #                    RunID=self.RunID,
        #                    logger=self.logger,
        #                    error=self.error,
        #                    output_path=os.path.dirname(self.out_path) + "/IBSI_profile/").format_features()

        return self.data

    

