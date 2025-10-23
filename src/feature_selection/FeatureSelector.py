import glob
import logging
import os
import pandas as pd
import numpy as np
from tqdm import *
import torch
from pathlib import Path
import sys

import optuna
from optuna.trial import Trial

from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.model_training.PerformancePlotter import PerformancePlotter
from rptk.src.feature_extraction.generate_radiomics_feature_map import RadiomicsMapGenerator
from rptk.src.model_training.ModelTrainer import ModelTrainer

from sklearn.pipeline import FeatureUnion  # get unique selected features
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import SelectFdr  # Benjamini Hochberg procedure optimizes ANOVAR with reduction of FPR

from sklearn.feature_selection import SequentialFeatureSelector

from mlxtend.math import num_combinations
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from pandas.api.types import is_numeric_dtype

from rptk.src.feature_filtering.ibsi_feature_formater import IBSIFeatureFormater

# import warnings filter
from warnings import simplefilter

# from .HyperparameterGenerator import GridHyperParameterGenerator

class FeatureSelectionPipeline:
    """
    Select Features from the feature space.
    :param input_path (str): Path of the input file with the radiomics features
    :param data (pd.DataFrame): Dataframe with the radiomics features
    :param label_file (str): CSV file with the labels (Prediction_Label)
    :param model: model to use for feature selection
    :param logger (logging): Logger for logging
    :param error (logging): Error logging
    :param Prediction_Label (str): string as the name of the column with the label
    :param n_cpus (int): number of CPUs to use for feature selection
    :param RundID (str): ID for the Run
    :param out_folder (str): path to the output folder for data writing
    :param scoring (str): matrics to score the features
    :param test_size (float): Size of the test set
    :param n_features (int): If a specific number of best features should be extracted else 0 to extract features until select_tol
    :param select_tol (float): Parameter to define min feature importance difference for adding/removing features
    :param fit_params (dict): Parameter for model fitting
    :param self_optimize (bool): Set true for enabling self-optimization
    :param num_splits: number of splits for cross-validation (0=no cross-validation)
    """

    def __init__(self,
                 input_path: str = None,
                 data: pd.DataFrame = None,
                 label_file: str = None,
                 model="all",
                 logger: logging.Logger = None,
                 error=None,
                 Prediction_Label: str = None,
                 n_cpus: int = 1,
                 RunID: str = None,
                 out_folder: str = "",
                 scoring: str = "roc_auc",
                 test_size: float = 0.3,
                 n_features: int = None,
                 select_tol: float = 0.05,
                 fit_params: dict = None,
                 self_optimize: bool = True,
                 num_splits: int = 5,  # Number of splits for cross-validation
                 backward_sfs: bool = True,  # Use backward sequential feature selection
                 forward_sfs: bool = True,  # Use forward sequential feature selection
                 rand_state: int = 1234,
                 save_model: bool = False,
                 sfs_lib: str = "mlxtend",  # "mlxtend" or "sklearn"
                 verbose: int = 0,
                 # Either "sklearn" or "mlxtend" algorithm for sequential feature selection (more plots with mlxtend)
                 use_gpu: bool = True,  # Use GPUs for some models to train if possible
                 num_device: int = 1, # Number of GPUs to use
                 extractor: str = "PyRadiomics",  # default extractor
                 modality: str = "CT",  # default modality
                 critical_feature_size: int = 200,  # critical feature size to select features as it may take too long
                 min_feature: int = 5,  # min features to extract if critical feature size
                 max_feature: int = 20,
                 task:str = None,
                 trainer: ModelTrainer = None,
                 imbalance_method: str = "SMOTE"):  

        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        if fit_params is None:
            fit_params = {}

        self.input_path = input_path  # Path of the input file with the radiomics features
        self.data = data  # Dataframe with filtered radiomics features
        self.Prediction_Label = Prediction_Label
        self.label_file = label_file
        self.n_cpus = n_cpus
        self.RunID = RunID
        self.out_folder = out_folder
        self.model = model
        self.logger = logger
        self.error = error
        self.scoring = scoring
        self.num_splits = num_splits
        self.test_size = test_size
        self.n_features = n_features
        self.select_tol = select_tol
        self.params = fit_params
        self.self_optimize = self_optimize
        self.backward_sfs = backward_sfs
        self.forward_sfs = forward_sfs
        # self.use_optuna = use_optuna
        # self.iter = _iter
        # self.trial = Trial
        # self.extended_parameter_set = extended_parameter_set
        self.rand_state = rand_state
        self.save_model = save_model
        self.sfs_lib = sfs_lib
        self.use_gpu = use_gpu
        self.num_device = num_device
        self.verbose = verbose
        self.extractor = extractor
        self.modality = modality
        self.critical_feature_size = critical_feature_size
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.task = task
        self.trainer = trainer
        self.imbalance_method = imbalance_method

        # self.model_save_dir = os.path.dirname(self.out_folder) + "/models"
        self.model_save_dir = Path(self.out_folder + '/model_out/trained_models/')
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.best_model = None
        self.params = {}

        # if self.logger is None:
        self.logger = LogGenerator(
            log_file_name=self.out_folder + "/RPTK_feature_selection_" + self.RunID + ".log",
            logger_topic="RPTK Feature Selection"
        ).generate_log()

        # if self.error is None:
        self.error = LogGenerator(
            log_file_name=self.out_folder + "/RPTK_feature_selection_" + self.RunID + ".err",
            logger_topic="RPTK Feature Selection error"
        ).generate_log()

        if self.input_path is None:
            if self.data is None:
                self.error.error("No input_path nor data given to input feature space!")
            else:
                self.logger.info("Get data from provided data frame.")
        else:
            if self.data is None:
                self.logger.info("Reading data from provided input path: {}".format(self.input_path))
                self.data = pd.read_csv(self.input_path)
                if len(self.data.columns) < 3:
                    self.data = pd.read_csv(self.input_path, sep=";")

        self.logger.info("### Configuration of Selection ###")

        if self.sfs_lib == "sklearn" and (self.backward_sfs or self.forward_sfs):
            self.logger.info("Performing Sequential Feature selection with sklearn.")
        elif self.sfs_lib == "mlxtend" and (self.backward_sfs or self.forward_sfs):
            self.logger.info("Performing Sequential Feature selection with mlxtend.")

        self.logger.info("Scoring: " + self.scoring)
        self.logger.info("Use Forward Selection: " + str(self.forward_sfs))
        self.logger.info("Use Backward Selection: " + str(self.backward_sfs))

        # create the name of the file where the selected feature names should be stored
        self.sel_file_path = str(
            self.out_folder + '/Performed_Feature_selection_status_' + self.RunID + "_" + str(self.rand_state) + '.csv')

        self.out_file_path = str(
            self.out_folder + '/Feature_selection_' + self.RunID + "_" + str(self.rand_state) + '.csv')

        self.logger.info("Store feature selected in : " + str(self.out_file_path))
        print("Store feature selected in : " + str(self.out_file_path))

    def check_for_nan(self):
        """
        Find features with nan and drop them.
        """
        
        # check for index before processing
        if self.data.index.name == "ID":
            self.data["ID"] = self.data.index
        
        if "ID" in self.data.columns:
            if self.data["ID"].astype(str).str.isnumeric().any():
                if "Image" in self.data.columns:
                    self.data["ID"] = self.data.index.apply(lambda x: str(os.path.basename(experiment["Image"]).split('_')[0]))
                elif (self.data.index.name == "config"): 
                    self.data["ID"] = self.data.index.apply(lambda x: str(x.split('_')[0]))
                elif "config" in self.data.columns :
                    self.data["ID"] = self.data["config"].apply(lambda x: str(x.split('_')[0]))

                # if ID is still numeric --> add string
                if self.data["ID"].astype(str).str.isnumeric().any():
                    self.data["ID"] = 'ID-' + self.data["ID"].astype(str)
                    
            self.data = self.data.set_index("ID")
            #else:
            #    print("ID is not numeric! Please check input data!")
                # self.data.drop(columns=["ID"], inplace=True)
        else:
            print("ID is missing in data! please check input data!")
            self.error.error("ID is missing in data! please check input data!")
        
        # Identify columns containing only NaN values
        nan_columns = list(self.data.loc[:, self.data.isnull().all()].columns)  # list(self.data.columns[self.data.isnull().all()].values)

        # remove columns coming from matching error
        for col in nan_columns:
            if "id_subject" in col:
                nan_columns.remove(col)

        if len(nan_columns) > 0:
            self.logger.info("Drop columns containing only NaN values: " + str(nan_columns))

            # Drop columns containing only NaN values
            self.data.drop(columns=nan_columns, inplace=True)

        if self.data.isnull().values.any():
            nan_samples = self.data.index[self.data.isna().any(axis=1)].values
            self.error.warning("Detected NaN value for {} samples.".format(str(len(nan_samples))))
            self.error.warning(nan_samples)

            # check if all samples are going to get dropped
            if len(nan_samples) == self.data.shape[0]:
                # drop columns with any nan
                self.data = self.data.dropna(axis=1, how='any')
            else:
                self.data = self.data.dropna()

    def exhaustive_selection(self, model, X, y):
        """
        Perform Exhaustive feature selection
        :param model: model to perform exhaustive feature selection with
        :param X: (pd.DataFrame) Data to train on
        :param y: (pd.Series) Data label for training
        :return: Exhaustive feature selection object after fitting
        """
        if self.task is None:
            if self.trainer  is None:
                self.trainer = ModelTrainer(data=self.data,  # data where the model looks at
                                    Prediction_Label=self.Prediction_Label,  # Name of the label in data
                                    out_folder=self.out_folder,
                                    ensemble=True,
                                    model_save_dir=self.out_folder + "/models",
                                    plot_save_dir=self.out_folder + "/plots",
                                    model=self.model,
                                    model_params=None,
                                    use_cross_validation=True,
                                    rand_state=self.rand_state,
                                    logger=self.logger,
                                    error=self.error,
                                    RunID=self.RunID,
                                    cross_val_splits=self.num_splits,
                                    n_cpus=self.n_cpus,
                                    device=device,
                                    imbalance_method=self.imbalance_method)

            # Split data into test/train sets
            self.trainer.split_data()

            self.task = self.trainer.task

        efs = EFS(model,
                  min_features=1,
                  max_features=X.shape[1],
                  scoring=self.scoring,
                  print_progress=True,
                  n_jobs=self.n_cpus,
                  pre_dispatch=1,
                  cv=self.num_splits).fit(X, y)

        selected, best_selected = self.get_best_features(selection=efs,
                                                         X=X)

        return selected, best_selected

    def sklearn_feature_selection(self, model, direction, n_features, X=None, y=None):
        """
        Perform a Sequential feature selection with the sklearn framework
        :param model: a model to train inside a sequential feature selection framework
        :param direction: forward or backward for the direction of sequential feature selection
        :param n_features: number of features to select ("auto" select best features)
        :param X: (pd.DataFrame) Data to train on
        :param y: (pd.Series) Data label for training
        :return: sfs object
        """

        if len(self.params) > 0:
            if n_features == "auto":
                sfs = SequentialFeatureSelector(model,
                                                tol=self.select_tol,
                                                n_features_to_select=n_features,
                                                direction=direction,
                                                scoring=self.scoring,
                                                n_jobs=self.n_cpus).fit(X, y,
                                                                        # eval_set=[(self.X_val, self.y_val)],
                                                                        # **self.params
                                                                        )
            else:
                sfs = SequentialFeatureSelector(model,
                                                n_features_to_select=n_features,
                                                direction=direction,
                                                scoring=self.scoring,
                                                n_jobs=self.n_cpus).fit(X, y,
                                                                        # eval_set=[(self.X_val, self.y_val)],
                                                                        # **self.params
                                                                        )
        else:
            if n_features == "auto":
                sfs = SequentialFeatureSelector(model,
                                                tol=self.select_tol,
                                                n_features_to_select=n_features,
                                                direction=direction,
                                                scoring=self.scoring,
                                                n_jobs=self.n_cpus).fit(X, y)
            else:
                sfs = SequentialFeatureSelector(model,
                                                n_features_to_select=n_features,
                                                direction=direction,
                                                scoring=self.scoring,
                                                n_jobs=self.n_cpus).fit(X, y)
        return sfs

    def mlxtend_feature_selector(self, model, direction: str, n_features=None, X=None, y=None):
        """
        Perform a Sequential feature selection with the mlxtend framework
        :param model: model to train inside sequential feature selection framework
        :param direction: forward or backward for direction of sequential features selection
        :param n_features: How many features to extract (best = optimize number)
        :param X: (pd.DataFrame) Data to train on
        :param y: (pd.Series) Data label for training
        :return: sfs object
        """
        
        if type(model).__name__ == "XGBClassifier":
            pre_dispatch = self.n_cpus
            # os.environ['OMP_NUM_THREADS'] = "1"
        else:
            pre_dispatch = self.n_cpus
            # os.environ['OMP_NUM_THREADS'] = str(self.n_cpus)
        
        if n_features is None:
            n_features = "best"

        if direction == "forward":
            if len(self.params) > 0:
                sfs = SFS(model,
                          k_features=n_features,
                          forward=True,
                          scoring=self.scoring,
                          cv=self.num_splits,
                          verbose=self.verbose,
                          n_jobs=self.n_cpus,
                          pre_dispatch=pre_dispatch
                          ).fit(X, y,
                                # eval_set=[(self.X_val, self.y_val)],
                                **self.params)
            else:
                sfs = SFS(model,
                          k_features=n_features,
                          forward=True,
                          scoring=self.scoring,
                          cv=self.num_splits,
                          verbose=self.verbose,
                          n_jobs=self.n_cpus,
                          pre_dispatch=pre_dispatch
                          ).fit(X, y)
        else:
            if len(self.params) > 0:
                sfs = SFS(model,
                          k_features=n_features,
                          forward=False,
                          scoring=self.scoring,
                          cv=self.num_splits,
                          verbose=self.verbose,
                          n_jobs=self.n_cpus,
                          pre_dispatch=pre_dispatch
                          ).fit(X, y,
                                # eval_set=[(self.X_val, self.y_val)],
                                **self.params)
            else:
                sfs = SFS(model,
                          k_features=n_features,
                          forward=False,
                          scoring=self.scoring,
                          cv=self.num_splits,
                          verbose=self.verbose,
                          n_jobs=self.n_cpus,
                          pre_dispatch=pre_dispatch
                          ).fit(X, y)

        # if it takes too long abort with key
        sfs.finalize_fit()

        self.logger.info('Best ' + str(self.scoring) + ' score from ' + str(type(model).__name__) + ' for ' + str(direction) + ' selection: %.3f' % sfs.k_score_)
        # print('Best Features:', sfs.k_feature_names_)

        # self.logger.info('Best score: %.2f' % sfs.k_score_)
        self.logger.info('Selected Features: ' + str(sfs.k_feature_names_))
        
        results = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
        results.to_csv(self.out_folder + "/model_out/" + str(type(model).__name__) + "_SFS_results.csv")

        return sfs

    def get_best_features(self, selection=None, X=None):
        """
        Get feature names of the best two features and the entire selected feature space
        :param selection: Selection object
        :param X: Data where the sfs has been applied on
        :return _tmp_list, best_two_features: list of all selected features, list of best two selected features
        """
        _tmp_list = []
        best_two_features = []
        best = 0

        if (selection is None) or (X is None):
            self.error.error("Please provide SFS object and the input data from selecting the best features!")
            raise ValueError("Please provide SFS object and the input data from selecting the best features!")

        if self.sfs_lib == "mlxtend":
            X_selected = selection.k_feature_names_
        elif self.sfs_lib == "sklearn":
            X_selected = selection.get_feature_names_out()
        else:
            self.error.error("Method {} is not supported for feature selection!".format(self.sfs_lib))
            raise ValueError("Method {} is not supported for feature selection!".format(self.sfs_lib))

        if type(X_selected) is np.ndarray:
            for i in X_selected:
                _tmp_list.append(i)
                if best < 2:
                    best_two_features.append(i)
                best += 1
        else:
            for col in X_selected:
                _tmp_list.append(col)
                if best < 2:
                    best_two_features.append(col)
                best += 1
        
        best_two_features = self.check_for_real_transformations(best_two_features, self.data.columns)
        
        return _tmp_list, best_two_features

    def sequential_feature_selection(self, direction: str = "forward", model=None, X=None, y=None):
        """
        Apply the Sequential Feature Selection (SFS) on the given model and feature names
        :param direction: forward or backward
        :param model: model to apply the SFS
        :param X: data to perform feature selection on
        :param y: Label to train feature selection
        :return: selected features, best two features
        """

        if self.task is None:
            if self.trainer  is None:
                self.trainer = ModelTrainer(data=self.data,  # data where the model looks at
                                    Prediction_Label=self.Prediction_Label,  # Name of the label in data
                                    out_folder=self.out_folder,
                                    ensemble=True,
                                    model_save_dir=self.out_folder + "/models",
                                    plot_save_dir=self.out_folder + "/plots",
                                    model=self.model,
                                    model_params=None,
                                    use_cross_validation=True,
                                    rand_state=self.rand_state,
                                    logger=self.logger,
                                    error=self.error,
                                    RunID=self.RunID,
                                    cross_val_splits=self.num_splits,
                                    n_cpus=self.n_cpus,
                                    device=device,
                                    imbalance_method=self.imbalance_method)

            # Split data into test/train sets
            self.trainer.split_data()

            self.task = self.trainer.task

        if self.task == "binary_classification":
            self.scoring = "roc_auc"

        elif self.task == "multi_class_classification":
            self.scoring = "roc_auc_ovr"

        elif self.task == "regression":
            self.scoring = "neg_mean_squared_error"

        else:
            self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
            raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")

        if model is None:
            model = self.model

        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train

        sfs = None

        if self.n_features is None:
            # if there are more fetaures than critical feature size reduce selection
            if len(X.columns) > self.critical_feature_size:
                # option only available for mlxtend
                if self.sfs_lib == "mlxtend":
                    self.n_features = (self.min_feature, self.max_feature)

                    self.error.warning("Number of features are above critical feature count {}. Feature selection may take very long. Therefore, best features between {} and {} are selected.".format(str(self.critical_feature_size), str(self.min_feature), str(self.max_feature)))
                    print("Number of features are above critical feature count {}. Feature selection may take very long. Therefore, best features between {} and {} are selected.".format(str(self.critical_feature_size), str(self.min_feature), str(self.max_feature)))
                else:
                    self.error.warning("Number of features are above critical feature count {}. Feature selection may take very long. Change feature selector to mlxtend to reduce computation for feature selection.".format(str(self.critical_feature_size)))
                    print("Number of features are above critical feature count {}. Feature selection may take very long. Change feature selector to mlxtend to reduce computation for feature selection.".format(str(self.critical_feature_size)))

        if self.sfs_lib == "sklearn":
            # extract an optimal number of features
            if self.n_features is not None:
                sfs = self.sklearn_feature_selection(model=model,
                                                     direction=direction,
                                                     n_features=self.n_features,
                                                     X=X,
                                                     y=y)

            else:
                sfs = self.sklearn_feature_selection(model=model,
                                                     direction=direction,
                                                     n_features="auto",
                                                     X=X,
                                                     y=y)

        elif self.sfs_lib == "mlxtend":
            # extract an optimal number of features
            if self.n_features is not None:
                sfs = self.mlxtend_feature_selector(model=model,
                                                    direction=direction,
                                                    n_features=self.n_features,
                                                    X=X,
                                                    y=y)
            else:
                sfs = self.mlxtend_feature_selector(model=model,
                                                    direction=direction,
                                                    n_features="best",
                                                    X=X,
                                                    y=y)

            # plot performance of sfs
            PerformancePlotter(output_path=self.out_folder + "/plots",
                               RunID=self.RunID,
                               error=self.error,
                               logger=self.logger).plot_sfs_features(sfs=sfs,
                                                                     title=direction + "_" + str(
                                                                         type(model).__name__) + "_" + str(
                                                                         self.sfs_lib))

        if sfs is not None:
            # Filter for selected features
            selected, best_selected = self.get_best_features(selection=sfs,
                                                             X=X)
        else:
            self.error.error("Sequential Feature selection Failed!")
            print("Sequential Feature selection Failed!")
            selected = []
            best_selected = []

        return selected, best_selected

    def write_selected_features_2_csv(self, selected_features, best_selected_features, model, method):
        """
        Write summary of selected features to csv outfile
        :param selected_features: list of selected features
        :param best_selected_features: list of two best features
        :param model: model which has been used for selection
        :param method: either exhaustive feature selection or sequential feature selection
        """

        selected_features_file_path = self.out_folder + "/model_out/Selected_features_" + str(
            type(model).__name__) + "_" + method + ".csv"
        best_selected_features_file_path = self.out_folder + "/model_out/Best_Selected_features_" + str(
            type(model).__name__) + "_" + method + ".csv"

        # generate dir if not exist
        out_dir = os.path.dirname(selected_features_file_path)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isfile(selected_features_file_path):
            with open(selected_features_file_path, 'w+') as fp:
                for item in selected_features:
                    # write each features on a new line
                    fp.write("%s\n" % item)
        elif (os.path.getsize(selected_features_file_path) == 0):
            with open(selected_features_file_path, 'w+') as fp:
                for item in selected_features:
                    # write each features on a new line
                    fp.write("%s\n" % item)
        else:
            self.logger.info(
                "Feature selection with {} in {} has been performed already in {}.".format(str(type(model).__name__),
                                                                                           method,
                                                                                           selected_features_file_path))

        if not os.path.isfile(best_selected_features_file_path):
            with open(best_selected_features_file_path, 'w+') as fp:
                for item in best_selected_features:
                    # write each features on a new line
                    fp.write("%s\n" % item)
        elif (os.path.getsize(best_selected_features_file_path) == 0):
            with open(best_selected_features_file_path, 'w+') as fp:
                for item in best_selected_features:
                    # write each features on a new line
                    fp.write("%s\n" % item)
        else:
            self.logger.info(
                "Feature selection with {} in {} has been performed already in {}.".format(str(type(model).__name__),
                                                                                           method,
                                                                                           best_selected_features_file_path))

        if not os.path.isfile(self.sel_file_path):
            selected_features_dict = {"Selection_Model": [],
                                      "Selection_Method": [],
                                      # "Selected_Features": [],
                                      # "Best_selected_Features": []
                                      }
            df = pd.DataFrame.from_dict(selected_features_dict)
        else:
            df = pd.read_csv(self.sel_file_path, index_col=0)

        if (type(model).__name__ not in df["Selection_Model"]) and \
                (method not in df.loc[df["Selection_Model"] == type(model).__name__, "Selection_Method"]):
            entry = {"Selection_Model": [type(model).__name__],
                     "Selection_Method": [method],
                     # "Selected_Features": [selected_features],
                     # "Best_selected_Features": [best_selected_features]
                     }

            df = pd.concat([df, pd.DataFrame.from_dict(entry)], ignore_index=True)

            df.to_csv(self.sel_file_path)

    def check_performed_selection(self, model, selection_out: pd.DataFrame, selection_method: str):
        """
        Search for performed features selections in output file
        :param model: model used for selection
        :param selection_out: (pd.DataFrame) from the selected output file
        :param selection_method: performed selection method
        :return: True if it exists else False
        """
        
        if len(selection_out) > 0:
            if str(type(model).__name__) in selection_out["Selection_Model"].tolist():
                if selection_method in selection_out.loc[
                    selection_out["Selection_Model"] == str(type(model).__name__), "Selection_Method"].tolist():
                    skip = True
                    self.logger.info(
                        "Found performed selection method with {} executed in {}.".format(str(type(model).__name__),
                                                                                          selection_method))
                else:
                    skip = False
            else:
                skip = False
        else:
            skip = False
        return skip
    
    def check_for_real_transformations(self, selected_features, filtered_features):
        """
        Check if selected feautres are in the feature space with correct transformation
        :param selected_features: list of selcted features
        :param filtered_features; list of filtered features
        """

        real_selected_features = []
        for selected_feature in selected_features:
            if selected_feature not in filtered_features:

                # correct transformation
                if "Gradient" in selected_feature:
                    selected_feature=selected_feature.replace('Gradient', 'gradient')
                elif "LBP2D" in selected_feature:
                    selected_feature=selected_feature.replace('LBP2D', 'lbp-2D')
                elif "Wavelet" in selected_feature:
                    selected_feature=selected_feature.replace('Wavelet', 'wavelet-')
                elif "Square_" in selected_feature:
                    selected_feature=selected_feature.replace('Square_', 'square_')
                elif "Mean" in selected_feature:
                    selected_feature=selected_feature.replace('Mean', 'mean')
                elif "Laplacian_of_Gaussian" in selected_feature:
                    selected_feature=selected_feature.replace('Laplacian_of_Gaussian', 'laplacian_of_gaussian')
                elif "Separable_Wavelet" in selected_feature:
                    selected_feature=selected_feature.replace('Separable_Wavelet', 'separable_wavelet')
                elif "Nonseparable_Wavelet" in selected_feature:
                    selected_feature=selected_feature.replace('Nonseparable_Wavelet', 'wavelet')
                elif "Gaussian" in selected_feature:
                    selected_feature=selected_feature.replace('Gaussian', 'gauss')
                elif "Gabor" in selected_feature:
                    selected_feature=selected_feature.replace('Gabor', 'gabor')
                elif "Laws" in selected_feature:
                    selected_feature=selected_feature.replace('Laws', 'laws')
                elif "LBP3D" in selected_feature:
                    selected_feature=selected_feature.replace('LBP3D', 'lbp-3D')
                elif "Exponential" in selected_feature:
                    selected_feature=selected_feature.replace('Exponential', 'exponential')
                elif "Logarithm" in selected_feature:
                    selected_feature=selected_feature.replace('Logarithm', 'logarithm')
                elif "SquareRoot" in selected_feature:
                    selected_feature=selected_feature.replace('SquareRoot', 'squareroot')
                elif "LoG" in selected_feature:
                    selected_feature=selected_feature.replace('LoG', 'log-')
                elif "Wavelet" in selected_feature:
                    selected_feature=selected_feature.replace('Wavelet', 'wavelet-')

                if selected_feature in filtered_features:
                    real_selected_features.append(selected_feature)
                else:
                    self.error.warning("Could not find {} in filtered feature space!".format(selected_feature))
                    print("Could not find {} in filtered feature space!".format(selected_feature))
            else:
                real_selected_features.append(selected_feature)

        return real_selected_features

    def select_features(self):
        """
        Main method to perform feature selection
        :return: list of selected features and list of best two selected features
        """

        added_evaluation = 0
        self.selected_features = []
        
        if os.path.isfile(str(self.model_save_dir) + "/" + "Feature_Selection_Model_Performance.csv"):
            Evaluation = pd.read_csv(str(self.model_save_dir) + "/" + "Feature_Selection_Model_Performance.csv", index_col=0)
            Evaluation = Evaluation.drop_duplicates()
        else:
            Evaluation = pd.DataFrame({"Origin":[],
                                        "Model": [],
                                        "AUC": [],
                                        "AUC_lower": [],
                                        "AUC_upper": [],
                                        "F1": [],
                                        "F1_lower": [],
                                        "F1_upper": [],
                                        "Average_Precision": [],
                                        "sens_sample": [],
                                        "sens_lower": [],
                                        "sens_upper": [],
                                        "sp_sample": [],
                                        "sp_lower": [],
                                        "sp_upper": [],
                                        "Youden_Index": []
                                    })

        # 1. Load data and make sanity check
        self.check_for_nan()
        
        # drop index columns
        if self.data.columns.str.startswith('index').any():
            self.data = self.data.loc[:,~self.data.columns.str.startswith('index')]

        if self.data.index.name != "ID":
            self.data.index = self.data["ID"]

        if self.data.columns.str.startswith('ID').any():
            self.data = self.data.loc[:,~self.data.columns.str.startswith('ID')]

        if self.data.columns.str.startswith('ROI_Label').any():
            self.data = self.data.loc[:,~self.data.columns.str.startswith('ROI_Label')]

        #if self.data.columns.str.contains('config', case=False).any():
        #    self.data = self.data.loc[:,~self.data.columns.str.contains('config', case=False)]

        # check if gpu is available and allowed
        if torch.cuda.is_available() and self.use_gpu:
            device = "cuda"
            print("Processing on GPU & CPU...")
        else:
            print("No GPU, processing on CPU ...")
            device = "cpu"

        # 2. Split the Data and get models
        if self.trainer is None:
            self.trainer = ModelTrainer(data=self.data,  # data where the model looks at
                                Prediction_Label=self.Prediction_Label,  # Name of the label in data
                                out_folder=self.out_folder,
                                ensemble=True,
                                model_save_dir=self.out_folder + "/models",
                                plot_save_dir=self.out_folder + "/plots",
                                model=self.model,
                                model_params=None,
                                use_cross_validation=True,
                                rand_state=self.rand_state,
                                logger=self.logger,
                                error=self.error,
                                RunID=self.RunID,
                                cross_val_splits=self.num_splits,
                                n_cpus=self.n_cpus,
                                device=device,
                                imbalance_method=self.imbalance_method)

        # Split data into test/train sets
        self.trainer.split_data()

        self.task = self.trainer.task

        self.X_train = self.trainer.X_train
        self.y_train = self.trainer.y_train

        self.X_test = self.trainer.X_test
        self.y_test = self.trainer.y_test
        
        self.trainer.generate_model_instance()

        # Get list of all available models if "all" has been specified 
        if self.model == "all":
            
            self.model = self.trainer.feature_selection_models
            self.trainer.model = self.trainer.feature_selection_models
        else:
            select_model_list = []

            # multiple models selected
            if isinstance(self.model, list):
                for model in self.model:
                    # only model names in the list
                    if isinstance(model, str):
                        for select_model in self.trainer.feature_selection_models:
                            if model == type(select_model).__name__:
                                select_model_list.append(select_model)
                    else:
                        for select_model in self.trainer.feature_selection_models:
                            if type(model).__name__ == type(select_model).__name__:
                                select_model_list.append(select_model)
            elif isinstance(self.model, str):
                for select_model in self.trainer.feature_selection_models:
                    if self.model == type(select_model).__name__:
                        select_model_list.append(select_model)
            else:
                self.error.error("Model {} is not supported!".format(self.model))
                raise ValueError("Model {} is not supported!".format(self.model))

            self.model = select_model_list   

        if not os.path.isfile(self.out_file_path):

            selection_out = pd.DataFrame()
            if os.path.isfile(self.sel_file_path) and (os.path.getsize(self.sel_file_path) > 0):
                print("Reading CSV File from Output ...")
                selection_out = pd.read_csv(self.sel_file_path)

                # drop columns from wrong in formatting
                for col in selection_out:
                    if "Unnamed" in col:
                        selection_out = selection_out.drop(columns=[col])
                    if "ROI_Label" in col:
                        selection_out = selection_out.drop(columns=[col])

                selection_out.to_csv(self.sel_file_path)

                self.logger.info(
                    "Found {} performed Feature selection/s from previous runs!".format(str(selection_out.shape[0])))
                
                print("Found {} performed Feature selection/s from previous runs!".format(str(selection_out.shape[0])))
                
            pbar = tqdm(self.model, desc="Perform " +  self.extractor + " Feature Selection", total = len(self.model))
            for model in self.model:
                selected_features4model = []
                got_model = os.path.isfile(str(self.model_save_dir) + "/" + str(type(model).__name__) + "_trained_after_selection.sav")
                
                if type(model).__name__ == "XGBClassifier":
                    pbar.update(1)
                    continue
                
                if not got_model:
                    pbar.set_description("Perform " +  self.extractor + " Feature Selection with " + str(type(model).__name__))
                    # print("Start feature selection with {}".format(str(type(model).__name__)))
                    self.logger.info("Start feature selection with {}".format(str(type(model).__name__)))
                    if self.X_train.shape[1] < 500:
                        all_comb = np.sum([num_combinations(n=self.X_train.shape[1], k=i)
                                        for i in range(1, self.X_train.shape[1] + 1)])
                    else:
                        all_comb = 1000000

                    # Do exhaustive selection if it does not take too long (less than 100,000 perturbations)
                    if all_comb < 1000000:
                        self.logger.info("Exhaustive feature selection is feasible for this data with {} features.".format(
                            str(self.X_train.shape[1])))
                        print("Exhaustive feature selection is feasible for this data with {} features.".format(
                            str(self.X_train.shape[1])))
                        es_selected = []
                        if not self.check_performed_selection(model=model,
                                                            selection_out=selection_out,
                                                            selection_method="Exhaustive_feature_selection"):

                            es_selected, es_best_selected = self.exhaustive_selection(model=model,
                                                                                    X=self.X_train,
                                                                                    y=self.y_train)

                            self.write_selected_features_2_csv(selected_features=es_selected,
                                                            best_selected_features=es_best_selected,
                                                            model=model,
                                                            method="Exhaustive_feature_selection")

                        else:
                            selected_features_file_path = self.out_folder + "/model_out/Selected_features_" + str(
                                type(model).__name__) + "_" + "Exhaustive_feature_selection" + ".csv"
                            file1 = open(selected_features_file_path, 'r')

                            for line in file1:
                                line = line[:-1]  # line = line.replace("\\\\n", "").replace("\\n", "")
                                es_selected.append(line)

                        self.selected_features += es_selected
                        selected_features4model = es_selected

                    # 3. Feature selection
                    # 3.1 Forward feature selection
                    if self.forward_sfs:
                        f_selected = []
                        if not self.check_performed_selection(model=model,
                                                            selection_out=selection_out,
                                                            selection_method="Sequential_forward_selection"):

                            f_selected, f_best_selected = self.sequential_feature_selection(direction="forward", model=model)
                            self.write_selected_features_2_csv(selected_features=f_selected,
                                                            best_selected_features=f_best_selected,
                                                            model=model,
                                                            method="Sequential_forward_selection")
                        else:
                            selected_features_file_path = self.out_folder + "/model_out/Selected_features_" + str(
                                type(model).__name__) + "_" + "Sequential_forward_selection.csv"

                            file1 = open(selected_features_file_path, 'r')

                            for line in file1:
                                line = line[:-1]  # line = line.replace("\\\\n", "").replace("\\n", "")
                                f_selected.append(line)

                        # self.logger.info("Selected Features: " + str(f_selected))
                        self.selected_features += f_selected
                        selected_features4model = f_selected

                    # 3.2 Backward feature selection
                    if self.backward_sfs:
                        b_selected = []
                        if not self.check_performed_selection(model=model,
                                                            selection_out=selection_out,
                                                            selection_method="Sequential_backward_selection"):

                            b_selected, b_best_selected = self.sequential_feature_selection(direction="backward", model=model)
                            self.write_selected_features_2_csv(selected_features=b_selected,
                                                            best_selected_features=b_best_selected,
                                                            model=model,
                                                            method="Sequential_backward_selection")
                        else:
                            selected_features_file_path = self.out_folder + "/model_out/Selected_features_" + str(
                                type(model).__name__) + "_" + "Sequential_backward_selection.csv"

                            file1 = open(selected_features_file_path, 'r')

                            for line in file1:
                                line = line[:-1]  # .replace("\\\\n", "").replace("\\n", "")
                                b_selected.append(line)

                        # self.logger.info("Selected Features: " + str(b_selected))
                        self.selected_features += b_selected
                        selected_features4model = b_selected
                    


                    self.logger.info("Finished Sequential Feature Selection with " + str(type(model).__name__))
                    print("Finished Sequential Feature Selection with", type(model).__name__)

                    print("Performance before feature selection with {} Features.".format(str(len(self.X_train.columns))))

                    # TODO: temporary not resolved problems with multi class classification
                    if self.task == "binary_classification":
                    
                        # performance before feature selection
                        before_model = model
                        before_model = before_model.fit(self.X_train, self.y_train)
                        before_evaluation = self.trainer.extended_eval(y_true=self.y_test, X=self.X_test, model=before_model)
                        before_evaluation["Origin"] = "Before_Feature_Selection"
                        self.logger.info("AUROC Performance of model before feature selection: " +
                                        str(before_evaluation["AUC"].values[0]))
                        
                        Evaluation = pd.concat([Evaluation, before_evaluation], ignore_index=True)
                        added_evaluation +=1
                        
                        # Save Model:
                        if not os.path.isfile(str(self.model_save_dir) + "/" + str(type(model).__name__) + "_trained_after_selection.sav"):
                            self.trainer.save_model(model=before_model,
                                            file_path=str(self.model_save_dir) + "/" + str(type(model).__name__) + "_trained_before_selection.sav")

                        print("Performance after feature selection of", len(selected_features4model), "Features")

                        # performance after feature selection
                        after_model = model
                        after_model = after_model.fit(self.X_train[selected_features4model], self.y_train)
                        after_evaluation = self.trainer.extended_eval(y_true=self.y_test, X=self.X_test[selected_features4model],
                                                                model=after_model)

                        after_evaluation["Origin"] = "After_Feature_Selection"
                        self.logger.info(
                            "AUROC Performance of model after feature selection: " + str(after_evaluation["AUC"].values[0]))

                        # Save Model:
                        if not os.path.isfile(str(self.model_save_dir) + "/" + str(type(model).__name__) + "_trained_after_selection.sav"):
                            self.trainer.save_model(model=after_model,
                                            file_path=str(self.model_save_dir) + "/" + str(type(model).__name__) + "_trained_after_selection.sav")
                        added_evaluation+=1
                        Evaluation = pd.concat([Evaluation, after_evaluation], ignore_index=True)

                else:
                    print("Found trained", type(model).__name__ ,"model!")
                    selected_features_file_path = self.out_folder + "/model_out/Selected_features_" + str(
                        type(model).__name__) + "*.csv"

                    files = glob.glob(selected_features_file_path)
                    f_selected = []

                    for file in files:
                        file1 = open(file, 'r')

                        for line in file1:
                            line = line[:-1]  # line = line.replace("\\\\n", "").replace("\\n", "")
                            f_selected.append(line)

                        # self.logger.info("Selected Features: " + str(f_selected))
                        self.selected_features += f_selected
                    
                    # drop ROI Label
                    if "ROI_Label" in self.selected_features:
                        self.selected_features.remove("ROI_Label")
                
                pbar.update(1)

            self.logger.info("Selected {} different features!".format(str(len(set(self.selected_features)))))
            print("Selected {} different features!".format(str(len(set(self.selected_features)))))

            if os.path.isfile(str(self.model_save_dir) + "/" + "Feature_Selection_Model_Performance.csv"):
                if added_evaluation > 0:
                    Evaluation.to_csv(str(self.model_save_dir) + "/" + "Feature_Selection_Model_Performance.csv", mode='a', index=False, header=False)  # add to csv file for each model
            else:
                Evaluation.to_csv(str(self.model_save_dir) + "/" + "Feature_Selection_Model_Performance.csv", index=False)

            # Perform ANOVA feature selection for categorical features
            # f_statistic, p_values = f_classif(self.X_train, self.y_train)

            if ("ID" in self.data.columns) or (self.data.index.name == "ID"):
                if self.data.index.name == "ID":
                    self.data["ID"] = self.data.index

                if "ID" not in self.selected_features:
                    self.selected_features.append("ID")
            else:
                self.error.warning(
                    "No ID column found in input data. Please add manually to output file {}".format(self.sel_file_path))

            # add needed configuration to data
            self.selected_features.append(self.Prediction_Label)
            #selected_features.append("Image") # not needed
            #selected_features.append("Mask") # not needed

            if "config" in self.data.columns:
                self.selected_features.append("config")
            elif self.data.index.name != "config":
                self.error.warning("Configuration of the feature format is wrong! Please check the format of the features before selection.")

            # check for correct feature name configuration
            self.selected_features = self.check_for_real_transformations(self.selected_features, self.data.columns)
            
            selected_feature_space = pd.DataFrame()
            selected_feature_space = self.data.copy()[list(set(self.selected_features))]

            remove = []
            # check for columns having equal values but different names
            for col in selected_feature_space.columns:
                for col1 in selected_feature_space.columns:
                    if col != col1:
                        if selected_feature_space[col].equals(selected_feature_space[col1]):
                            if ((col, col1) not in remove) and ((col1, col) not in remove):
                                remove.append((col, col1))

            for i in remove:
                selected_feature_space = selected_feature_space.drop(columns=i[1])

            if not os.path.isfile(self.out_file_path):
                # drop duplicated columns with same values
                selected_feature_space = selected_feature_space.T.drop_duplicates().T

                if selected_feature_space.columns.str.startswith('ROI_Label').any():
                    selected_feature_space =selected_feature_space.loc[:,~selected_feature_space.columns.str.startswith('ROI_Label')]

                selected_feature_space.to_csv(self.out_file_path)

            if self.extractor == "PyRadiomics":
                self.logger.info("Extract Feature Maps from selected Features...")

                if self.modality == "CT":
                    self.extraction_yaml = "rptk/src/config/PyRadiomics/CT.yaml"
                elif self.modality == "MRI" or self.modality == "MR":
                    self.extraction_yaml = "rptk/src/config/PyRadiomics/MRI.yaml"

                # check for Img and Mask columns
                #if "Image" not in selected_feature_space.columns:
                if selected_feature_space.index.name != "config":
                    if "config" in selected_feature_space.columns:
                        selected_feature_space = selected_feature_space.set_index("config")
                if self.data.index.name != "config":
                    if "config" in self.data.columns:
                        self.data = self.data.set_index("config")
                    else:
                        self.error.warning("No configuration column found in data. Please add configuration column to data!")

                #selected_feature_space = pd.concat([selected_feature_space, pd.DataFrame([self.data["Mask"]])], axis=1)
                if "Image" in selected_feature_space.columns:
                    for i, row in tqdm(selected_feature_space.iterrows(), total=len(selected_feature_space), desc="Generating Feature Maps"):
                        RadiomicsMapGenerator(
                                            list_of_features=selected_feature_space.columns,
                                            path_to_img=row["Image"],
                                            path_to_msk=row["Mask"],
                                            out_path=self.out_folder,
                                            extraction_yaml=self.extraction_yaml,
                                            logger=self.logger,
                                            error=self.error,
                                            RunID=self.RunID
                                            ).generate_feature_maps()
                else:
                    self.logger.info("Can not generate Radiomics feature maps. No Image/Mask in selected features!")
                    print("Can not generate Radiomics feature maps. No Image/Mask in selected features!")
            
            if selected_feature_space.index.name == "ID":
                if "ID" in selected_feature_space.columns:
                    selected_feature_space.drop(["ID"], axis=1, inplace=True)
            else:
                if "ID" in selected_feature_space.columns:
                    if is_numeric_dtype(selected_feature_space["ID"]):
                        print("Warning: ID is not correctly formated!")
                        raise ValueError("ID is not correctly formated!")
                    else:
                        selected_feature_space.index=selected_feature_space["ID"]
        else:
            print("Found selected features from previous run: {}".format(str(self.out_file_path)))
            self.logger.info("Found selected features from previous run: {}".format(str(self.out_file_path)))
            selected_feature_space = pd.read_csv(self.out_file_path, index_col = 0)

            # drop duplicated columns with same values
            selected_feature_space = selected_feature_space.T.drop_duplicates().T
        
        # check for features which are not informative
        for col in selected_feature_space.copy().columns:
            if col == "ID":
                if selected_feature_space.index.name == "ID":
                    selected_feature_space = selected_feature_space.drop(columns=["ID"])
                    print(f"Warning: {col} found in feature space! This feature does not have any relevant information! Dropping ...")
                    self.error.warning(f"{col} found in feature space! This feature does not have any relevant information! Dropping ...")
                else:
                    selected_feature_space = selected_feature_space.set_index("ID")

            if "ID." in str(col):
                if selected_feature_space.index.name == "ID":
                    selected_feature_space = selected_feature_space.drop(columns=[col])
                    print(f"Warning: {col} found in feature space! This feature does not have any relevant information! Dropping ...")
                    self.error.warning(f"{col} found in feature space! This feature does not have any relevant information! Dropping ...")
                else:
                    selected_feature_space = selected_feature_space.set_index(col)
                    selected_feature_space = selected_feature_space.index.rename("ID")

            if col == "ROI_Label":
                print("Warning: ROI_Label found in feature space! This feature does not have any relevant information! Dropping ...")
                self.error.warning("ROI_Label found in feature space! This feature does not have any relevant information! Dropping ...")
                selected_feature_space = selected_feature_space.drop(columns=["ROI_Label"])

        # check for IBSI coverage
        IBSIFeatureFormater(extractor=self.extractor,
                            features=selected_feature_space,
                            RunID=self.RunID,
                            logger=self.logger,
                            error=self.error,
                            output_path=os.path.dirname(self.out_folder) + "/selected_features/IBSI_profile/").format_features()


        return selected_feature_space
