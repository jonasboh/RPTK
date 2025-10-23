import pandas as pd
import pickle5 as pickle
from tqdm import tqdm
from typing import Callable, Union
import matplotlib.pyplot as plt
from pathlib import Path
import re
from pathlib import Path
import json
import seaborn as sns
import os
import optuna
from optuna.trial import Trial
import time
import shap
import statistics
import numpy as np
import random
import torch
import glob
# from loaders import BarLoader, SpinningLoader, TextLoader
from collections import Counter

# RPTK import
from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.model_training.PerformancePlotter import PerformancePlotter
from rptk.src.model_training.HyperparameterConfigurator import HyperparameterConfigurator
from rptk.src.model_training.Optimizer import Optimizer

# Mlxtend
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.plotting import plot_decision_regions as plot_dcr
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.classifier import EnsembleVoteClassifier

# Neptune & Wnadb
import neptune
import neptune.integrations.optuna as optuna_utils
import wandb

# GPU support
# from cuml.ensemble import RandomForestRegressor as RandomForestRegressor_GPU
# from cuml.ensemble import RandomForestClassifier as RandomForestClassifier_GPU

# Sklearn
import sklearn
from sklearn.model_selection import train_test_split, validation_curve, ShuffleSplit, cross_val_score, GridSearchCV, \
    cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import roc_curve, make_scorer, accuracy_score, balanced_accuracy_score, mean_squared_error, \
    roc_auc_score, average_precision_score, f1_score, confusion_matrix, auc
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFdr, SelectFpr
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
# from tabnet import TabNet, TabNetClassifier
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# import warnings filter
from warnings import simplefilter

class TabnetWrapper:
    """
    Wrapper to convert Pandas DataFrame input to NumPy array for TabNetClassifier.
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        """Fit the model, ensuring TabNet gets NumPy array inputs."""
        self.model.fit(X.values, y)
        return self

    def predict(self, X):
        """Predict while ensuring X is a NumPy array."""
        predictions = self.model.predict(X.values)

        # Ensure predictions are in the right format
        if isinstance(predictions, np.ndarray) and predictions.ndim == 1:
            return predictions  # Already a 1D array, return as is

        # Convert single-column predictions to a 1D NumPy array
        return np.array(predictions).flatten()

    def predict_proba(self, X):
        """Predict probabilities while ensuring X is a NumPy array and output shape is correct."""
        probas = self.model.predict_proba(X.values)

        # Ensure output is a NumPy array
        probas = np.array(probas)

        # If it's a 1D array, reshape to (n_samples, n_classes)
        if probas.ndim == 1:
            probas = probas.reshape(-1, 1)

        return probas
    
    def get_params(self, deep=True):
        """Pass through get_params to avoid sklearn compatibility issues."""
        return self.model.get_params(deep)
    
    @property
    def classes_(self):
        """Pass through classes_ attribute for compatibility."""
        return self.model.classes_


class _ListFoldSplitter:
    """
    A simple sklearn-like CV splitter that yields pre-defined (train_idx, val_idx) pairs.
    """
    def __init__(self, fold_indices):
        # fold_indices: list of tuples (train_idx_array, val_idx_array) referencing rows of X passed to split()
        self.fold_indices = fold_indices
    def split(self, X=None, y=None, groups=None):
        for tr, va in self.fold_indices:
            yield tr, va
    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.fold_indices)


def _ensure_str_list(x):
    return [str(i) for i in x]


def get_thread_budget():
    try:
        n_cpus = os.cpu_count() or 1
    except Exception:
        n_cpus = 1
    return n_cpus


class ModelTrainer:
    """
    Class to train the models.
    """
    def _load_autoradiomics_splits(self, path):
        """
        Load AutoRadiomics splits.json and return:
          - test_ids: list[str] test IDs
          - folds: list[tuple[list[str], list[str]]] (train_ids, val_ids) per fold
          - n_folds: int
        Expected JSON keys:
          {
            "split_type": "...",
            "test": [...],
            "train": {
               "fold_0": [[train_ids...], [val_ids...]],
               "fold_1": [[...],[...]],
               ...
            }
          }
        """
        
        with open(path, "r") as f:
            cfg = json.load(f)

        if "test" not in cfg or "train" not in cfg:
            raise ValueError("splits.json must contain both 'test' and 'train' sections.")

        test_ids = [str(x) for x in cfg["test"]]
        folds = []
        train_section = cfg["train"]
        # deterministic ordering by fold number
        for k in sorted(train_section.keys(), key=lambda s: int(s.split('_')[-1])):
            entry = train_section[k]
            if not (isinstance(entry, list) and len(entry) == 2):
                raise ValueError(f"Invalid fold entry for {k}; expected [train_ids, val_ids].")
            tr_ids = [str(x) for x in entry[0]]
            va_ids = [str(x) for x in entry[1]]
            folds.append((tr_ids, va_ids))

        # strong consistency: no overlap between test and any train/val
        flat_trainval = set([i for tr, va in folds for i in tr+va])
        overlap = set(test_ids) & flat_trainval
        if overlap:
            raise ValueError(f"IDs overlap between test and train/val: {sorted(list(overlap))[:5]} ...")

        # soft warning: val IDs in multiple folds
        val_counter = Counter([vid for _, va in folds for vid in va])
        multi = [vid for vid, c in val_counter.items() if c > 1]
        if multi:
            self.logger.warning(f"Validation IDs appear in multiple folds (first occurrence will be used): {multi[:5]} ...")
            self.error.warning(f"Warning: Validation IDs in multiple folds: {multi[:5]} ...")

        return test_ids, folds, len(folds)

    def _make_listfold_splitter_from_ids(self, X_index, folds):
        """
        Build a splitter that enforces EXACT AutoRadiomics fold membership:
        For each fold f: train_idx := indices of train_ids[f], val_idx := indices of val_ids[f].
        Any IDs missing in X are ignored (with a warning). Raises if a fold ends up empty.
        """

        id_to_pos = {str(idx): i for i, idx in enumerate(list(X_index))}
        fold_indices = []
        missing_any = False
        for fold_id, (tr_ids, va_ids) in enumerate(folds):
            tr_idx = [id_to_pos[i] for i in tr_ids if i in id_to_pos]
            va_idx = [id_to_pos[i] for i in va_ids if i in id_to_pos]

            miss_tr = [i for i in tr_ids if i not in id_to_pos]
            miss_va = [i for i in va_ids if i not in id_to_pos]
            if miss_tr or miss_va:
                missing_any = True
                if miss_tr:
                    self.logger.warning(f"Fold {fold_id}: {len(miss_tr)} train IDs from splits.json not in X_train. Example: {miss_tr[:5]}")
                if miss_va:
                    self.logger.warning(f"Fold {fold_id}: {len(miss_va)} val IDs from splits.json not in X_train. Example: {miss_va[:5]}")

            if len(va_idx) == 0:
                raise ValueError(f"Fold {fold_id} has no validation samples after alignment with current X_train.")
            if len(tr_idx) == 0:
                raise ValueError(f"Fold {fold_id} has no training samples after alignment with current X_train.")

            fold_indices.append( (np.array(tr_idx, dtype=int), np.array(va_idx, dtype=int)) )

        if missing_any:
            print("Warning: Some IDs from splits.json were not found in X_train; see logger for details.")
        
        return _ListFoldSplitter(fold_indices)


    def __init__(self,
                 data: pd.DataFrame(),  # data where the model looks at
                 Prediction_Label: str,  # Name of the label in data
                 out_folder: str,
                 selected_features_path: str = None,  # path to selected features folder if data needs to get revisited
                 predict_only: bool = False,  # if only prediction is needed and no training
                 ensemble: bool = True,
                 ensemble_best_models: bool = False,  # if the best models should be ensembled
                 ensemble_best_n_models: int = 3,  # how many of the best models raked should be incuded in the ensembling
                 model_save_dir: str = None,
                 plot_save_dir: str = None,
                 model_name=None,
                 model=None,  # model to be used or list of models to be used
                 model_params=None,
                 model_type=None,
                 model_path=None,   # path to pretrained model
                 use_cross_validation: bool = True,
                 rand_state: float = None,
                 train_idx: list = None,  # ID of training samples is specific samples are needed
                 test_idx: list = None,  # ID of test samples is specific samples are needed
                 val_idx: list = None,  # ID of validate samples is specific samples are needed
                 logger=None,  # logger for log
                 error=None,  # logger for error
                 RunID=None,  # RunID for selection of the Run
                 cross_val_splits=5,  # number of splits for cross-validation
                 n_cpus: int = 1,  # number of CPUs to be used
                 log_file_path: str = "",
                 test_size: float = 0.3,
                 cv=None,
                 autorad_config=False,  # Performing cross validation and seeding like AutoRadiomics
                 groups=None,
                 scoring: str = "roc_auc",
                 device: str = "cuda",  # Available: cpu, gpu, cuda
                 stable_pretraining: bool = True,
                 # whether to perform pre evaluaion of model size where staböe performance has been reached
                 shap_plot_save_dir: str = None,  # Path to plot SHAP plots
                 roc_plot_save_dir: str = None,  # Path to plot ROC curves
                 run_neptune: bool = False,  # Use neptune for model training visualization
                 neptune_project: str = None,  # Neptune project to enter
                 neptune_api_token: str = None,  # Neptune api token for using neptune
                 task: str = None,  # Task to perform:  "binary_classification", "multi_class_classification" or "regression" if not set id > 5 different vales = regression else multi_class_classification
                 model_names: list = None,  # possibility to provide names of models
                 best_selected_features_folder_path: str = None,
                 # Path to folder from feature selection where pairs of best features are stored
                 optimize: bool = True,
                 extended_parameter_set: bool = True,
                 super_extended_parameter_set: bool = False,
                 use_optuna: bool = True,
                 optimization_iter: int = 200,  # number of iterations for hyperparameter optimizetion and training
                 use_wandb: bool = False,
                 wandb_login_key: str = None,
                 wandb_project_name: str = None,
                 shap_analysis: bool = True,
                 neptune_run_name: str = None,
                 imbalance_method: str = "SMOTE",  # method to handle class imbalance 'SMOTE', 'BorderlineSMOTE', 'RandomOver', 'RandomUnder', 'RepeatedEditedNearestNeighbours'
                 autoradiomics_splits_path: str = None,  # get json filts file directly from outoradiomics run
                 ):

        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        self.data = data
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.val_idx = val_idx
        self.Prediction_Label = Prediction_Label
        self.selected_features_path = selected_features_path
        self.predict_only = predict_only
        self.out_folder = out_folder
        self.ensemble = ensemble
        self.model_save_dir = model_save_dir
        self.plot_save_dir = plot_save_dir
        self.model_name = model_name
        self.model = model
        self.model_params = model_params
        self.model_type = model_type
        self.model_path = model_path
        self.ensemble_best_models = ensemble_best_models
        self.ensemble_best_n_models = ensemble_best_n_models
        self.use_cross_validation = use_cross_validation
        self.rand_state = rand_state
        self.logger = logger
        self.error = error
        self.RunID = RunID
        self.cross_val_splits = cross_val_splits
        self.n_cpus = n_cpus
        self.log_file_path = log_file_path
        self.test_size = test_size
        self.cv = cv
        self.autorad_config = autorad_config
        self.groups = groups
        self.scoring = scoring
        self.device = device
        self.stable_pretraining = stable_pretraining
        self.shap_plot_save_dir = shap_plot_save_dir
        self.roc_plot_save_dir = roc_plot_save_dir
        self.run_neptune = run_neptune
        self.neptune_project = neptune_project
        self.neptune_api_token = neptune_api_token
        self.task = task
        self.model_names = model_names
        self.best_selected_features_folder_path = best_selected_features_folder_path
        self.optimize = optimize
        self.extended_parameter_set = extended_parameter_set
        self.use_optuna = use_optuna
        self.optimization_iter = optimization_iter
        self.use_wandb = use_wandb
        self.wandb_login_key = wandb_login_key
        self.wandb_project_name = wandb_project_name
        self.shap_analysis = shap_analysis
        self.neptune_run_name = neptune_run_name
        self.imbalance_method = imbalance_method
        self.autoradiomics_splits_path = autoradiomics_splits_path

        self.model_instance = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.X_train_global = pd.DataFrame()
        self.X_test_global = pd.DataFrame()
        self.X_val_global = pd.DataFrame()

        self.ml_models = None
        self.simple_models = None
        self.available_models = None
        self.feature_selection_models = None

        self.super_extended_parameter_set = super_extended_parameter_set

        self.ensemble_per_model = {}

        self.fold = None

        if self.rand_state is None:
            self.rand_state = 1234
        
        if self.autorad_config: # adopt seed to autoradiomics used seed for syncronization
            self.rand_state = 123

        if self.use_wandb:
            if self.wandb_login_key is None:
                self.error.warning("Wandb logging is not possible. Please enter a wandb key in wandb_login_key")
            else:
                wandb.login(key=self.wandb_login_key)

        if self.use_optuna:
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        if self.cv is None:
            if self.groups is not None:
                self.cv = GroupShuffleSplit(n_splits=self.cross_val_splits,
                                            test_size=self.test_size,
                                            random_state=random.seed(self.rand_state))
            else:
                if self.autorad_config:
                    self.cv = StratifiedKFold(n_splits=self.cross_val_splits, shuffle=True, random_state=self.rand_state)
                else:
                    self.cv = ShuffleSplit(n_splits=self.cross_val_splits,
                                        test_size=self.test_size,
                                        random_state=random.seed(self.rand_state))
        
        # control threading
        thread_limit = str(self.n_cpus)
        for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
            os.environ[var] = thread_limit

        # seed everyting
        random.seed(self.rand_state)
        os.environ["PYTHONHASHSEED"] = str(self.rand_state)
        np.random.seed(self.rand_state) 
        torch.manual_seed(self.rand_state)
        torch.cuda.manual_seed(self.rand_state)
        torch.backends.cudnn.deterministic = True

        if self.RunID is None:
            self.RunID = time.strftime("%Y%m%d-%H%M%S")

        if self.run_neptune:
            if self.neptune_run_name is None:
                self.neptune_run_name = "RPTK run " + self.RunID

        self.logger = LogGenerator(
            log_file_name=self.out_folder + "/RPTK_model_training_" + self.RunID + ".log",
            logger_topic="RPTK Model Training"
        ).generate_log()

        self.error = LogGenerator(
            log_file_name=self.out_folder + "/RPTK_model_training_" + self.RunID + ".err",
            logger_topic="RPTK Model Training error"
        ).generate_log()

        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)

        if self.model_save_dir is None:
            self.model_save_dir = self.out_folder + "/models"
            self.logger.info(
                "Model save directory not specified. Using default directory: {}".format(self.model_save_dir))

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        if self.best_selected_features_folder_path is None:
            self.best_selected_features_folder_path = os.path.abspath(
                os.path.join(os.path.dirname(self.out_folder), 'selected_features', "model_out"))

        self.pre_training_results = pd.DataFrame({"Model": [],
                                                  "Parameter_name": [],
                                                  "Stable_Parameter_value": [],
                                                  "Stable_performance_AUROC": [],
                                                  "Max_Parameter_value": [],
                                                  "Max_performance_AUROC": []
                                                  })

        self.model_parameter = pd.DataFrame({"Models": [],
                                             "Params": [],
                                             "Val_AUC": [],
                                             "Test_AUC": []
                                             })

        self.prediction_summary = pd.DataFrame({"Models": [],
                                                "Val_AUC": [],
                                                "Test_AUC": [],
                                                "Bootstrap_AUC": [],
                                                "Bootstrap_AUC_std": [],
                                                "Bootstrap_Sensif": [],
                                                "Bootstrap_Sensif_std": [],
                                                "Bootstrap_Specif": [],
                                                "Bootstrap_Specif_std": [],
                                                "Bootstrap_F1": [],
                                                "Bootstrap_F1_std": [],
                                                "Val_F1": [],
                                                "Test_F1": [],
                                                "Val_ACC": [],
                                                "Test_ACC": [],
                                                "Val_AP": [],
                                                "Test_AP": []
                                                })

        if os.path.exists(self.out_folder + "/Prediction_summary.csv"):
            if os.path.getsize(self.out_folder + "/Prediction_summary.csv") > 0:
                self.prediction_summary = pd.read_csv(self.out_folder + "/Prediction_summary.csv", index_col=0)
            else:
                self.error.warning("Found empty Prediction_summary.csv. Maybe from failed run. Ignoring this file ...")

    def set_seed(self, seed=1234):
        """
        Seed various random number generators to ensure reproducibility.

        Args:
            seed (int): Seed value to set for random number generators.

        Returns:
            None
        """

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_unique_index(self, X, y, random_state):
        """
        Get unique index for splitting if multiple ROIs per sample for group wise split
        X and y Need to have ID as index
        """

        X = X.reset_index()
        y = y.reset_index()

        unique_idx = list(set(X["ID"].values.tolist()))
        unique_y = y.drop_duplicates()
        unique_y = unique_y.set_index("ID")

        train_idx, test_idx, _, _ = train_test_split(unique_idx,
                                                     unique_y[self.Prediction_Label],
                                                     random_state=random_state,
                                                     shuffle=True,
                                                     test_size=self.test_size)
        return train_idx, test_idx
    
    def drop_non_float_convertible_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features which are not numbers
        """

        convertible_cols = []
        for col in df.columns:
            try:
                df[col].astype(float)
                convertible_cols.append(col)
            except (ValueError, TypeError):
                # Skip columns that cannot be converted to float
                self.error.warning(f"Could not convert {col} to float! Dropping ...")
                print(f"Could not convert {col} to float! Dropping ...")
                continue

        return df[convertible_cols]

    def config_input_data(self, data:pd.DataFrame):
        """
        Check if input data has all necessary columns and is in correct format
        :param data: pd.DataFrame containing ID and predciton label
        :return data: pd.DataFrame with ID as index and containing predciton label
        """

        # Check if "ID" is in data columns and index drop ID in columns
        if "ID" in data.columns:
            if data.index.name != "ID":
                data.index = data["ID"]
            else:
                data.drop(columns=["ID"], inplace=True)
        else:
            # if there is no ID --> need to generate
            if "config" == data.index.name:
                # remove duplicate
                if "config" in data.columns:
                    data.drop(columns=["config"], inplace=True)

                data["ID"] = data.index.apply(lambda x: str(x.split('_')[0]))
            else:
                if "config" in data.columns:
                    data["ID"] = data["config"].apply(lambda x: str(x.split('_')[0]))
                else:
                    print("Missing config and ID in data. Please provide Sample Identifier.")

        # self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        if "ID" != data.index.name:
            if "ID" in data.columns:
                data.index = data["ID"]
                data.drop(columns=["ID"], inplace=True)
            elif "config" != data.index.name:
                print(data.index.name, "ID" in data.columns)
                self.error.error("No ID provided in input data.")
                raise ValueError("No ID provided in input data! Check input data!")

        # check for columns included lists
        # Identify columns that contain lists
        list_columns = [col for col in data.columns if data[col].apply(lambda x: isinstance(x, list)).any()]
        
        if len(list_columns) > 0:
            self.logger.info("Found {} columns with lists in the data: {}".format(len(list_columns), str(list_columns)))
            print("Found {} columns with lists in the data: {}".format(len(list_columns), str(list_columns)))

            # Replace lists with their first element
            for col in list_columns:
                data[col] = data[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

        if len(data.copy()[data.copy().duplicated()]) > 0:
            print("Found {} duplicated Samples in the data!".format(str(len(data.copy()[data.copy().duplicated()]))))
            self.error.warning("Found {} duplicated Samples in the data!".format(str(len(data.copy()[data.copy().duplicated()]))))
            data = data.drop_duplicates()

        if self.Prediction_Label is None:
            self.error.error("Please provide a prediction label.")
        elif self.Prediction_Label not in data.columns:
            self.error.error("Could not find prediction label: {}".format(self.Prediction_Label))
            print("Could not find prediction label: {}".format(self.Prediction_Label))
        
        # drop non float parameters
        data = self.drop_non_float_convertible_columns(data)

        return data

    def export_used_splits_simple(self, filepath: str = None, X_folds_val: list = None, X_folds_train: list = None):
        """
        Write the splits actually used by this trainer in the SAME JSON structure
        as AutoRadiomics (split_type/test/train with fold_* arrays of [train_ids, val_ids]).

        Requirements:
        - AutoRadiomics splits were provided (self.autoradiomics_splits_path not None).
        - self._autoradiomics_folds: list of (train_ids, val_ids) per fold (strings)
        - self._autoradiomics_test_ids: list of test IDs (strings)
        - X_folds_val: list for use in RPTK giving IDs for left out fold
        - X_folds_train: list for use in RPTK giving IDs for training folds

        The method filters to the IDs present in X_train/X_test after alignment,
        and preserves the order from the original splits.
        """
        test_list = []
        train_section = {}

        if self.autoradiomics_splits_path is not None:
            if getattr(self, "autoradiomics_splits_path", None) is None:
                raise ValueError("AutoRadiomics splits are not active; nothing to export in AR format.")
            if getattr(self, "_autoradiomics_folds", None) is None:
                raise ValueError("AutoRadiomics folds are not available; ensure split_data() has run with AR splits.")
            if getattr(self, "_autoradiomics_test_ids", None) is None:
                # still allow writing, but as empty test if not set
                self._autoradiomics_test_ids = []

            # Ensure we only write IDs that are present in the aligned data
            present_train = set(str(i) for i in self.X_train.index)
            present_test  = set(str(i) for i in self.X_test.index) if hasattr(self, "X_test") else set()

            
            for fold_id, (tr_ids, va_ids) in enumerate(self._autoradiomics_folds):
                tr_ids_present = [str(i) for i in tr_ids if str(i) in present_train]
                va_ids_present = [str(i) for i in va_ids if str(i) in present_train]
                train_section[f"fold_{fold_id}"] = [tr_ids_present, va_ids_present]
        else:
            present_train = set(str(i) for i in self.X_train.index)
            present_test  = set(str(i) for i in self.X_test.index)

            fold_id = 0
            for train, val in zip(X_folds_train, X_folds_val):
                train_section[f"fold_{str(fold_id)}"] = [train, val]
                fold_id += 1

        if self.autoradiomics_splits_path is not None:
            test_list = [str(i) for i in self._autoradiomics_test_ids if str(i) in present_test]
        else:
            test_list = present_test

        splits = {
            "split_type": "test + cross-validation on the training",
            "test": test_list,
            "train": train_section,
        }

        # default output path
        if filepath is None:
            os.makedirs(self.out_folder, exist_ok=True)
            filepath = os.path.join(self.out_folder, "splits_used.json")

        with open(filepath, "w") as f:
            json.dump(splits, f, indent=4)

        self.logger.info(f"Wrote splits to: {filepath}")
        print(f"Wrote splits to: {filepath}")
        return filepath

    def config_index_compatiblilty(self, x: pd.DataFrame, index_list:list, y: pd.Series= None):
        """
        Ensure that idx in index_list are also index in x and y
        x: Data wihtout prediction label
        y: label with sync index 
        index_list: list of indexes
        return 
        x,y with sync index
        """
        x["correct_index"] = np.nan

        for x_index in x.copy().index.to_list():
            if x_index not in index_list:
                if "_" in x_index:
                    # check if index is the first part sepperated by _ --> this happends if the index was a number
                    sub_id = x_index.split('_')[0]
                    for id in index_list:
                        if str(id) == str(sub_id):
                            x.loc[x_index, "correct_index"] = str(sub_id)
                            break
                elif " " in x_index:
                    # check if index is the first part sepperated by _ --> this happends if the index was a number
                    sub_id = x_index.split(' ')[0]
                    for id in index_list:
                        if str(id) == str(sub_id):
                            x.loc[x_index, "correct_index"] = str(sub_id)
                            break
                else:
                    # check if ID might be a substing
                    for id in index_list:
                        if str(id) != str(x_index):
                            if str(id) in str(x_index):
                                x.loc[x_index, "correct_index"] = str(id)
                                self.logger.info("Found {} as a part of {}. Please double check the correct transformation and correct if necessay!".format(str(id), str(x_index)))
                                print("Found {} as a part of {}. Please double check the correct transformation and correct if necessay!".format(str(id), str(x_index)))
                                self.error.warning("Found {} as a part of {}. Please double check the correct transformation and correct if necessay!".format(str(id), str(x_index)))
                                break
            else:
                x.loc[x_index, "correct_index"] = str(x_index)

        if x['correct_index'].isnull().sum() > 0:
            self.error.error("Could not resolve index incompatibility! Check train/test/validation index and syncronize it.")
            print("Could not resolve index incompatibility! Check train/test/validation index and syncronize it.")
            return None, None
            # raise ValueError("Could not resolve index incompatibility! Check train/test/validation index and syncronize it.")
        else:
            self.logger.info("Resolved index incompatibility!")
            print("Resolved index incompatibility!")

            # resolve index
            if not y is None:
                x[self.Prediction_Label] = y

            x.set_index("correct_index", inplace=True)
            x.index = x.index.rename('ID')
            x.index = x.index.astype(str, copy = False)

            if not y is None:
                y = x[self.Prediction_Label]
                x.drop([self.Prediction_Label], axis = 1, inplace = True) 

        return x,y

    def verify_scoring_matrics(self):
        """
        According to the task we select the correct metrics
        """

        # Adapt matrix to the task
        if self.task == "binary_classification":
            self.scoring = "roc_auc"
        elif self.task == "multi_class_classification":
            self.scoring = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
        else:
            self.scoring = "neg_mean_absolute_error"

    def verify_task(self):
        """
        Verify task and set y accordingly
        """

        if len(self.data[self.Prediction_Label].unique()) == 2:
            self.task = "binary_classification"
        elif len(self.data[self.Prediction_Label].unique()) < 5:
            self.task = "multi_class_classification"
        elif len(self.data[self.Prediction_Label].unique()) > 5:
            self.task = "regression"

        if self.task == "binary_classification" or self.task == "multi_class_classification":
            self.y = self.data[self.Prediction_Label].astype('int')
        elif self.task == "regression":
            self.y = self.data[self.Prediction_Label].astype('float')
        else:
            self.error.error(
                "Task {} not supported! Please take one of the following: binary_classification, multi_class_classification, or regression".format(
                    str(self.task)))

        print(3 * "#", "Performing", self.task)
        print("Detected {} Predictions Labels: {}".format(str(len(self.data[self.Prediction_Label].unique())),str(self.data[self.Prediction_Label].unique())))
        print("Feature space",self.data.shape)

    def random_control(self, random_state: int = None):
        """
        Random control for data split and training
        :param random_state: seed for splitting
        """

        if random_state is None:
            random_state = self.rand_state

        self.set_seed(seed=random_state)

    def predict_class_imbalance(self, df, label_col='Prediction_Label', threshold=1.5):
        """
        Predicts whether the dataset is imbalanced, checks if the imbalance is significant,
        and plots the class distribution.

        :param df: Input dataframe with a target column.
        :param label_col: Name of the target column containing class labels.
        :param threshold: Imbalance threshold (default 1.5). If the ratio of the most 
                        common class to the least common class exceeds this value, 
                        the dataset is considered imbalanced.
        :return: A dictionary with class distribution, imbalance ratio, and significance.
        """

        print("Check for class imbalance ...")
        self.logger.info("Check for class imbalance ...")

        # Compute class distribution
        class_counts = df[label_col].value_counts()
        
        # Calculate imbalance ratio
        majority_class = class_counts.max()
        minority_class = class_counts.min()
        imbalance_ratio = majority_class / minority_class if minority_class > 0 else float('inf')

        # Predict imbalance
        is_imbalanced = imbalance_ratio > threshold


        # Check if the imbalance is significant
        total_samples = len(df)

        # if we only look at data splits imbalance could be lowered
        if total_samples < len(self.data):
            class_counts_total = self.data[label_col].value_counts()
            majority_class = class_counts_total.max()
            minority_class = class_counts_total.min()
            total_samples = len(self.data)

        imbalance_percentage = round((majority_class / total_samples),1) * 100
        is_significant = imbalance_percentage >= 70  # Consider significant if one class dominates >70%
        
        if is_significant:
            print(f"Class imbalance is significant in dataset. {self.imbalance_method} will be applied")
            self.logger.info(f"Class imbalance is significant in dataset. {self.imbalance_method} will be applied")
        else:
            print("No significant class imbalance detected.")
            self.logger.info("No significant class imbalance detected.")

        return {
            "class_distribution": class_counts.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced,
            "is_significant": is_significant
        }

    def resolve_class_imbalance(self, df, label_col='Prediction_Label', method='SMOTE', random_state=42):
        """
        Perform oversampling using the specified method and append simulated samples
        with 'simu-' prefixed to their ID. Additionally, preserve string columns by
        reassigning them to their original IDs after oversampling.

        :param df: Input dataframe with ID as index and Prediction_Label as the target.
        :param label_col: Name of the target column to balance.
        :param method: Method for oversampling ('SMOTE', 'BorderlineSMOTE', 'RandomOver', 'RandomUnder').
        :param random_state: Random seed for reproducibility.
        :return: Oversampled dataframe with simulated IDs and string values reassigned.
        """
        # Separate string and numeric columns
        string_cols = df.select_dtypes(include=['object']).columns.tolist()
        num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        num_cols.remove(label_col)  # Keep label column separate
        
        # Store string values separately (indexed by ID)
        string_values = df[string_cols].copy()
        df = df[num_cols + [label_col]]  # Keep only numeric and target column
        
        # Oversampling method selection
        if method == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
        elif method == 'BorderlineSMOTE':
            sampler = BorderlineSMOTE(random_state=random_state)
        elif method == 'RandomOver':
            sampler = RandomOverSampler(random_state=random_state)
        elif method == 'RandomUnder':
            if self.task == "binary_classification":
                # accept 2 : 1 ratio for under sampling
                sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=0.5)
            else:
                sampler = RandomUnderSampler(random_state=random_state)
        elif method == 'RepeatedEditedNearestNeighbours':
            sampler = RepeatedEditedNearestNeighbours(n_jobs= self.n_cpus)
        else:
            raise ValueError("Unsupported method. Use 'SMOTE', 'BorderlineSMOTE', 'RepeatedEditedNearestNeighbours', 'RandomOver' or 'RandomUnder'")
        
        # Apply oversampling
        X_resampled, y_resampled = sampler.fit_resample(df.drop(columns=[label_col]), df[label_col])
        
        # Create new dataframe
        df_resampled = pd.DataFrame(X_resampled, columns=df.drop(columns=[label_col]).columns)
        df_resampled[label_col] = y_resampled
        
        # Assign new IDs, mark simulated ones
        original_ids = df.index.tolist()

        # adding samples to the data
        if method == 'RandomOver' or method == "SMOTE" or method == "BorderlineSMOTE":
            num_new_samples = len(df_resampled) - len(df)
            if num_new_samples > 0:
                simulated_ids = [f"simu-{i}" for i in range(num_new_samples)]
                new_index = original_ids + simulated_ids
            else:
                new_index = original_ids
            
            df_resampled.index = new_index

            # Re-integrate string values
            final_string_values = pd.DataFrame(index=df_resampled.index, columns=string_cols)
            for col in string_cols:
                final_string_values.loc[original_ids, col] = string_values[col]
                for sim_id in simulated_ids:
                    orig_id = np.random.choice(original_ids)  # Randomly assign an original ID's string values
                    final_string_values.loc[sim_id, col] = string_values.loc[orig_id, col]
            
            # Merge numerical and string values back
            df_final = df_resampled.join(final_string_values)
        else:
            # Removing samples from the data
            df_final = df_resampled

        return df_final


    def feature_format_check(self):
        """
        Check if features in X are in correct format and convert them if necessary
        :param feature_names: list of feature names
        """

        feature_names = self.X.columns

        # Check for correct feature format
        for feature in tqdm(feature_names, desc="Checking Feature Format", unit="Feature"):
            if feature in self.X.columns:
                self.X[feature] = pd.to_numeric(self.X[feature], downcast='float', errors='ignore') # self.X[feature].astype(float)
                
                # remove if column is None
                if feature is None:
                    if feature in self.X.columns:
                        self.X = self.X.drop([None], axis=1)
                        continue

                if self.X[feature].dtype == "object":
                    # for format problems if there is a , as digit seperator we need to convert it accordingly
                    try:
                        self.X[feature] = self.X[feature].replace(',','.', regex=True).astype(float)
                    except:
                        # if this is not the problem it might be a string parameter
                        self.error.warning("Need to drop {} because it is not a numerical feature.".format(feature))
                        print("Need to drop {} because it is not a numerical feature.".format(feature))

                        self.X = self.X.drop(feature, axis=1)

        # check format
        if self.X.index.name != "ID":
            if "ID" in self.X.columns:
                self.X.index = self.X["ID"]
                print("Need to config data as index needs to be ID.")
                self.logger.info("Need to config data as index needs to be ID.")
            else:
                self.error.error("ID is missing in dataset. Please check your data!")
                raise ValueError("ID is missing in dataset. Please check your data!")

    # 1. Split Data
    def split_data(self, random_state=None):
        """
        Split train/validation/test set for model training.
        :param random_state: seed for splitting
        """

        if random_state is None:
            random_state = self.rand_state

        self.random_control(random_state=random_state)

        print("Splitting Feature space", self.data.shape)

        # check if we have already data splits we want to go for
        self.train_indx_path = self.out_folder + '/Train_idx_' + str(self.RunID) + "_" + str(random_state) + '.csv'
        self.val_indx_path = self.out_folder + '/Val_idx_' + str(self.RunID) + "_" + str(random_state) + '.csv'
        self.test_indx_path = self.out_folder + '/Test_idx_' + str(self.RunID) + "_" + str(random_state) + '.csv'

        self.data = self.config_input_data(data=self.data.copy())

        if self.data.index.name != "ID":
            if "ID" in self.data.columns:
                self.data.set_index("ID", inplace=True)
            else:
                self.error.error("ID is missing in input data.")
                raise ValueError("ID is missing in input data.")

        # Check if data includes loaded index for training and testing data
        if (not self.train_idx is None) and (not self.test_idx is None):
            IDX_not_included = False
            for idx in self.train_idx:
                if idx not in self.data.index.to_list():
                    self.error.warning("Provided Training or Testing index {} is not included in data. Trying to fix this ...".format(str(idx)))
                    print("Provided Training or Testing index {} is not included in data. Trying to fix this ...".format(str(idx)))
                    IDX_not_included = True
                    break

            if not IDX_not_included:
                for idx in self.test_idx:
                    if idx not in self.data.index.to_list():
                        self.error.warning("Provided Training or Testing index {} is not included in data. Trying to fix this ...".format(str(idx)))
                        print("Provided Training or Testing index {} is not included in data. Trying to fix this ...".format(str(idx)))
                        IDX_not_included = True
                        break
            # get path to selected feautes

            # check if index config is wrong
            if IDX_not_included:
                data, _ = self.config_index_compatiblilty(x=self.data.copy(), index_list=self.train_idx + self.test_idx, y= None) 
                
                if not isinstance(self.train_idx[0], str):
                    self.train_idx = [str(e) for e in self.train_idx]
                    self.test_idx = [str(e) for e in self.test_idx]        

                if not data is None:
                    self.data = data
                else:
                    if self.selected_features_path is None:
                        selected_features_path = str(Path(self.out_folder).parent) + "/selected_features/Feature_selection_"
                    else:
                        selected_features_path = self.selected_features_path + "Feature_selection_"

                    selected_feature_file_path = glob.glob(selected_features_path + "*.csv")

                    if len(selected_feature_file_path) > 1:
                        self.error.error("Could not perform model prediction! Can not find correct feature set from selection and provided IDs are not fitting to data. Found files: " + str(selected_feature_file_path))
                        raise ValueError("Could not perform model prediction! Can not find correct feature set from selection and provided IDs are not fitting to data. Found files: " + str(selected_feature_file_path))
                    elif len(selected_feature_file_path) == 0:
                        self.error.error("Could not perform model prediction! Can not find correct feature set from selection and provided IDs are not fitting to data.  No files found.")
                        raise ValueError("Could not perform model prediction! Can not find correct feature set from selection and provided IDs are not fitting to data. No files found.")
                    else:
                        selected_features = pd.read_csv(selected_feature_file_path[0], index_col=0)
                        self.logger.info("Trying to load data from selected features ...")
                        print("Trying to load data from selected features ...")
                        self.data = self.config_input_data(data=selected_features.copy())

                        IDX_not_included = False
                        for idx in self.train_idx:
                            if idx not in self.data.index.to_list():
                                self.error.warning("Provided Training or Testing index {} is not included in data at all.".format(str(idx)))
                                #raise ValueError("Provided Training or Testing index {} is not included in data at all.".format(str(idx)))
                                IDX_not_included = True
                                break

                        if not IDX_not_included:
                            for idx in self.test_idx:
                                if idx not in self.data.index.to_list():
                                    self.error.warning("Provided Training or Testing index {} is not included in data at all.".format(str(idx)))
                                    #raise ValueError("Provided Training or Testing index {} is not included in data at all.".format(str(idx)))
                                    IDX_not_included = True
                                    break

                        if IDX_not_included:
                            data, _ = self.config_index_compatiblilty(x=self.data.copy(), index_list=self.train_idx + self.test_idx, y= None) 
                            if not data is None:
                                self.data = data
                            else:
                                raise ValueError("Could not resolve index incompatibility! Check train/test/validation index and syncronize it.")

                            if not isinstance(self.train_idx[0], str):
                                self.train_idx = [str(e) for e in self.train_idx]
                                self.test_idx = [str(e) for e in self.test_idx]                

        self.verify_task()

        self.verify_scoring_matrics()

        # Proceed with label distribution plots etc. using these assignments.
        self.X = self.data.loc[:, self.data.columns != self.Prediction_Label]
        
        self.X = self.X.loc[:,~self.X.columns.duplicated()]

        self.feature_format_check()

         # check if files exist make train idx and test idx
        if os.path.exists(self.train_indx_path) and os.path.exists(self.test_indx_path):
            self.train_idx = None
            self.test_idx = None
            self.val_idx = None

        # === Use AutoRadiomics splits if provided ===
        if self.autoradiomics_splits_path is not None:
            test_ids, folds, n_folds = self._load_autoradiomics_splits(self.autoradiomics_splits_path)
            # Build the training pool = union of all train/val IDs across folds
            train_pool_ids = []
            for tr, va in folds:
                train_pool_ids.extend(tr)
                train_pool_ids.extend(va)
            # unique, preserve order
            seen = set()
            train_pool_ids = [i for i in train_pool_ids if not (i in seen or seen.add(i))]

            # Ensure string indices for alignment
            self.data.index = self.data.index.astype(str)
            self.X.index = self.X.index.astype(str)
            self.y.index = self.y.index.astype(str)

            available = set(self.X.index)
            miss_test = [i for i in test_ids if i not in available]
            miss_train = [i for i in train_pool_ids if i not in available]
            if miss_test:
                self.logger.warning(f"{len(miss_test)} test IDs from splits.json not found in dataset. Example: {miss_test[:5]}")
            if miss_train:
                self.logger.warning(f"{len(miss_train)} train/val IDs from splits.json not found in dataset. Example: {miss_train[:5]}")

            valid_test_ids = [i for i in test_ids if i in available]
            valid_train_ids = [i for i in train_pool_ids if i in available]

            # Assign final splits
            self.X_train = self.X.loc[valid_train_ids]
            self.y_train = self.y.loc[valid_train_ids]
            self.X_test  = self.X.loc[valid_test_ids]
            self.y_test  = self.y.loc[valid_test_ids]

            # For CV optimization, we do not keep a single fixed X_val here
            self.X_val = pd.DataFrame()
            self.y_val = pd.Series(dtype=self.y_train.dtype)
            
            # Prepare CV splitter that uses EXACT fold membership for (train, val)
            self.cv = self._make_listfold_splitter_from_ids(self.X_train.index, folds)
            self.cross_val_splits = n_folds

            # Store for downstream components if needed
            self._autoradiomics_folds = folds
            self._autoradiomics_test_ids = valid_test_ids

            # Persist indices for reproducibility (same format as before)
            self.train_idx = valid_train_ids
            self.test_idx  = valid_test_ids

            pd.Series({"ID":self.train_idx}, index=self.train_idx).to_csv(self.train_indx_path, index=False)
            pd.Series({"ID":self.test_idx}, index=self.test_idx).to_csv(self.test_indx_path, index=False)

            try:
                self.export_used_splits_simple()  # -> <out_folder>/splits_used.json
            except Exception as e:
                self.error.warning(f"Failed to write splits_used.json: {e}")
        
        if (not self.train_idx is None) and (not self.test_idx is None):

            self.X_train = self.X.loc[self.train_idx]
            self.y_train = self.y.loc[self.train_idx]

            self.X_test = self.X.loc[self.test_idx]
            self.y_test = self.y.loc[self.test_idx]

            # Perform cross-validation means no validation data is needed
            if self.cross_val_splits != 0:
                self.X_val = pd.DataFrame()
                self.y_val = pd.Series()

            elif self.val_idx is not None:
                self.X_val = self.X.loc[self.val_idx]
                self.y_val = self.y.loc[self.val_idx]

        elif os.path.exists(self.train_indx_path) and os.path.exists(self.test_indx_path):
            
            from rptk.rptk import RPTK
            
            # recreate train/val/test set from file
            train_idx_df = pd.read_csv(self.train_indx_path)
            test_idx_df = pd.read_csv(self.test_indx_path)
            
            #if train_idx_df["ID"].str.isnumeric().any() or test_idx_df["ID"].str.isnumeric().any():
            #    train_idx_df = RPTK.normalize_id_length_end(train_idx_df, id_col="ID")
            #    test_idx_df = RPTK.normalize_id_length_end(test_idx_df, id_col="ID")
 
            self.train_idx = train_idx_df.ID.astype('str').to_list()
            self.test_idx = test_idx_df.ID.astype('str').to_list()
            
            idxs = self.train_idx + self.test_idx
            incopartible_idxs = []
            index_compatible = True

            self.X.index = self.X.index.astype('str')
            self.y.index = self.y.index.astype('str')

            # check for correct index format
            for x_index in self.X.index.to_list():
                if x_index not in idxs:
                    index_compatible=False
                    incopartible_idxs.append(x_index)
                    self.logger.info(f"Need to configure index for syncronization {x_index}")
                    print(f"Need to configure index for syncronization {x_index}")
                    
            # only a few index are not included in the data
            if len(incopartible_idxs) != len(self.X.index.to_list()):
                 self.X.drop(incopartible_idxs, inplace=True)
                 self.y.drop(incopartible_idxs, inplace=True)
                 index_compatible=True

            if not index_compatible:

                X ,y = self.config_index_compatiblilty(x=self.X.copy(), y=self.y.copy(), index_list=idxs)

                if (not X is None) & (not y is None):
                    self.X = X
                    self.y = y
                else:
                    print(self.X.index[0],self.y.index[0])
                    raise ValueError("Could not resolve index incompatibility! Check train/test/validation index and syncronize it.")

                if not isinstance(self.train_idx[0], str):
                    self.train_idx = [str(e) for e in self.train_idx]
                    self.test_idx = [str(e) for e in self.test_idx]


            self.X_train = self.X.loc[self.train_idx]
            self.y_train = self.y.loc[self.train_idx]

            self.X_test = self.X.loc[self.test_idx]
            self.y_test = self.y.loc[self.test_idx]

            # Perform cross-validation means no validation data is needed
            if self.cross_val_splits != 0:
                self.X_val = pd.DataFrame()
                self.y_val = pd.Series()

            elif os.path.exists(self.val_indx_path):
                
                val_indx_df = pd.read_csv(self.val_indx_path)
                if val_indx_df["ID"].str.isnumeric().any():
                    val_indx_df = RPTK.normalize_id_length_end(val_indx_df, id_col="ID")
                val_idx = val_indx_df.ID.to_list()

                self.X_val = self.X.loc[val_idx]
                self.y_val = self.y.loc[val_idx]

        else:
            # Split the data into training and testing sets (e.g. train 70%/test 30%)

            # if multiple ROIs are included
            if self.X.index.duplicated().sum() > 0:

                print("Got multiple ROIs. Need to group Train/Test split!")
                self.logger.info("Got multiple ROIs. Need to group Train/Test split!")

                train_idx, test_idx = self.get_unique_index(X=self.X.copy(), y=self.y.copy(), random_state=random_state)

                # convert to df
                y = self.y.reset_index()
                y = y.set_index("ID")

                self.X_train = self.X[self.X.index.isin(train_idx)]
                self.y_train = y[y.index.isin(train_idx)][self.Prediction_Label]

                self.X_test = self.X[self.X.index.isin(test_idx)]
                self.y_test = y[y.index.isin(test_idx)][self.Prediction_Label]


            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                        self.y,
                                                                                        random_state=random_state,
                                                                                        shuffle=True,
                                                                                        test_size=self.test_size)

            if self.cross_val_splits > 0:
                self.X_val = pd.DataFrame()
                self.y_val = pd.Series()

            else:

                if self.X_train.index.duplicated().sum() > 0:

                    train_idx, val_idx = self.get_unique_index(X=self.X_train.copy(), 
                                                               y=self.y_train.copy(),
                                                               random_state=random_state)

                    # convert to df
                    y_train = self.y_train.reset_index()
                    y_train = y_train.set_index("ID")

                    X_train = self.X_train[self.X_train.index.isin(train_idx)]
                    self.y_train = y_train[y_train.index.isin(train_idx)][self.Prediction_Label]

                    self.X_val = self.X_train[self.X_train.index.isin(val_idx)]
                    self.y_val = y_train[y_train.index.isin(val_idx)][self.Prediction_Label]

                    self.X_train = X_train
                    del X_train

                else:
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                                          self.y_train,
                                                                                          random_state=random_state,
                                                                                          shuffle=True,
                                                                                          test_size=self.test_size)

        train_idx = self.y_train.copy().index
        val_idx = self.y_val.copy().index
        test_idx = self.y_test.copy().index

        # check if no training index is in the testing data
        overlap = [x for x in train_idx if x in test_idx]
        if len(overlap) > 0:
            clean = [x for x in train_idx if x not in test_idx]

        if "Transformations" in self.X_train.columns:
            # remove information about Image Transformation from feature space
            self.X_train.drop(['Transformations'], axis = 1, inplace = True) 
            
        if "Transformations" in self.X_test.columns:
            self.logger.info("Remove transformed samples from test set.")
            print("Remove transformed samples from test set.")

            # make copy of data as sample size is impacted also the label needs to get included
            X_test_with_label = self.X_test.copy()
            X_test_with_label[self.Prediction_Label] = self.y_test.copy()

            # No transformed samples in Test set only features from original Images
            X_test_with_label = X_test_with_label.loc[X_test_with_label['Transformations'] == 0]
            
            # remove information about Image Transformation from feature space
            X_test_with_label.drop(['Transformations'], axis = 1, inplace = True) 

            self.y_test = X_test_with_label[self.Prediction_Label]

            # remove label from test set
            X_test_with_label = X_test_with_label.loc[:, X_test_with_label.columns != self.Prediction_Label]
            self.X_test = X_test_with_label.copy()

        if len(self.X_val) > 0:
            if 'Transformations' in self.X_val.columns:
                self.X_val.drop(['Transformations'], axis = 1, inplace = True)

        # Save Training
        pd.Series(train_idx, index=train_idx).to_csv(self.train_indx_path, index=False)

        # Save Validation
        pd.Series(val_idx, index=val_idx).to_csv(self.val_indx_path, index=False)

        # Save Testing
        pd.Series(test_idx, index=test_idx).to_csv(self.test_indx_path, index=False)

        # plot distribution of labels in test/training/validation set
        PerformancePlotter(output_path=self.out_folder + "/plots",
                           X=self.X,
                           y=self.y,
                           RunID=self.RunID,
                           logger=self.logger,
                           error=self.error, task = self.task).plot_label_distribution(training_series=self.y_train,
                                                                     testing_series=self.y_test,
                                                                     validation_series=self.y_val,
                                                                     )


        self.logger.info("Training data " + str(self.X_train.shape))
        self.logger.info("Validation data " + str(self.X_val.shape))
        self.logger.info("Testing data " + str(self.X_test.shape))
        
        print("Training data " + str(self.X_train.shape))
        print("Validation data " + str(self.X_val.shape))
        print("Testing data " + str(self.X_test.shape))

        if len(self.X_val) > 0:
            print("Validation data " + str(self.X_val.shape))

        # Save Data 
        train_data_path = self.out_folder + '/Train_data_' + str(self.RunID) + "_" + str(random_state) + '.csv'
        val_data_path = self.out_folder + '/Val_data_' + str(self.RunID) + "_" + str(random_state) + '.csv'
        test_data_path = self.out_folder + '/Test_data_' + str(self.RunID) + "_" + str(random_state) + '.csv'

        if not os.path.exists(train_data_path):
            X_train_incl_label = self.X_train.copy()
            #X_train_incl_label[self.Prediction_Label] = self.y_train.copy()
            X_train_incl_label.to_csv(train_data_path)

        if not os.path.exists(val_data_path):
            if len(self.X_val) > 0:
                X_val_incl_label = self.X_val.copy()
                #X_val_incl_label[self.Prediction_Label] = self.y_val.copy()
                X_val_incl_label.to_csv(val_data_path)

        if not os.path.exists(test_data_path):
            X_test_incl_label = self.X_test.copy()
            #X_test_incl_label[self.Prediction_Label] = self.y_test.copy()
            X_test_incl_label.to_csv(test_data_path)

        train_data = self.X_train.copy()
        train_data[self.Prediction_Label] = self.y_train.copy()

        imbalance = self.predict_class_imbalance(train_data, self.Prediction_Label)
        if imbalance["is_significant"]:
            if not self.imbalance_method is None:
                print(f"Significant class imbalance detected {imbalance['class_distribution']}. Applying {self.imbalance_method}.")
                self.error.warning(f"Significant class imbalance detected {imbalance['class_distribution']}. Applying {self.imbalance_method}.")

                train_data = self.resolve_class_imbalance(df=train_data, label_col=self.Prediction_Label, method=self.imbalance_method, random_state=random_state)
                self.X_train = train_data.drop(columns=[self.Prediction_Label])
                self.y_train = train_data[self.Prediction_Label]

                PerformancePlotter(output_path=self.out_folder + "/plots",
                                    X=self.X,
                                    y=self.y,
                                    RunID=self.RunID + "_imbalance_resolved",
                                    logger=self.logger,
                                    error=self.error, 
                                    task = self.task).plot_label_distribution(training_series=self.y_train,
                                                                                testing_series=self.y_test,
                                                                                validation_series=self.y_val,
                                                                                )

                self.logger.info("Upsampled Training data " + str(self.X_train.shape))
                self.logger.info("Validation data " + str(self.X_val.shape))
                self.logger.info("Testing data " + str(self.X_test.shape))
                
                print("Upsampled Training data " + str(self.X_train.shape))
                print("Validation data " + str(self.X_val.shape))
                print("Testing data " + str(self.X_test.shape))
            else:
                print(f"Significant class imbalance detected {imbalance['class_distribution']}. No imbalance resolution method specified.")
                self.error.warning(f"Significant class imbalance detected {imbalance['class_distribution']}. No imbalance resolution method specified.")

    # 2. Get model/s
    def generate_model_instance(self):
        """
        Define which model need to get trained and specify starting parameter
        self.model is the list of models or a single model
        """

        if self.task == "binary_classification":
            Gradient_boost = GradientBoostingClassifier(random_state=random.seed(self.rand_state))
            RF = RandomForestClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state))
            # LGBM = LGBMClassifier(n_jobs=self.n_cpus, random_state=self.rand_state, device=self.device, verbose=-1,
            #                     num_threads=self.n_cpus)
            if ("cuda" in self.device) or ("gpu" in self.device):
                # RF = RandomForestClassifier_GPU(random_state=random.seed(self.rand_state))
                LGBM = LGBMClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state), device=self.device, verbose=-1)

                XGBoost = XGBClassifier(use_label_encoder=False, n_jobs=self.n_cpus, eval_metric="auc",
                                        device=self.device, verbosity=0,
                                        random_state=random.seed(self.rand_state), objective="binary:logistic",
                                        tree_method='gpu_hist', predictor='gpu_predictor')

                TabNet = TabNetClassifier(  # feature_columns=self.X.columns.to_list(),
                    # num_classes=len(self.y.unique()),
                    verbose=0,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    device_name=self.device,
                    seed=self.rand_state)
            else:

                LGBM = LGBMClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state), device=self.device, verbose=-1,
                                      num_threads=self.n_cpus)

                XGBoost = XGBClassifier(use_label_encoder=False, n_jobs=int(self.n_cpus / 2), eval_metric="auc",
                                        device=self.device, verbosity=0,
                                        random_state=random.seed(self.rand_state), objective="binary:logistic")

                TabNet = TabNetClassifier(  # feature_columns=self.X.columns.to_list(),
                    # num_classes=len(self.y.unique()),
                    verbose=0,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    device_name="cpu",
                    seed=self.rand_state
                )

            linear_svc = SVC(kernel="linear", random_state=random.seed(self.rand_state), probability=True)
            svc = SVC(kernel="rbf", random_state=random.seed(self.rand_state), probability=True)  # loss="squared_hinge",
            # lasso = Lasso(random_state=random.seed(self.rand_state))

        elif self.task == "multi_class_classification":

            Gradient_boost = GradientBoostingClassifier(random_state=random.seed(self.rand_state))
            # LGBM = LGBMClassifier(n_jobs=self.n_cpus, random_state=self.rand_state, device=self.device, verbosity=-1, num_threads=self.n_cpus)
            RF = RandomForestClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state))
            if ("cuda" in self.device) or ("gpu" in self.device):
                # RF = RandomForestClassifier_GPU(random_state=random.seed(self.rand_state))
                LGBM = LGBMClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state), device=self.device, verbose=-1,
                                      num_threads=self.n_cpus)

                XGBoost = XGBClassifier(use_label_encoder=False, n_jobs=self.n_cpus, eval_metric="auc",
                                        device=self.device, verbosity=0,
                                        random_state=random.seed(self.rand_state), objective="multi:softprob",
                                        num_class=int(len(self.data[self.Prediction_Label].unique())),
                                        tree_method='gpu_hist', predictor='gpu_predictor')

                TabNet = TabNetClassifier(  # feature_columns=self.X.columns.to_list(),
                    # num_classes=len(self.y.unique()),
                    verbose=0,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    device_name=self.device,
                    seed=self.rand_state)
            else:
                RF = RandomForestClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state))
                LGBM = LGBMClassifier(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state), device=self.device, verbose=-1)

                XGBoost = XGBClassifier(use_label_encoder=False, n_jobs=int(self.n_cpus / 2), eval_metric="auc",
                                        device=self.device, verbosity=0, num_class=int(len(self.data[self.Prediction_Label].unique())),
                                        random_state=random.seed(self.rand_state), objective="multi:softprob")

                TabNet = TabNetClassifier(  # feature_columns=self.X.columns.to_list(),
                    # num_classes=len(self.y.unique()),
                    verbose=0,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    device_name="cpu",
                    seed=self.rand_state)

            linear_svc = SVC(kernel="linear", random_state=random.seed(self.rand_state), probability=True)  # loss="squared_hinge",
            svc = SVC(kernel="rbf", random_state=random.seed(self.rand_state), probability=True)
            # lasso = Lasso(random_state=random.seed(self.rand_state))

        else:
            # self.task == "regression":
            Gradient_boost = GradientBoostingRegressor(random_state=random.seed(self.rand_state))
            RF = RandomForestRegressor(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state))

            if ("cuda" in self.device) or ("gpu" in self.device):
                # RF = RandomForestRegressor_GPU(random_state=random.seed(self.rand_state))
                LGBM = LGBMRegressor(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state), device=self.device, verbose=-1)
                # params = {"tree_method": 'gpu_hist', "predictor": 'gpu_predictor'}

                XGBoost = XGBRegressor(use_label_encoder=False, n_jobs=int(self.n_cpus / 2), eval_metric="mae",
                                       device=self.device, verbosity=0,
                                       random_state=random.seed(self.rand_state), objective="reg:absoluteerror",
                                       tree_method='gpu_hist', predictor='gpu_predictor')

                TabNet = TabNetRegressor(n_d=24, n_a=24, n_steps=1, gamma=1.3,
                                         lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                                         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                         mask_type='entmax',
                                         scheduler_params=dict(mode="min",
                                                               patience=5,
                                                               min_lr=1e-5,
                                                               factor=0.9, ),
                                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                         verbose=300,
                                         seed=self.rand_state)
            else:
                RF = RandomForestRegressor(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state))

                LGBM = LGBMRegressor(n_jobs=self.n_cpus, random_state=random.seed(self.rand_state), device=self.device, verbose=-1,
                                     num_threads=self.n_cpus)

                XGBoost = XGBRegressor(use_label_encoder=False, n_jobs=int(self.n_cpus / 2), eval_metric="mae",
                                       device=self.device, verbosity=0,
                                       random_state=random.seed(self.rand_state), objective="reg:absoluteerror")

                TabNet = TabNetRegressor(n_d=24, n_a=24, n_steps=1, gamma=1.3,
                                        lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                                         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                         mask_type='entmax',
                                         scheduler_params=dict(mode="min",
                                                               patience=5,
                                                               min_lr=1e-5,
                                                               factor=0.9, ),
                                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                         verbose=300,
                                         seed=self.rand_state)

            linear_svc = SVC(kernel="linear", random_state=random.seed(self.rand_state), probability=True)
            svc = SVC(kernel="rbf", random_state=random.seed(self.rand_state), probability=True)
            # lasso = Lasso(random_state=random.seed(self.rand_state))

        # type(linear_svc).__name__ = "LinearSVC"

        self.ml_models = [RF,
                          TabNet,
                          Gradient_boost,
                          LGBM,
                          XGBoost]

        self.simple_models = [svc,
                              # lasso
                              ]

        self.available_models = self.ml_models + self.simple_models

        # XGBoost Selection on CPUs takes quite long so we are going to skip it
        if ("cuda" in self.device) or ("gpu" in self.device):
            self.feature_selection_models = [RF,
                                             Gradient_boost,
                                             # LGBM,
                                             XGBoost,
                                             # SklearnClassifier(SVC(kernel='linear',
                                             #                      probability=True,
                                             #                      random_state=random.seed(self.rand_state))),
                                             linear_svc,
                                             svc,
                                             # CalibratedClassifierCV(estimator=Lasso(random_state=random.seed(self.rand_state)),
                                             #                       n_jobs=self.n_cpus),
                                             KNeighborsClassifier(n_jobs=self.n_cpus)
                                             ]
        else:
            self.feature_selection_models = [RF,
                                             Gradient_boost,
                                             linear_svc,
                                             svc,
                                             KNeighborsClassifier(n_jobs=self.n_cpus)
                                             ]

        # No model to load
        if self.model_path is None:

            # No model to load but predict only
            if self.predict_only:
                self.error.error("No model to load! Please provide a model path to load a trained model.")
                raise ValueError("No model to load! Please provide a model path to load a trained model.")

            # no model defines
            if self.model is None:
                # no name of the model
                if self.model_name is None:
                    # take all available models
                    self.model = self.available_models

                # if multiple model names are defined
                elif isinstance(self.model_name, list):
                    self.model = []
                    # check if model name is in availabel models
                    for model in self.available_models:
                        if type(model).__name__ in self.model_name:
                            self.model.append(model)
                    if len(self.model) == 0:
                        self.error.error("Model names {} not in available models!".format(self.model_name))
                else:
                    # check if model name is in availabel models
                    for model in self.available_models:
                        if self.model_name == type(model).__name__:
                            self.model = [model]
                    if self.model is None:
                        self.error.error("Model name {} not in available models!".format(self.model_name))

            elif isinstance(self.model, list) or (self.model == "all"):

                if self.model == "all":
                    self.model = self.available_models
                else:
                    model_list = []
                    for model in self.model:
                        model_included = False
                        for available_model in self.available_models:
                            if type(model).__name__ != "str":
                                if type(model).__name__ == type(available_model).__name__:
                                    model_included = True
                            else:
                                if model == type(available_model).__name__:
                                    model_included = True
                                    model_list.append(available_model)

                        if not model_included:
                            if type(self.model).__name__ != "str":
                                self.error.warning("Model {} not in available models!".format(type(model).__name__))
                            else:
                                self.error.warning("Model {} not in available models!".format(model))
                    self.model = model_list

            else:
                model_included = False

                if self.model == "all":
                    self.model = self.available_models
                else:
                    for available_model in self.available_models:
                        if type(self.model).__name__ != "str":
                            if type(available_model).__name__ == type(self.model).__name__:
                                model_included = True
                        else:
                            if type(available_model).__name__ == self.model:
                                model_included = True
                                self.model = [available_model]

                    if not model_included:
                        self.error.error("Model {} not in available models!".format(type(self.model).__name__))
                        raise ValueError("Model {} not in available models!".format(type(self.model).__name__))
        else:
            print("Found trained model!")
            self.logger.info("Found trained model at " + str(self.model_path))
            self.model = [self.load_model(self.model_path)]
            self.model_name = type(self.model).__name__

            print(f"Loading trained {self.model_name} ...")
            self.logger.info(f"Loading trained {self.model_name} ...")

        if not len(self.model) > 1:
            self.ensemble = False

    # 2.1 Load existing models
    def load_model(self, file_path):
        """
        Load a model from model path
        :return: loaded model
        """
        if not os.path.exists(file_path):
            self.error.error("Model is not in file! Check the model file: " + str(file_path))
            raise ValueError("Model is not in file! Check the model file: " + str(file_path))
        else:
            loaded_model = pickle.load(open(file_path, 'rb'))
            self.logger.info(f"Loaded {type(loaded_model).__name__} model")

        if not loaded_model is None:
            try:
                if "Ensemble" in str(type(loaded_model).__name__):
                    if len(loaded_model.clfs) > 1:
                        cols_when_model_builds = loaded_model.clfs[0].feature_names_in_.tolist()
                else:
                    try:
                        cols_when_model_builds = loaded_model.feature_names_in_.tolist()
                    except:
                        print(f"Could not load features from {str(type(loaded_model).__name__)}. Taking features from input ...")
                        self.logger.info(f"Could not load features from {str(type(loaded_model).__name__)}. Taking features from input ...")
                        cols_when_model_builds = self.X_train_global.copy().columns.to_list()
                
                if len(cols_when_model_builds) < len(self.X_train.columns.to_list()):
                    features_in_data = self.X_train.columns.to_list()
                    missing_features = [feature for feature in features_in_data if feature not in cols_when_model_builds]
                    if len(missing_features) > 0:
                        print(f"Model is missing features for training: {missing_features}")
                        self.error.warning(f"Model is missing features for training: {missing_features}")

                # adapt order of features to loaded model so it can get fit - if it differs get from global copy
                self.X_train = self.X_train_global.copy()[cols_when_model_builds]
                self.X_test = self.X_test_global.copy()[cols_when_model_builds]

                if len(self.X_val) > 0:
                    self.X_val = self.X_val[cols_when_model_builds]
            except Exception as ex:
                if "Ensemble" in str(type(loaded_model).__name__):
                    if len(loaded_model.clfs) > 1:
                        model_name = str(type(loaded_model.clfs[0]).__name__) + " Ensemble"
                else:
                    model_name = str(type(loaded_model).__name__)

                self.error.warning("Loading {} without adapting feature order! {}".format(model_name, ex))
                print("Loading {} without adapting feature order! {}".format(model_name, ex))
                
                try:
                    cols_when_model_builds = loaded_model.feature_names_in_.tolist()
                except:
                    print(f"Could not load features from {str(type(loaded_model).__name__)}. Taking features from input ...")
                    self.logger.info(f"Could not load features from {str(type(loaded_model).__name__)}. Taking features from input ...")
                    cols_when_model_builds = self.X_train_global.copy().columns.to_list()
                
                if len(cols_when_model_builds) < len(self.X_train.columns.to_list()):
                    features_in_data = self.X_train.columns.to_list()
                    missing_features = [feature for feature in features_in_data if feature not in cols_when_model_builds]
                    if len(missing_features) > 0:
                        print(f"Model is missing features for training: {missing_features}")
                        self.error.warning(f"Model is missing features for training: {missing_features}")

                self.X_train = self.X_train_global.copy()[cols_when_model_builds]
                self.X_test = self.X_test_global.copy()[cols_when_model_builds]
        else:
            self.error.error("Model is not in file! Check the model file: " + str(file_path))
            raise ValueError("Model is not in file! Check the model file: " + str(file_path))

        return loaded_model

    # 2.2 save models
    def save_model(self, model, file_path):
        """
        Save a model to model path
        :return: saved model
        """
        pickle.dump(model, open(file_path, 'wb'))

    # 3. Get a stable and efficient model size
    def get_stable_model_size(self, model, param_name, param_range):
        """
        Calculate the min size of the model to get stable performance results
        :param model: model to optimize
        :param param_name: name of parameter to optimize
        :param param_range: range of parameter to test stability and performance
        :return: x_max, y_max, x_stable, y_stable performance measure of model with max performance and most stable performance
        """
        
        if "TabNet" in type(model).__name__:
            train_score = []
            test_score = []

            total_processing = self.cross_val_splits * len(param_range)
            pbar = tqdm(total=total_processing, desc="Pretraining " + str(type(model).__name__), unit="steps")

            for train_index, val_index in self.cv.split(X=self.X_train, y=self.y_train, groups=self.groups):

                # get Split
                X_train, X_val = self.X_train.iloc[train_index], self.X_train.iloc[val_index]

                try:
                    y_train, y_val = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                except AttributeError:
                    y_train, y_val = self.y_train[train_index], self.y_train[val_ind]

                train_score_tmp = []
                test_score_tmp = []

                print("Split Train: {} Split Val: {}".format(str(X_train.shape), str(X_val.shape)))

                for param in param_range:
                    model = clone(model).set_params(**{param_name: param})
                    if "TabNet" in type(model).__name__:
                        model.fit(
                            X_train=X_train.values,
                            y_train=y_train.values,
                            eval_set=[(X_train.values, y_train.values), (X_val.values, y_val.values)],
                            eval_name=['train', 'valid'],
                            eval_metric=['auc'],
                            max_epochs=self.optimization_iter,
                            patience=10,
                            batch_size=32,
                            virtual_batch_size=32,
                            num_workers=0,
                            weights=1,
                            drop_last=True
                        )
                        
                        # Calculate the averade AUC from all epochs
                        train_score_tmp.append(sum(model.history['train_auc']) / len(model.history['train_auc']))
                        test_score_tmp.append(sum(model.history['valid_auc']) / len(model.history['valid_auc']))
                        
                    else:
                        model = model.fit(X_train, y_train)
                        
                        y_pred_prob_train = model.predict_proba(X_train)[:, 1]
                        y_pred_prob_val = model.predict_proba(X_val)[:, 1]
                        
                        if self.task == "binary_classification":
                            train_score_tmp.append(roc_auc_score(y_train, y_pred_prob_train))
                            test_score_tmp.append(roc_auc_score(y_val, y_pred_prob_val))
                        elif self.task == "multi_class_classification":
                            train_score_tmp.append(roc_auc_score(y_train, y_pred_prob_train, multi_class='ovr'))
                            test_score_tmp.append(roc_auc_score(y_val, y_pred_prob_val, multi_class='ovr'))
                        else:
                            self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                            raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                        
                    # val_preds = model.predict_proba(X_val.values)
                    # val_auc = roc_auc_score(y_score=val_preds[:,1], y_true=y_val)

                    # train_preds = model.predict_proba(X_train.values)
                    # train_auc = roc_auc_score(y_score=train_preds[:,1], y_true=y_train)
                    
                    pbar.update(1)
                    
                train_score.append(train_score_tmp)
                test_score.append(test_score_tmp)

            pbar.close()
            x_max, y_max, x_stable, y_stable = PerformancePlotter(output_path=self.plot_save_dir + "/pretraining",
                                                                    X=self.X_train.values.copy(),
                                                                    y=self.y_train.values.copy(),
                                                                    y_pred=None,
                                                                    RunID=self.RunID,
                                                                    error=self.error,
                                                                    logger=self.logger).plot_hyperparameter_validation_curve(
                train_score_Num=np.array(train_score),
                test_score_Num=np.array(test_score),
                model=model,
                param_name=param_name,
                param_values=param_range)

        else:

            train_score, test_score = validation_curve(model,
                                                       X=self.X_train.copy(),
                                                       y=self.y_train.copy(),
                                                       param_name=param_name,
                                                       param_range=param_range,
                                                       scoring=self.scoring,
                                                       cv=self.cv,
                                                       groups=self.groups,
                                                       error_score="raise",
                                                       n_jobs=self.n_cpus)

            x_max, y_max, x_stable, y_stable = PerformancePlotter(output_path=self.plot_save_dir + "/pretraining",
                                                                  X=self.X_train.copy(),
                                                                  y=self.y_train.copy(),
                                                                  y_pred=None,
                                                                  RunID=self.RunID,
                                                                  error=self.error,
                                                                  logger=self.logger).plot_hyperparameter_validation_curve(
                train_score_Num=train_score,
                test_score_Num=test_score,
                model=model,
                param_name=param_name,
                param_values=param_range)

        return x_max, y_max, x_stable, y_stable

    def objective(self, trial: Trial):
        """
        Objective function for optimization
        :param trial:
        :return:
        """

        # Stable Parameter for optimization
        parameter = None

        # Exclude pretrained parameter from optimization seems to harm performance
        if len(self.pre_training_results) > 0:
            if type(self.tmp_model).__name__ in self.pre_training_results["Model"].values.tolist():
                parameter = self.pre_training_results.loc[
                    self.pre_training_results["Model"] == type(self.tmp_model).__name__, "Stable_Parameter_value"]

        # self.hyperparameter = {}
        self.params = {}
        if type(self.tmp_model).__name__ == "RandomForestClassifier":
            # self.params["eval_metric"] = "roc_auc"
            if self.extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 200, 800, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,
                    # Number of Trees for the forest
                    "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.001),
                    'max_depth': trial.suggest_int('max_depth', 2, 30, step=3),  # Max depth of a tree
                    "max_samples": trial.suggest_float('max_samples', 0.2, 0.8, step=0.01),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    'random_state': random.seed(self.rand_state)
                    # "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 100, step=3)
                }
            elif self.super_extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 200, 800, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,
                    # Number of Trees for the forest
                    "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.05, step=0.001),
                    'max_depth': trial.suggest_int('max_depth', 2, 30, step=3),  # Max depth of a tree
                    "max_samples": trial.suggest_float('max_samples', 0.2, 0.9, step=0.01),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 100, step=3),
                    "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.05, 0.3, step=0.01),
                    'random_state': random.seed(self.rand_state)
                }
            else:
                # MICCAI Settings
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 10, 700, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,  # Number of Trees for the forest
                    "ccp_alpha": trial.suggest_categorical("ccp_alpha", [0.001, 0.01, 0.02, 0.03]),
                    'max_depth': trial.suggest_categorical('max_depth', [None, 2, 4, 6, 8, 10]),  # Max depth of a tree
                    "max_samples": trial.suggest_categorical('max_samples', [0.2, 0.4, 0.6, 0.8,]),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    'random_state': random.seed(self.rand_state)
                }
        elif (type(self.tmp_model).__name__ == "SVC") or (type(self.tmp_model).__name__ == "LinearSVC"):
            if self.extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_float("C", 1.0, 3.0, step=0.1)
                else:
                    parameter = float(parameter)
                self.hyperparameter = {
                    # "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                    # "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                    # "dual": [dual],
                    "tol": trial.suggest_float("tol", 1e-6, 1e-2, step=1e-1),
                    "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                    "C": parameter,
                    'random_state': random.seed(self.rand_state)
                }
            else:
                if parameter is None:
                    parameter = trial.suggest_float("C", 1.0, 3.0, step=0.2)
                else:
                    parameter = float(parameter)
                self.hyperparameter = {
                    # "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                    # "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                    # "dual": [dual],
                    "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                    "C": parameter,
                    'random_state': random.seed(self.rand_state)
                }
        # TODO: solve problems with Lasso: AttributeError: Lasso has none of the following attributes: decision_function, predict_proba.
        # And it gives numbers instead of selected features for backward feature selection
        elif type(self.tmp_model).__name__ == "Lasso":
            if self.extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_float("alpha", 0.1, 10, step=0.4)
                else:
                    parameter = float(parameter)
                self.hyperparameter = {
                    "alpha": parameter,
                    "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                    "max_iter": trial.suggest_int("max_iter", 500, 1500, step=10),
                    "tol": trial.suggest_float("tol", 1e-10, 1e-1, step=1e-1),
                    'random_state': random.seed(self.rand_state)
                }
            else:
                if parameter is None:
                    parameter = trial.suggest_float("alpha", 0.1, 10, step=0.5)
                self.hyperparameter = {
                    "alpha": parameter,
                    "tol": trial.suggest_float("tol", 1e-5, 1e-1, step=1e-1),
                    'random_state': random.seed(self.rand_state)
                }
        elif type(self.tmp_model).__name__ == "GradientBoostingClassifier":
            # self.params["eval_metric"] = "roc_auc"
            if self.extended_parameter_set:
                hypergenerator = HyperparameterConfigurator(logger=self.logger, error=self.error, RunID=self.RunID)
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 200, 800, step=10)
                else:
                    parameter = int(parameter)
                if len(self.X_train_optimize.copy()) < 500: # if sample size too high the model training is hanging using this parameters
                    self.hyperparameter = {
                        "n_estimators": parameter,  # Number of Trees for the forest
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30, step=3),  # Max depth of a tree
                        "validation_fraction": trial.suggest_float('validation_fraction', 0.1, 0.4, step=0.01),
                        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 1, 100, step=5),
                        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes",
                                                                    hypergenerator.generate_max_leaf_nodes(self.X_train_optimize.copy())),
                        "warm_start": trial.suggest_categorical("warm_start", [False, True]),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001),
                        "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
                        "tol": trial.suggest_float("tol", 1e-4, 0.3, step=0.01),
                        "loss": trial.suggest_categorical("loss", ["log_loss", "exponential"]),
                        'random_state': random.seed(self.rand_state)
                    }

                else:
                    self.hyperparameter = {
                        "n_estimators": parameter,  # Number of Trees for the forest
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30, step=3),  # Max depth of a tree
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001),
                        "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
                        "tol": trial.suggest_float("tol", 1e-4, 0.3, step=0.01),
                        "loss": trial.suggest_categorical("loss", ["log_loss", "exponential"]),
                        'random_state': random.seed(self.rand_state)
                    }
            else:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 300, 600, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,  # Number of Trees for the forest
                    'max_depth': trial.suggest_int('max_depth', 2, 30, step=3),  # Max depth of a tree
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001),
                    'random_state': random.seed(self.rand_state)
                }
        elif type(self.tmp_model).__name__ == "LGBMClassifier":
            self.params = {"eval_metric": "auc"}

            if "cuda" in self.device:
                boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf'])
            else:
                boosting_type = trial.suggest_categorical('boosting_type', ['dart', 'rf'])

            if self.extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 200, 800, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,
                    'num_leaves': trial.suggest_int('num_leaves', 20, 70),  # max number of leaves in one tree
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                                                          int(self.X_train_optimize.copy().shape[0] * 0.1),
                                                          int(self.X_train_optimize.copy().shape[0] * 0.2)),
                    # min 10% or 20% of the data
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.025, step=0.001),
                    'max_bin': trial.suggest_int('max_bin', 40, 100, step=3),
                    # max number of bins that feature values will be bucketed in
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0, step=0.1),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.2, step=0.01),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.2, step=0.01),
                    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.4, step=0.01),
                    'boosting_type': boosting_type,
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 11, step=3),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.99, step=0.1),
                    'verbosity': -1,
                    'random_state': random.seed(self.rand_state)
                }
            else:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 300, 600, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,
                    'max_depth': trial.suggest_int('max_depth', 3, 15, step=3),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                                                          int(self.X_train_optimize.copy().shape[0] * 0.1),
                                                          int(self.X_train_optimize.copy().shape[0] * 0.2)),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 11, step=3),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.99, step=0.1),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.99, step=0.1),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, step=0.01),
                    'num_leaves': trial.suggest_int('num_leaves', 40, 80, step=5),
                    # 'early_stopping_round': 20,
                    'verbosity': -1,
                    'random_state': random.seed(self.rand_state)
                    # 'first_metric_only': True
                }
        elif type(self.tmp_model).__name__ == "XGBClassifier":
            # self.params["eval_metric"] = "roc_auc"
            if self.extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 200, 800, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,
                    'booster': trial.suggest_categorical('booster', ["gbtree", "gblinear"]),
                    # Select the type of model to run at each iteration
                    'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                    'gamma': trial.suggest_int('gamma', 0, 4, step=1),
                    'max_depth': trial.suggest_int('max_depth', 4, 8, step=1),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 3, step=1),
                    'max_delta_step': trial.suggest_int('max_delta_step', 0, 3, step=1),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.8, step=0.1),
                    # 'sampling_method':["uniform","gradient_based"],
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8, step=0.1),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.8, step=0.1),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.1),
                    'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                    'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                    'tree_method': trial.suggest_categorical('tree_method', ['auto', 'approx', 'hist']),
                    'random_state': random.seed(self.rand_state)
                }
            else:
                if parameter is None:
                    parameter = trial.suggest_int("n_estimators", 300, 600, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    "n_estimators": parameter,
                    'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                    'max_depth': trial.suggest_int('max_depth', 4, 8, step=1),
                    'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                    'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                    'random_state': random.seed(self.rand_state)
                }
        elif "TabNet" in type(self.tmp_model).__name__:
            if self.extended_parameter_set:
                if parameter is None:
                    parameter = trial.suggest_int('n_d', 100, 200, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    'n_d': parameter,
                    'n_a': trial.suggest_int('n_a', 2, 100, step=5),
                    'n_steps': trial.suggest_int('n_steps', 3, 10, step=1),  # Number of steps in the architecture
                    'gamma': trial.suggest_float('gamma', 1.0, 2.0, step=0.1),
                    # This is the coefficient for feature reusage in the masks.
                    'n_independent': trial.suggest_int('n_independent', 1, 4, step=1),
                    # Number of independent Gated Linear Units layers at each step.
                    'n_shared': trial.suggest_int('n_shared', 1, 4, step=1),
                    # Number of shared Gated Linear Units at each step
                    'momentum': trial.suggest_float('momentum', 0.02, 0.2, step=0.01),
                    'lambda_sparse': trial.suggest_float('lambda_sparse', 0.001, 0.1, step=0.0001),
                    'mask_type': trial.suggest_categorical('mask_type', ['entmax', "sparsemax"]),
                    'scheduler_params': dict(is_batch_level=True,
                                             max_lr=5e-2,
                                             steps_per_epoch=int(self.X_train_optimize.copy().shape[0] / 32),
                                             epochs=self.optimization_iter),
                    'scheduler_fn': torch.optim.lr_scheduler.OneCycleLR,
                    'seed': int(self.rand_state)
                }
            else:
                if parameter is None:
                    parameter = trial.suggest_int('n_d', 100, 200, step=10)
                else:
                    parameter = int(parameter)
                self.hyperparameter = {
                    'n_d': parameter,
                    'n_a': trial.suggest_int('n_a', 2, 100, step=5),
                    'mask_type': trial.suggest_categorical('mask_type', ['entmax', "sparsemax"]),
                    'seed': int(self.rand_state)
                }

        elif type(self.tmp_model).__name__ == "SelectKBest":

            if self.extended_parameter_set:
                self.hyperparameter = {
                    "score_func": trial.suggest_categorical("score_func", ["f_classif", "SelectFdr", "SelectFpr"]),
                    "k": trial.suggest_categorical("k", ["all", 10, self.n_features]),
                    'random_state': random.seed(self.rand_state)
                }
            else:
                self.hyperparameter = {
                    "score_func": trial.suggest_categorical("score_func", ["f_classif", "SelectFdr"]),
                    "k": trial.suggest_categorical("k", [10, self.n_features]),
                    'random_state': random.seed(self.rand_state)
                }
        else:
            warnings.warn(
                f"Hyperparameter optimization for {type(self.tmp_model).__name__} not implemented!"
            )

        # model = self.tmp_model.set_params(**self.hyperparameter)
        model = clone(self.tmp_model).set_params(**self.hyperparameter)
        if not len(self.X_train_optimize.copy()) > 0:
            self.error.warning("Data for optimizing is empty!")
            print("Warning Data is empty!")

        if not len(self.y_train_optimize.copy()) > 0:
            self.error.warning("Data for optimizing is empty!")
            print("Warning Label is empty!")

        if "TabNet" in type(model).__name__:
            model.fit(
                X_train=self.X_train_optimize.copy().values,
                y_train=self.y_train_optimize.copy().values,
                eval_set=[(self.X_train_optimize.copy().values, self.y_train_optimize.values),
                          (self.X_val_optimize.copy().values, self.y_val_optimize.copy().values)],
                eval_name=['train', 'valid'],
                eval_metric=['auc'],
                max_epochs=self.optimization_iter,
                patience=10,
                drop_last=True,
                batch_size=32,
                virtual_batch_size=32,
                num_workers=0,
                weights=1,
            )
        else:
            model = model.fit(self.X_train_optimize.copy(), self.y_train_optimize.copy())

        y_pred_prob = None
        y_pred = None

        if "TabNet" in type(model).__name__:
            y_pred = model.predict(self.X_val_optimize.values)
            y_pred_prob = model.predict_proba(self.X_val_optimize.values)[:, 1]

        elif "Ensemble" in str(type(model).__name__):
            # Ensemble optimization normally not perfromed
            if len(model.clfs) > 1:
                if self.check_tabnet_in_ensemble(model):
                    y_pred = model.predict(self.X_val_optimize.values)
                    y_pred_prob = model.predict_proba(self.X_val_optimize.values)[:, 1]
                else:
                    y_pred = model.predict(self.X_val_optimize)
                    y_pred_prob = model.predict_proba(self.X_val_optimize)[:, 1]
            else:
                print("No models in Ensemble for optimization!")
                self.error.warning("No models in Ensemble for optimization!")


        else:
            y_pred = model.predict(self.X_val_optimize)
            y_pred_prob = model.predict_proba(self.X_val_optimize)[:, 1]

        auroc = 0.0
        f1 = 0.0
        bal_acc = 0.0

        if y_pred_prob is not None:
            if self.task == "binary_classification":
                auroc = roc_auc_score(self.y_val_optimize, y_pred_prob)
            elif self.task == "multi_class_classification":
                auroc = roc_auc_score(self.y_val_optimize, y_pred_prob, multi_class='ovr')
            else:
                self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
        else:
            print("Can not calculate AUC without predictions! Error coming from fitting model.")
            self.error.warning("Can not calculate AUC without predictions! Error coming from fitting model.")

        if y_pred is not None:
            f1 = f1_score(self.y_val_optimize, y_pred)
            bal_acc = balanced_accuracy_score(self.y_val_optimize, y_pred)
        else:
            print("Can not calculate F1 and Accuracy without predictions! Error coming from fitting model.")
            self.error.warning(
                "Can not calculate F1 and Accuracy without predictions! Error coming from fitting model.")

        if self.use_wandb:
            if self.fold is None:
                wandb.log({"Optimized_Val_AUROC": round(auroc, 4)})
                wandb.log({"Optimized_Val_F1": round(f1, 4)})
                wandb.log({"Optimized_Val_Bal_ACC": round(bal_acc, 4)})
            else:
                wandb.log({f"Optimized_Val_AUROC{self.fold}": round(auroc, 4)})
                wandb.log({f"Optimized_Val_F1{self.fold}": round(f1, 4)})
                wandb.log({f"Optimized_Val_Bal_ACC{self.fold}": round(bal_acc, 4)})

        if self.run_neptune:
            self.neptune_run["Optimized_Val_AUROC"].append(round(auroc, 4), step=trial.number)  # = round(auroc, 4)
            self.neptune_run["Optimized_Val_F1"].append(round(f1, 4), step=trial.number)  # = round(f1, 4)
            self.neptune_run["Optimized_Val_Bal_ACC"].append(round(bal_acc, 4),
                                                             step=trial.number)  # = round(bal_acc, 4)

        return auroc, f1  # , bal_acc

    def apply_optimizer(self, X_train, y_train, X_val=None, y_val=None, model=None, model_name=None):
        """
        Apply Optuna or the Grid Search optimizer to the model.
        :param X_train: Data for training
        :param y_train: Label for training
        :param X_val: Data for validation
        :param y_val: Label for validation
        :param model: Model to optimize
        :param model_name: name of model
        :return: Best model with Hyperparameters
        """

        auroc_train = []
        auroc_val = []
        auroc_test = []

        if model is None:
            model = self.model
        if model_name is None:
            model_name = type(model).__name__

        if (X_val is None) and (y_val is None):

            if X_train.index.duplicated().sum() > 0:

                train_idx, val_idx = self.get_unique_index(X=X_train.copy(), y=y_train.copy(),
                                                           random_state=random.seed(self.rand_state))

                # convert to df
                y_train_df = y_train.reset_index()
                y_train_df = y_train_df.set_index("ID")

                X_train_ = X_train[X_train.index.isin(train_idx)]
                y_train = y_train_df[y_train_df.index.isin(train_idx)][self.Prediction_Label]

                X_val = X_train[X_train.index.isin(val_idx)]
                y_val = y_train_df[y_train_df.index.isin(val_idx)][self.Prediction_Label]

                X_train = X_train_
                del X_train_

            else:
                X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                                  y_train,
                                                                  random_state=random.seed(self.rand_state),
                                                                  shuffle=True,
                                                                  test_size=self.test_size)

        self.X_train_optimize = X_train
        self.y_train_optimize = y_train

        self.X_val_optimize = X_val
        self.y_val_optimize = y_val

        neptune_callback = None

        self.logger.info("Executing Optuna Optimizing with " + str(self.optimization_iter) + " Iterations")

        if self.use_optuna:

            # TODO: Not working with sqlite yet!
            # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            study_name = "RPTK_optimizing_" + str(type(model).__name__)  # Unique identifier of the study.
            # storage_name = "sqlite:///" + self.out_folder + "/{}.db".format(study_name)

            # URL = "sqlite:///" + self.out_folder + "/RPTK_optimizing_" + str(type(model).__name__) + "_db.sqlite3"

            if self.run_neptune:
                self.neptune_run = neptune.init_run(
                    custom_run_id=model_name,
                    name=self.neptune_run_name,
                    project=self.neptune_project,
                    api_token=self.neptune_api_token,
                )
            else:
                self.neptune_run = None

            if self.neptune_run is not None:
                neptune_callback = optuna_utils.NeptuneCallback(self.neptune_run)
            else:
                if self.run_neptune:
                    print("Something did not work on the neptune study generation!")

            # storage = optuna.storages.RDBStorage(url=URL)
            self.tmp_model = clone(model)
            study = optuna.create_study(directions=["maximize", "maximize"],  # , "maximize"],
                                        sampler=optuna.samplers.TPESampler(seed=int(self.rand_state)),
                                        # storage=storage_name,  # Specify the storage URL here.
                                        study_name=study_name,
                                        pruner=optuna.pruners.HyperbandPruner()
                                        )
            if self.neptune_run is not None:
                study.optimize(self.objective,
                               n_jobs=self.n_cpus,
                               n_trials=self.optimization_iter,
                               gc_after_trial=True,
                               catch=(AttributeError, IndexError, KeyError),
                               show_progress_bar=True,
                               callbacks=[neptune_callback]
                               )
            else:
                study.optimize(self.objective,
                               n_jobs=self.n_cpus,
                               n_trials=self.optimization_iter,
                               gc_after_trial=True,
                               catch=(AttributeError, IndexError),
                               show_progress_bar=True)

            simple_model_name = self.get_simple_model_name(model_name)

            optimization_log_out_dir = ""

            if simple_model_name != "":
                if not os.path.exists(self.out_folder + "/optimization_log/" + simple_model_name + "/"):
                    os.makedirs(self.out_folder + "/optimization_log/" + simple_model_name + "/",exist_ok=True)
                optimization_log_out_dir = self.out_folder + "/optimization_log/" + simple_model_name + "/"

            else:
                if not os.path.exists(self.out_folder + "/optimization_log/" + model_name + "/"):
                    os.makedirs(self.out_folder + "/optimization_log/" + model_name + "/",exist_ok=True)
                optimization_log_out_dir = self.out_folder + "/optimization_log/" + model_name + "/"


            if optimization_log_out_dir != "":
                study.trials_dataframe().to_csv(
                    optimization_log_out_dir + model_name + "_optimization_summary.csv")

            else:
                os.makedirs(self.out_folder + "/optimization_log/" + model_name + "/",exist_ok=True)
                study.trials_dataframe().to_csv(
                    self.out_folder + "/optimization_log/" + model_name + "/" + model_name + "_optimization_summary.csv")

            # Calculate the percentage of each state in the 'state' column
            state_counts = study.trials_dataframe()['state'].value_counts(normalize=True) * 100

            print("#### Evaluate optimization")
            self.logger.info("#### Evaluate optimization")

            for state, percentage in state_counts.items():
                print(f"State: {state}, Percentage: {percentage:.2f}%")
                self.logger.info(f"State: {state}, Percentage: {percentage:.2f}%")

            print("Optimization Done!")
            self.logger.info(f"Number of best trials: {len(study.best_trials)}")
            trial_with_highest_auc = max(study.best_trials, key=lambda t: t.values[0])
            self.logger.info(f"Highest Val AUC: {trial_with_highest_auc.values}")
            print(f"Highest Val AUC: {trial_with_highest_auc.values}")

            trial_with_highest_f1 = max(study.best_trials, key=lambda t: t.values[1])
            self.logger.info(f"Highest Val F1: {trial_with_highest_f1.values}")
            print(f"Highest Val F1: {trial_with_highest_f1.values}")

            best_params = trial_with_highest_auc.params

            if simple_model_name != "":
                if not os.path.exists(self.plot_save_dir + "/optimization/" + simple_model_name + "/"):
                    os.makedirs(self.plot_save_dir + "/optimization/" + simple_model_name + "/")
            else:
                if not os.path.exists(self.plot_save_dir + "/optimization/" + model_name + "/"):
                    os.makedirs(self.plot_save_dir + "/optimization/" + model_name + "/")

            try:
                fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0],
                                                                target_name="AUROC")

                if simple_model_name != "":
                    fig.write_image(self.plot_save_dir + "/optimization/" + simple_model_name + "/" + model_name +
                                "_" + self.RunID + "_hyperparameter_importance.png")
                else:
                    fig.write_image(self.plot_save_dir + "/optimization/" + model_name + "/" + study_name +
                                "_" + self.RunID + "_hyperparameter_importance.png")
            except Exception as ex:
                self.error.warning("Could not plot hypereparameter importance: " + str(ex))
                print("Could not plot hypereparameter importance: " + str(ex))
        else:
            try:
                all_available_hyperparameter = HyperparameterConfigurator(model=model,
                                                                        optimizer_lib=None,
                                                                        logger=None,
                                                                        error=None,
                                                                        out_path="",
                                                                        interested_hyperparameter=None,
                                                                        extended_parameter_set=None)

                Hypergenerator = GridHyperParameterGenerator(
                                                            model=model,
                                                            x=self.X_train.copy(),
                                                            y=self.y_train.copy(),
                                                            use_optuna=self.use_optuna,
                                                            extended_parameter_set=self.extended_parameter_set)

                self.hyperparameter = Hypergenerator.generate_hyperparameter_set()

                clf = GridSearchCV(estimator=model,
                                    param_grid=self.hyperparameter,
                                    cv=self.num_splits,
                                    scoring=self.scoring,
                                    n_jobs=self.n_cpus)

            except Exception as ex:
                    self.error.warning("Could not plot hypereparameter importance: " + str(ex))
                    print("Could not plot hypereparameter importance: " + str(ex))

            if "TabNet" in type(model).__name__:
                clf.fit(self.X_train.values, self.y_train.values)
            else:
                clf = clf.fit(self.X_train, self.y_train)

            best_params = clf.best_params_

        # loader.stop()

        best_model = model.set_params(**best_params)

        if len(self.params) > 0:
            if "TabNet" in type(model).__name__:
                best_model.fit(
                    X_train=X_train.values,
                    y_train=y_train.values,
                    eval_set=[(X_train.values, y_train.values),
                              (X_val.values, y_val.values)],
                    eval_name=['train', 'valid'],
                    eval_metric=['auc'],
                    max_epochs=self.optimization_iter,
                    patience=10,
                    batch_size=32,
                    virtual_batch_size=32,
                    num_workers=self.n_cpus,
                    weights=1,
                    drop_last=True,
                    **self.params
                )
            else:
                best_model = best_model.fit(X_train, y_train,
                                            # eval_set=[(self.X_val, self.y_val)],
                                            **self.params)
        else:
            if "TabNet" in type(model).__name__:
                best_model.fit(
                                X_train=X_train.values,
                                y_train=y_train.values,
                                eval_set=[(X_train.values, y_train.values),
                                        (X_val.values, y_val.values)],
                                eval_name=['train', 'valid'],
                                eval_metric=['auc'],
                                max_epochs=self.optimization_iter,
                                patience=10,
                                batch_size=32,
                                virtual_batch_size=32,
                                num_workers=self.n_cpus,
                                weights=1,
                                drop_last=True
                            )
            else:
                best_model = best_model.fit(X_train, y_train)

        if self.neptune_run is not None:
            self.neptune_run.stop()

        y_pred_prob = None
        y_pred_prob_val = None
        y_pred_prob_test = None

        if "TabNet" in type(model).__name__:
            # Check for overfitting models
            y_pred_prob = model.predict_proba(X_train.values)[:, 1]
            y_pred_prob_val = model.predict_proba(X_val.values)[:, 1]
            y_pred_prob_test = model.predict_proba(self.X_test.values)[:, 1]

        elif "Ensemble" in str(type(model).__name__):
            if len(model.clfs) > 1:
                if self.check_tabnet_in_ensemble(model):
                    # Check for overfitting models
                    y_pred_prob = model.predict_proba(X_train.values)[:, 1]
                    y_pred_prob_val = model.predict_proba(X_val.values)[:, 1]
                    y_pred_prob_test = model.predict_proba(self.X_test.values)[:, 1]
                else:
                    # Check for overfitting models
                    y_pred_prob = model.predict_proba(X_train)[:, 1]
                    y_pred_prob_val = model.predict_proba(X_val)[:, 1]
                    y_pred_prob_test = model.predict_proba(self.X_test)[:, 1]

        else:
            # Check for overfitting models
            y_pred_prob = model.predict_proba(X_train)[:, 1]
            y_pred_prob_val = model.predict_proba(X_val)[:, 1]
            y_pred_prob_test = model.predict_proba(self.X_test)[:, 1]

        if y_pred_prob is not None:
            if self.task == "binary_classification":
                auroc_train.append(roc_auc_score(y_train, y_pred_prob))
            elif self.task == "multi_class_classification":
                auroc_train.append(roc_auc_score(y_train, y_pred_prob, multi_class='ovr'))
            else:
                self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
        else:
            print("Can not calculate train AUC without predictions! Error coming from fitting model.")
            self.error.warning("Can not calculate train AUC without predictions! Error coming from fitting model.")

        if y_pred_prob_val is not None:
            if self.task == "binary_classification":
                auroc_val.append(roc_auc_score(y_val, y_pred_prob_val))
            elif self.task == "multi_class_classification":
                auroc_val.append(roc_auc_score(y_val, y_pred_prob_val, multi_class='ovr'))
            else:
                self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
        else:
            print("Can not calculate validation AUC without predictions! Error coming from fitting model.")
            self.error.warning("Can not calculate validation AUC without predictions! Error coming from fitting model.")

        if y_pred_prob_test is not None:
            if self.task == "binary_classification":
                auroc_test.append(roc_auc_score(self.y_test, y_pred_prob_test))
            elif self.task == "multi_class_classification":
                auroc_test.append(roc_auc_score(self.y_test, y_pred_prob_test, multi_class='ovr'))
            else:
                self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
        else:
            print("Can not calculate test AUC without predictions! Error coming from fitting model.")
            self.error.warning("Can not calculate test AUC without predictions! Error coming from fitting model.")

        # get feature importances
        feature_importances_dict = {}
        feature_names = self.X_train.columns

        # Check the model type and extract feature importances accordingly
        if type(model).__name__ == "RandomForestClassifier":
            feature_importances = model.feature_importances_
        elif type(model).__name__ == "TabNetClassifier":  # Assuming TabNetClassifier is the class name
            feature_importances = model.feature_importances_
        elif type(model).__name__ == "GradientBoostingClassifier":
            feature_importances = model.feature_importances_
        elif type(model).__name__ == "LGBMClassifier":  # LightGBM model class name
            feature_importances = model.feature_importances_
        elif type(model).__name__ == "XGBClassifier":  # XGBoost model class name
            feature_importances = model.feature_importances_
        elif type(model).__name__ == "SVC":  # SVM model class name
            if hasattr(model, 'coef_'):
                feature_importances = model.coef_[0]
            else:
                try:
                    feature_importances = permutation_importance(model, self.X_train, self.y_train)
                except:
                    feature_importances = None  # SVM with non-linear kernels won't have feature importances
        else:
            raise ValueError(f"Unsupported model type: {type(model).__name__}")

        feature_importances_dict[type(model).__name__] = feature_importances

        # Combine feature importances into a DataFrame
        feature_importances_df = pd.DataFrame(feature_importances_dict, index=feature_names)

        # Save to CSV
        # Define the main output folder
        feature_importance_folder = os.path.join(self.out_folder, "feature_importance")

        # Check if the main folder exists, if not, create it
        if not os.path.exists(feature_importance_folder):
            os.makedirs(feature_importance_folder)

        simple_model_name = self.get_simple_model_name(model_name)

        if simple_model_name != "":
            if not os.path.exists(feature_importance_folder + "/" + simple_model_name):
                os.makedirs(feature_importance_folder + "/" + simple_model_name)

            feature_importance_output_path = feature_importance_folder + "/" + simple_model_name + "/" + model_name + "_feature_importances.csv"
        else:
            feature_importance_output_path = feature_importance_folder + "/"+ model_name + "_feature_importances.csv"

        feature_importances_df.to_csv(feature_importance_output_path, index=True)

        # Output the feature importances for debugging or further processing

        return best_model, auroc_train, auroc_val, auroc_test

    def get_best_model(self, X, y, model, out_dir, X_val=None, y_val=None, filename=None, model_name=None):
        """
        Get the best-performing model from the optimization process.
        """

        train_auc = 0.0
        val_auc = 0.0
        test_auc = 0.0
        best_model = None

        if out_dir.endswith("/"):
            out_dir = out_dir[:-1]

        if model_name is None:
            model_name = type(model).__name__

        if filename is None:
            filename = out_dir + "/" + self.RunID + "_optimized_" + type(model).__name__ + '.sav'
        else:
            if out_dir not in filename:
                filename_ = out_dir + "/" + filename
                filename = filename_
                del filename_

        # best_models = []

        # check if model has been optimized already
        if os.path.exists(filename):
            self.logger.info(
                "Found optimized model at " + filename + "! Loading model " + type(model).__name__ + " ...")
            print("Found trained " + type(model).__name__ + ". Loading ...")
            best_model = self.load_model(filename)

        else:
            best_model, train_auc, val_auc, test_auc = self.apply_optimizer(X_train=X,
                                                                            y_train=y,
                                                                            X_val=X_val,
                                                                            y_val=y_val,
                                                                            model=model,
                                                                            model_name=model_name)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            if best_model is not None:
                self.save_model(model=best_model, file_path=filename)

        return best_model, train_auc, val_auc, test_auc

    def extended_eval(self, y_true, X=pd.DataFrame(), y_pred_prob=None, model="", youden_=0):
        """
        Make extended validation to get optimal parameters in the highest point of the AUC using Youden Index calculations
        :param y_true: true label
        :param X: Data wihtout label
        :param y_pred_prob: prediction probabilities
        :param model: model instance which gets evaluated
        :param youden_: Youden Index for optimizing based on the AUROC
        :return:
        """

        if y_pred_prob is None:
            y_pred_prob = []

        if (len(y_pred_prob) == 0) and (model != ""):

            if type(model).__name__ == "Lasso":
                # No predict_proba available for Lasso
                y_pred_prob = model.predict(X)
            elif type(model).__name__ == "KNeighborsClassifier":
                y_pred_prob = model.predict_proba(X.values)[:, 1]
            elif "TabNet" in type(model).__name__:
                y_pred_prob = model.predict_proba(X.values)[:, 1]
            elif "Ensemble" in str(type(model).__name__):
                if len(model.clfs) > 1:
                    if self.check_tabnet_in_ensemble(model):
                        y_pred_prob = model.predict_proba(X.values)[:, 1]
            else:
                y_pred_prob = model.predict_proba(X)[:, 1]

            if type(model).__name__ == "KNeighborsClassifier":
                # No predict_proba available for Lasso
                y_pred = model.predict(X.values)
            elif "TabNet" in type(model).__name__:
                y_pred = model.predict(X.values)[:, 1]
            elif "Ensemble" in str(type(model).__name__):
                if len(model.clfs) > 1:
                    if self.check_tabnet_in_ensemble(model):
                        y_pred = model.predict(X.values)[:, 1]
            else:
                y_pred = model.predict(X)

        elif (model == "") and (len(X) == 0):
            corr_y_pred = []
            if youden_ == 0:
                for i in y_pred_prob:
                    if i < 0.5:
                        corr_y_pred.append(0)
                    elif i >= 0.5:
                        corr_y_pred.append(1)
            else:
                # check if youden correction would lead to class imbalance
                # if all(i >= youden_ for i in y_pred_prob) or all(i < 30 for youden_ in y_pred_prob):
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
                specificity = 1 - fpr
                youden_index = tpr + specificity - 1

                best_idx = np.argmax(youden_index)
                best_threshold = thresholds[best_idx]
                best_j = youden_index[best_idx]
                best_fpr = fpr[best_idx]
                best_tpr = tpr[best_idx]

                for i in y_pred_prob:
                    if i < best_threshold:
                        corr_y_pred.append(0)
                    elif i >= best_threshold:
                        corr_y_pred.append(1)

            y_pred = corr_y_pred
            # y_pred = round(y_pred_prob)
        else:
            self.error.error("Can not evaluate without prediction! Please correct")
            raise ValueError("Can not evaluate without prediction! Please correct")

        if type(y_true) == pd.Series:
            y_true = y_true.values

        sample_statistic, lower, upper, auc_mean, auc_std = self.bootstrap_auc(y_true, y_pred_prob)
        ap = average_precision_score(y_true, y_pred)

        if self.task == "binary_classification":
            auroc = roc_auc_score(y_true, y_pred_prob)
        elif self.task == "multi_class_classification":
            auroc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
        else:
            self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
            raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")

        
        bal_acc = balanced_accuracy_score(y_true, y_pred)  # , adjusted=True)

        sens_sample_statistic, sens_lower, sens_upper, sens_mean, sens_std = self.bootstrap_sensitivity(y_true, y_pred)
        sp_sample_statistic, sp_lower, sp_upper, sp_mean, sp_std = self.bootstrap_specificity(y_true, y_pred)
        f1_sample_statistic, f1_lower, f1_upper, f1_mean, f1_std = self.bootstrap_f1(y_true, y_pred)
        # pos_lh = class_likelihood_ratios(y_true, y_pred)[0]
        f1 = f1_score(y_true, y_pred)

        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        specificity = 1 - fpr
        youden_index = tpr + specificity - 1

        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_j = youden_index[best_idx]
        best_fpr = fpr[best_idx]
        best_tpr = tpr[best_idx]

        youden = best_j

        print("AUROC:", round(auroc, 3), 
                "\nF1:", round(f1, 3),
                "\nBalanced Accuracy:", round(bal_acc, 3),
                "\nAverage Precision", round(ap, 3),
                "\nBootstrap AUC:", round(sample_statistic, 3),
                "\nBootstrap F1:", round(f1_sample_statistic, 3),
                "\nBootstrap Sensitifity", round(sens_sample_statistic, 3),
                "\nBootstrap Specificity", round(sp_sample_statistic, 3),
                "\nYouden Index:", str(round(youden, 2)))

        if youden != 0:
            corr_y_pred = []
            for i in y_pred_prob:
                if i < best_threshold:
                    corr_y_pred.append(0)
                elif i >= best_threshold:
                    corr_y_pred.append(1)

            y_pred = corr_y_pred

            sample_statistic_, lower_, upper_, auc_mean, auc_std = self.bootstrap_auc(y_true, y_pred)
            ap_ = average_precision_score(y_true, y_pred)

            if self.task == "binary_classification":
                auroc_ = roc_auc_score(y_true, y_pred)
            elif self.task == "multi_class_classification":
                auroc_ = roc_auc_score(y_true, y_pred, multi_class='ovr')
            else:
                self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")

            bal_acc_ = balanced_accuracy_score(y_true, y_pred)  # , adjusted=True)

            sens_sample_statistic_, sens_lower, sens_upper, sens_mean, sens_std = self.bootstrap_sensitivity(y_true,
                                                                                                             y_pred)
            sp_sample_statistic_, sp_lower_, sp_upper_, sp_mean, sp_std = self.bootstrap_specificity(y_true, y_pred)
            f1_sample_statistic_, f1_lower_, f1_upper_, f1_mean, f1_std = self.bootstrap_f1(y_true, y_pred)
            f1_ = f1_score(y_true, y_pred)

            print("After Youden Correction:")
            print("AUROC:", round(auroc_, 3), 
                  "\nF1:", round(f1_, 3),
                  "\nBalanced Accuracy:", round(bal_acc_, 3),
                  "\nAverage Precision", round(ap_, 3),
                  "\nBootstrap AUC:", round(auc_mean, 3),
                  "\nBootstrap F1:", round(f1_mean, 3),
                  "\nBootstrap Sensitivity", round(sens_mean, 3),
                  "\nBootstrap Specificity", round(sp_mean, 3)
                  )

        data = {"Model": [type(model).__name__],
                "AUC": [round(auc_mean, 3)],
                "AUC_lower": [round(lower, 3)],
                "AUC_upper": [round(upper, 3)],
                "F1": [round(f1_mean, 3)],
                "F1_lower": [round(f1_lower, 3)],
                "F1_upper": [round(f1_upper, 3)],
                "Average_Precision": [round(ap, 3)],
                "sens_sample": [round(sens_mean, 3)],
                "sens_lower": [round(sens_lower, 3)],
                "sens_upper": [round(sens_upper, 3)],
                "sp_sample": [round(sp_mean, 3)],
                "sp_lower": [round(sp_lower, 3)],
                "sp_upper": [round(sp_upper, 3)],
                "Youden_Index": [round(youden, 2)]}

        results = pd.DataFrame(data, index=[random.randint(1, 100)])

        return results

    def sensitivity(self, y_true, y_pred):
        """
        Calculate Sensitivity
        :param y_true: label
        :param y_pred: prediction prob
        :return: sensitivity
        """

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        Sensitivity = tp / (tp + fn)

        return Sensitivity

    def specificity(self, y_true, y_pred):
        """
        Calculate Specificity
        :param y_true: label
        :param y_pred: prediction prob
        :return: specificity
        """

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        return specificity

    def bootstrap_sensitivity(self, y_true, y_pred):
        """
        Get Sensitivity and 95% Confidence Interval from bootstrapping.
        :param y_true: label
        :param y_pred: prediction prob
        :return sample_statistic: generated values
        :return lower: lower bound of 95% CI
        :return upper: upper bound of 95% CI
        :retrun mean: mean value
        :return std: standard deviation
        """
        print("Calculate Sensitivity with 95% CI ... ")
        self.logger.info("Calculate Sensitivity with 95% CI ... ")

        sample_statistic, lower, upper, mean, std = self.bootstrap_statistic(
            self.sensitivity,
            y_true,
            y_pred,
            "sensitivity"
        )

        return sample_statistic, lower, upper, mean, std

    def bootstrap_auc(self, y_true, y_pred):
        """
        Get AUC and 95% Confidence Interval from bootstrapping.
        :param y_true: label
        :param y_pred: prediction prob
        :return sample_statistic: generated values
        :return lower: lower bound of 95% CI
        :return upper: upper bound of 95% CI
        :retrun mean: mean value
        :return std: standard deviation
        """

        print("Calculate AUROC with 95% CI ... ")
        self.logger.info("Calculate AUROC with 95% CI ... ")

        if self.task == "binary_classification":
            sample_statistic, lower, upper, mean, std = self.bootstrap_statistic(
            roc_auc_score,
            y_true,
            y_pred,
            "auroc"
            )

        elif self.task == "multi_class_classification":
            sample_statistic, lower, upper, mean, std = self.bootstrap_statistic(
            statistic=roc_auc_score,
            x=y_true,
            y=y_pred,
            metric="auroc",
            kwargs={'multi_class':'ovr'}
            )
        
        return sample_statistic, lower, upper, mean, std

    def bootstrap_specificity(self, y_true, y_pred):
        """
        Get Specificity and 95% Confidence Interval from bootstrapping.
        :param y_true: label
        :param y_pred: prediction prob
        :return sample_statistic: generated values
        :return lower: lower bound of 95% CI
        :return upper: upper bound of 95% CI
        :retrun mean: mean value
        :return std: standard deviation
        """
        print("Calculate Specificity with 95% CI ... ")
        self.logger.info("Calculate Specificity with 95% CI ... ")

        sample_statistic, lower, upper, mean, std = self.bootstrap_statistic(
            self.specificity,
            y_true,
            y_pred,
            "specificity"
        )

        return sample_statistic, lower, upper, mean, std

    def bootstrap_f1(self, y_true, y_pred):
        """
        Get F1 and 95% Confidence Interval from bootstrapping.
        :param y_true: label
        :param y_pred: prediction prob
        :return sample_statistic: generated values
        :return lower: lower bound of 95% CI
        :return upper: upper bound of 95% CI
        :retrun mean: mean value
        :return std: standard deviation
        """
        print("Calculate F1 Score with 95% CI ... ")
        self.logger.info("Calculate F1 Score with 95% CI ... ")

        sample_statistic, lower, upper, mean, std = self.bootstrap_statistic(
            f1_score,
            y_true,
            y_pred,
            "f1"
        )

        return sample_statistic, lower, upper, mean, std

    def bootstrap_statistic(self, statistic: Callable, x, y, metric, kwargs=None, num_folds=1000, ci_percentile=95):
        """
        Bootstrap statistic for comparing two groups.
        :param statistic: function that takes two lists of values and returns a statistic.
        :param x: list of values for group 1
        :param y: list of values for group 2
        :param num_folds: number of bootstrap samples to draw
        :param kwargs: additional arguments to pass to statistic as dict
        :return statistic: sample statistic for the two groups
        :return lower_bound: lower bound of the 95% confidence interval
        :return upper_bound: upper bound of the 95% confidence interval
        """
        stats = []
        random_state = 0
        #pbar = tqdm(total = num_folds,  desc="Bootstrapping")

        while len(stats) < num_folds:
            boot_x, boot_y = resample(
                x, y, replace=True, n_samples=len(x), random_state=random_state
            )
            random_state += 1

            if kwargs is None:
                # Try to calculate statistics - it can happen that there is only one class in the bootstrapped sample - calculatation failes
                try:
                    stat = statistic(boot_x, boot_y)
                except:
                    stat = None
            else:
                try:
                    stat = statistic(boot_x, boot_y, **kwargs)
                except:
                    stat = None

            if stat is not None:
                stats.append(stat)
                #pbar.update(1)

        #pbar.close()
        stats_arr = np.array(stats)

        if kwargs is None:
            sample_statistic = statistic(x, y)
        else:
            sample_statistic = statistic(x, y, **kwargs)

        mean_boot = statistics.mean(stats_arr)
        std_boot = statistics.stdev(stats_arr)
        lower_bound = np.percentile(stats_arr, (100 - ci_percentile) / 2)
        upper_bound = np.percentile(stats_arr, 100 - (100 - ci_percentile) / 2)

        return sample_statistic, lower_bound, upper_bound, mean_boot, std_boot

    def cv_AUC_generator(self, cv_models_trained=None, X_vals=None, y_vals=None, model_name=None, plotter=None, youden_annotation = False, val_preds=None, save_plot=True):
        """
        Generating the AUC plot for cross validation models
        :param cv_models_trained: list of all models trained on every fold
        :param X_vals: list of data for every fold
        :param y_vals: list of labels for every fold
        :param model_name: name of the model
        :param plotter: PerformancePlotter object for generating plots
        """

        tprs = []
        aucs = []
        best_js = []
        best_fprs = []
        best_tprs = []
        best_thresholds = []

        if model_name is None:
            model_name = ""

        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(10,10))

        fold = 0

        # make predictions with models directly
        if val_preds is None:
            # Calculate values for every fold
            for model, X_val, y_val in zip(cv_models_trained, X_vals, y_vals):
                if "TabNet" in type(model).__name__:
                    # Predict
                    y_pred = model.predict_proba(X_val.values)
                elif "Ensemble" in str(type(model).__name__):
                    if len(model.clfs) > 1:
                        if self.check_tabnet_in_ensemble(model):
                            y_pred = model.predict_proba(X_val.values)
                else:
                    # Predict
                    y_pred = model.predict_proba(X_val)

                # plot fold roc 
                viz = plotter.plot_cv_AUC(y=y_val, y_pred=y_pred[:, 1], fold=fold, ax=ax)

                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

                fpr, tpr, thresholds = roc_curve(y_val, y_pred[:, 1])
                specificity = 1 - fpr
                youden_index = tpr + specificity - 1

                best_idx = np.argmax(youden_index)
                best_thresholds.append(thresholds[best_idx])
                best_js.append(youden_index[best_idx])
                best_fprs.append(fpr[best_idx])
                best_tprs.append(tpr[best_idx])

                fold += 1
        else:
            # use performed predictions 
            for y_val, y_pred in zip(y_vals, y_preds):

                # plot fold roc 
                viz = plotter.plot_cv_AUC(y=y_val, y_pred=y_pred[:, 1], fold=fold, ax=ax)

                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

                fpr, tpr, thresholds = roc_curve(y_val, y_pred[:, 1])
                specificity = 1 - fpr
                youden_index = tpr + specificity - 1

                best_idx = np.argmax(youden_index)
                best_thresholds.append(thresholds[best_idx])
                best_js.append(youden_index[best_idx])
                best_fprs.append(fpr[best_idx])
                best_tprs.append(tpr[best_idx])

                fold += 1

        # Calculate values for mean
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        mean_best_fpr = np.mean(best_fprs, axis=0)
        mean_best_tpr = np.mean(best_tprs, axis=0)
        mean_best_j = np.mean(best_js, axis=0)
        mean_best_threshold = np.mean(best_thresholds, axis=0)

        if youden_annotation:
            # Youden point and red dashed lines
            ax.scatter(mean_best_fpr, mean_best_tpr, color='red', zorder=5, label=f'Best Youden J = {mean_best_j:.3f}\nThreshold = {mean_best_threshold:.2f}')
            ax.axvline(mean_best_fpr, color='red', linestyle='--')
            ax.axhline(mean_best_tpr, color='red', linestyle='--')

        # plot mean roc
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, float(std_auc)),
            lw=5,
            alpha=0.8,
        )

        # plot chance
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0)
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="red", label="Chance", alpha=0.8)

        # plot fold roc std
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        simple_model_name = self.get_simple_model_name(model_name)

        if simple_model_name != "": # Single model
            ax.set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                xlabel='False Positive Rate', 
                ylabel='True Positive Rate',
                # title=simple_model_name + "\nReceiver operating characteristic curve",
            )
            title_text = simple_model_name + "\nReceiver operating characteristic curve"
            ax.set_title(title_text, fontsize=22)
        else: # Ensemble
            if self.ensemble:
                ax.set(
                    xlim=[-0.05, 1.05],
                    ylim=[-0.05, 1.05],
                    xlabel='False Positive Rate', 
                    ylabel='True Positive Rate',
                    # title="Ensemble " + str(type(cv_models_trained[0]).__name__) + "\nReceiver operating characteristic curve",
                    )
                title_text = "Ensemble " + str(type(cv_models_trained[0]).__name__) + "\nReceiver operating characteristic curve"
                ax.set_title(title_text, fontsize=22)
            else:
                ax.set(
                    xlim=[-0.05, 1.05],
                    ylim=[-0.05, 1.05],
                    xlabel='False Positive Rate', 
                    ylabel='True Positive Rate',
                    # title=str(type(cv_models_trained[0]).__name__) + "\nReceiver operating characteristic curve",
                    )
                title_text = str(type(cv_models_trained[0]).__name__) + "\nReceiver operating characteristic curve"
                ax.set_title(title_text, fontsize=22)

        # Config legend
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0, fontsize=16)

        # Set the font size for the axis labels
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        # Set the font size for the tick labels
        ax.tick_params(axis='both', which='major', labelsize=16)

        if not os.path.exists(self.plot_save_dir + "/AUC"):
            os.makedirs(self.plot_save_dir + "/AUC", exist_ok=True)
        
        
        if simple_model_name != "":
            if not os.path.exists(self.plot_save_dir + "/AUC/" + simple_model_name + "/"):
                    os.makedirs(self.plot_save_dir + "/AUC/" + simple_model_name + "/", exist_ok=True)
        else:
            if not os.path.exists(self.plot_save_dir + "/AUC/" + model_name):
                os.makedirs(self.plot_save_dir + "/AUC/" + model_name, exist_ok=True)

        if save_plot:
            if self.optimize:
                if simple_model_name != "":
                    plt.savefig(
                        self.plot_save_dir + "/AUC/" + simple_model_name + "/" + str(
                            type(cv_models_trained[0]).__name__) + "_" + self.RunID + "_" + str(
                            round(mean_auc, 3)) + "_cv_AUROC_optimized.png")
                else:
                    plt.savefig(
                        self.plot_save_dir + "/AUC/" + model_name + "/" + str(
                            type(cv_models_trained[0]).__name__) + "_" + self.RunID + "_" + str(
                            round(mean_auc, 3)) + "_cv_AUROC_optimized.png")
            else:
                if simple_model_name != "":
                    plt.savefig(
                        self.plot_save_dir + "/AUC/" + simple_model_name + "/" + str(
                            type(cv_models_trained[0]).__name__) + "_" + self.RunID + "_" + str(
                            round(mean_auc, 3)) + "_cv_AUROC.png")
                else:
                    plt.savefig(
                        self.plot_save_dir + "/AUC/" + model_name + "/" + str(
                            type(cv_models_trained[0]).__name__) + "_" + self.RunID + "_" + str(
                            round(mean_auc, 3)) + "_cv_AUROC.png")
        else:
            return plt

        plt.clf()
        plt.close()

    def get_data_split(self, train_index, val_index, X, y):
        """"
        :param train_index: index for training data
        :param val_index: index for validation data
        :param X: data
        :param y: label
        """

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]

        try:
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        except AttributeError:
            y_train, y_val = y[train_index], y[val_index]

        return X_train, y_train, X_val, y_val

    def cv_model_training(self, model, model_name, X, y, groups, plotter):
        """
        Trains and plots models on cross-validation with AUC and saves models
        :param model: model to train and save
        :param model_name: Name of the model
        :param X: Data to learn from
        :param y: Label to learn from
        :param groups: Groups of data to concatinate
        :param plotter: PerformancePlotter instance
        :return cv_models: Trained models for every fold
        """

        if not os.path.exists(self.model_save_dir + "/CV"):
            os.mkdir(self.model_save_dir + "/CV")

        cv_models_for_ensemble = {}
        cv_models_paths = []
        cv_models = []
        X_vals = []
        y_vals = []

        X_trains = []
        y_trains = []

        train_auc_cv = []
        val_auc_cv = []
        test_auc_cv = []

        fold = 0

        # Cross Validation
        for train_index, val_index in self.cv.split(X=X, y=y, groups=groups):

            #### Training in CV ####

            print("Training Fold", fold, "...")
            self.logger.info("Training Fold " + str(fold) + " ...")
            self.fold = fold

            if self.optimize:
                if self.extended_parameter_set:
                    if not model_name.endswith("_extended_hyperparameters"):
                        model_name = model_name + "_extended_hyperparameters"

                filename = self.model_save_dir + "/CV/optimized/" + model_name + "/" + self.RunID + "_cv_optimized_fold_" + str(
                    fold) + "_" + str(type(model).__name__) + "_" + str(self.rand_state) + '.sav'
            else:
                filename = self.model_save_dir + "/CV/trained/" + model_name + "/" + self.RunID + "_cv_trained_fold_" + str(
                    fold) + "_" + str(type(model).__name__) + "_" + str(self.rand_state) + '.sav'

            # get Data Split
            X_train, y_train, X_val, y_val = self.get_data_split(train_index=train_index,
                                                                 val_index=val_index,
                                                                 X=X,
                                                                 y=y)
            if not os.path.exists(filename):

                if not model is None:
                    # reset model
                    model = clone(model)

                    if self.optimize:

                        self.logger.info("Optimizing " + type(model).__name__ + " on fold " + str(fold) + " ...")
                        print(3 * "#", "Optimizing " + type(model).__name__ + " on fold " + str(fold), 3 * "#")

                        # Optimize
                        best_model, train_auc, val_auc, test_auc = self.get_best_model(X=X_train,
                                                                                       y=y_train,
                                                                                       model=model,
                                                                                       out_dir=self.model_save_dir + "/CV/optimized/" + model_name + "/",
                                                                                       X_val=X_val,
                                                                                       y_val=y_val,
                                                                                       filename=filename,
                                                                                       model_name="optimized_" + str(
                                                                                           type(
                                                                                               model).__name__) + "_fold_" + str(
                                                                                           fold)
                                                                                       )
                        if not best_model is None:

                            model = best_model
                            train_auc_cv.append(train_auc)
                            val_auc_cv.append(val_auc)
                            test_auc_cv.append(test_auc)

                            if self.shap_analysis:
                                self.plotter.plot_shap(model=model, X=X_train, y=y_train,
                                                       output_path=self.shap_plot_save_dir + "/" + "optimized_" + model_name + "_fold_" + str(
                                                           fold), seed=self.rand_state)

                            if len(self.X_val) == 0:
                                self.eval_model(model=model,
                                                X_test=self.X_test,
                                                y_test=self.y_test,
                                                X_val=X_val,
                                                y_val=y_val,
                                                model_name="optimized_" + model_name + "_" + str(
                                                    fold) + "_on_fold_val")
                            else:
                                self.eval_model(model=model,
                                                X_test=self.X_test,
                                                y_test=self.y_test,
                                                X_val=self.X_val,
                                                y_val=self.y_val,
                                                model_name="optimized_" + model_name + "_" + str(
                                                    fold))

                            if not hasattr(model, "classes_"):
                                if "TabNet" in type(model).__name__:
                                    # Train
                                    model.fit(X_train=X_train.values,
                                              y_train=y_train.values,
                                              eval_metric=['auc'],
                                              max_epochs=self.optimization_iter,
                                              patience=10,
                                              batch_size=32,
                                              virtual_batch_size=32,
                                              num_workers=self.n_cpus,
                                              weights=1,
                                              drop_last=True
                                              )

                                else:
                                    # Train
                                    model = model.fit(X_train, y_train)



                            if self.shap_analysis:
                                self.plotter.plot_shap(model=model, X=X_train, y=y_train,
                                                       output_path=self.shap_plot_save_dir + "/" + "optimized_" + model_name + "_fold_" + str(
                                                           fold), seed=self.rand_state)

                    else:
                        ### No optimization only CV Training ###
                        if not hasattr(model, "classes_"):
                                if "TabNet" in type(model).__name__:
                                    # Train
                                    model.fit(X_train=X_train.values,
                                              y_train=y_train.values,
                                              eval_metric=['auc'],
                                              max_epochs=self.optimization_iter,
                                              patience=10,
                                              batch_size=32,
                                              virtual_batch_size=32,
                                              num_workers=self.n_cpus,
                                              weights=1,
                                              drop_last=True
                                              )

                                else:
                                    # Train
                                    model = model.fit(X_train, y_train)
                                    
                        if len(self.X_val) == 0:
                            self.eval_model(model=model,
                                            X_test=self.X_test,
                                            y_test=self.y_test,
                                            X_val=X_val,
                                            y_val=y_val,
                                            model_name="cv_trained_" + model_name + "_" + str(
                                                fold) + "_on_fold_val")
                        else:
                            self.eval_model(model=model,
                                            X_test=self.X_test,
                                            y_test=self.y_test,
                                            X_val=self.X_val,
                                            y_val=self.y_val,
                                            model_name="cv_trained_" + model_name + "_" + str(
                                                fold))

                    if not model is None:
                        # Save
                        self.save_model(model=model, file_path=filename)
                else:
                    print("Model not applicable.")
                    self.error.error("Model not applicable.")
            else:
                # Load
                model = self.load_model(filename)
                
                if not hasattr(model, "classes_"):
                    if "TabNet" in type(model).__name__:
                        model.fit(X_train=X_train.values,
                                  y_train=y_train.values,
                                  eval_metric=['auc'],
                                  max_epochs=self.optimization_iter,
                                  patience=10,
                                  batch_size=32,
                                  virtual_batch_size=32,
                                  num_workers=self.n_cpus,
                                  weights=1,
                                  drop_last=True
                                  )
                    else:
                        model = model.fit(X_train, y_train)

                if not model is None:
                    print("Found trained model!")
                    self.logger.info("Found trained model at " + filename + "!")

                    if "TabNet" in type(model).__name__:
                        # Check for overfitting models
                        y_pred_prob = model.predict_proba(X_train.values)[:, 1]
                        y_pred_prob_val = model.predict_proba(X_val.values)[:, 1]
                        y_pred_prob_test = model.predict_proba(self.X_test.values)[:, 1]
                    elif "Ensemble" in type(model).__name__:
                        if len(model.clfs) > 1:
                            if self.check_tabnet_in_ensemble(model):
                                y_pred_prob = model.predict_proba(X_train.values)[:, 1]
                                y_pred_prob_val = model.predict_proba(X_val.values)[:, 1]
                                y_pred_prob_test = model.predict_proba(self.X_test.values)[:, 1]
                    else:
                        if not model is None:
                            # Check for overfitting models
                            y_pred_prob = model.predict_proba(X_train)[:, 1]
                            y_pred_prob_val = model.predict_proba(X_val)[:, 1]
                            y_pred_prob_test = model.predict_proba(self.X_test)[:, 1]

                    if not model is None:
                        if self.task == "binary_classification":
                            train_auc_cv.append(roc_auc_score(y_train, y_pred_prob))
                            val_auc_cv.append(roc_auc_score(y_val, y_pred_prob_val))
                            test_auc_cv.append(roc_auc_score(self.y_test, y_pred_prob_test))

                        elif self.task == "multi_class_classification":
                            train_auc_cv.append(roc_auc_score(y_train, y_pred_prob, multi_class='ovr'))
                            val_auc_cv.append(roc_auc_score(y_val, y_pred_prob_val, multi_class='ovr'))
                            test_auc_cv.append(roc_auc_score(self.y_test, y_pred_prob_test, multi_class='ovr'))

                        else:
                            self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                            raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                    
                    if self.shap_analysis:
                        self.plotter.plot_shap(model=model, X=X_train, y=y_train,
                                                output_path=self.shap_plot_save_dir + "/" + "optimized_" + model_name + "_fold_" + str(fold), 
                                                seed=self.rand_state)    
            if not model is None:
                if not hasattr(model, "classes_"):
                    if "TabNet" in type(model).__name__:
                        model.fit(X_train=X_train.values,
                                  y_train=y_train.values,
                                  eval_metric=['auc'],
                                  max_epochs=self.optimization_iter,
                                  patience=10,
                                  batch_size=32,
                                  virtual_batch_size=32,
                                  num_workers=self.n_cpus,
                                  weights=1,
                                  drop_last=True
                                  )
                    else:
                        model = model.fit(X_train, y_train)

                
                # get ensemble per model for all cv folds
                cv_models.append(model)
                X_vals.append(X_val)
                y_vals.append(y_val)

                X_trains.append(X_train)
                y_trains.append(y_train)

                if "TabNet" in type(model).__name__:
                    y_pred = model.predict(X_val.values)
                elif "Ensemble" in str(type(model).__name__):
                    if len(model.clfs) > 1:
                        if self.check_tabnet_in_ensemble(model):
                            y_pred = model.predict(X_val.values)
                else:
                    y_pred = model.predict(X_val)
                
                simple_model_name = self.get_simple_model_name(model_name)

                if simple_model_name != "":
                    if not os.path.exists(self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "/"):
                        os.makedirs(self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "/")

                    plotter.plot_conv_matrix(title=str(type(model).__name__) + "_on_validation_fold",
                                            y=y_val,
                                            y_pred=y_pred,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "/",
                                            fold=fold)

                else:
                    plotter.plot_conv_matrix(title=str(type(model).__name__) + "_on_validation_fold",
                                            y=y_val,
                                            y_pred=y_pred,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/CV/" + model_name + "/",
                                            fold=fold)

                if model_name not in cv_models_for_ensemble:
                    cv_models_for_ensemble[model_name] = [model]
                else:
                    cv_models_for_ensemble[model_name].append(model)
           
            cv_models_paths.append(filename)
            fold += 1

        # save fold training
        if not os.path.exists(self.out_folder + "/splits_used.json"):
            try:
                self.export_used_splits_simple(X_folds_val=X_vals, X_folds_train=X_trains)
            except Exception as e:
                self.error.warning(f"Failed to write splits_used.json: {e}")
                print(f"Failed to write splits_used.json: {e}")

        if len(cv_models) > 0:
            self.cv_AUC_generator(cv_models, X_vals, y_vals, model_name, plotter)
            del X_vals
            del y_vals
            del X_trains
            del y_trains

        if not model is None:
            self.train_auc_cv = train_auc_cv
            self.val_auc_cv = val_auc_cv
            self.test_auc_cv = test_auc_cv
            self.model_name = model_name

            if not os.path.exists(self.out_folder + "/plots/overfitting_plot/"):
                    os.makedirs(self.out_folder + "/plots/overfitting_plot/")

            self.plotter.plot_overfitting_plot(train_auc_cv=train_auc_cv,
                                               val_auc_cv=val_auc_cv,
                                               test_auc_cv=test_auc_cv,
                                               model_name=model_name,
                                               output_path=self.out_folder + "/plots/overfitting_plot/")
            
        print(3*"#", "Ensembling " + model_name + " CV training", 3*"#")
        self.logger.info("### " + "Ensembling " + model_name + " CV training" + " ###")

        ### Perform ensembling on CV models ###
        if self.optimize:
            ensemble_soft, ensemble_hard, ensembled_model_names = self.ensembling(list_of_models=cv_models,
                                                            model_name=model_name + "_cv_optimized",
                                                            out_folder=self.model_save_dir + "/CV/optimized/" + model_name + "/",
                                                            fit_estimators=False,
                                                            get_ensambled_model_names=True)
        else:
            ensemble_soft, ensemble_hard, ensembled_model_names = self.ensembling(list_of_models=cv_models,
                                                           model_name=model_name + "_cv_trained",
                                                           out_folder=self.model_save_dir + "/CV/trained/" + model_name + "/",
                                                           fit_estimators=False,
                                                           get_ensambled_model_names=True)

        return cv_models, cv_models_for_ensemble

    def train_cv(self, model, model_name, plotter: PerformancePlotter, optimize=None):
        """
        Train the model in cross-validation fashion
        :param model: Classification model
        :param model_name: name of the model
        """
        if self.use_wandb:
            run = wandb.init(
                                project=self.wandb_project_name,
                                tags=["RPTK"],
                                name="Cross_validation_" + model_name
                            )

        ensemble_models = []
        cv_models = []

        print(3 * "#", "Training", str(type(model).__name__), "with Cross Validation", 3 * "#")

        if self.task == "binary_classification":
            self.scoring = "roc_auc"
        elif self.task == "multi_class_classification":
            self.scoring = "roc_auc"
        else:
            self.scoring = "neg_mean_absolute_error"

        if optimize is None:
            self.optimize = False

        if not os.path.exists(self.model_save_dir + "/CV/trained/" + model_name + "/"):
            os.makedirs(self.model_save_dir + "/CV/trained/" + model_name + "/")

        filename = self.model_save_dir + "/CV/trained/" + model_name + "/" + self.RunID + "_trained_" + model_name + '.sav'

        if "Ensemble" not in str(type(model).__name__):
            if "TabNet" not in type(model).__name__:
                scores = cross_val_score(model,
                                         self.X_train.copy(),
                                         self.y_train.copy(),
                                         groups=self.X_train.index,
                                         n_jobs=self.n_cpus,
                                         cv=self.cv,
                                         scoring=self.scoring)

                print(self.scoring + ": %0.2f (+/- %0.2f) [%s]" % (
                    round(scores.mean(), 3), round(scores.std(), 3), model_name))
                self.logger.info(
                    self.scoring + ": %0.2f (+/- %0.2f) [%s]" % (
                        round(scores.mean(), 3), round(scores.std(), 3), model_name))

        if not os.path.exists(filename):

            if not hasattr(model, "classes_"):
                if "TabNet" in type(model).__name__:
                    print("Need to retrain {}.".format(type(model).__name__))
                    # train model
                    model.fit(X_train=self.X_train.values,
                              y_train=self.y_train.values,
                              eval_metric=['auc'],
                              max_epochs=self.optimization_iter,
                              patience=10,
                              batch_size=32,
                              virtual_batch_size=32,
                              num_workers=self.n_cpus,
                              weights=1,
                              drop_last=True
                              )
                else:
                    print("Need to retrain {}.".format(type(model).__name__))
                    # train model
                    model = model.fit(self.X_train, self.y_train)

            # save trained model
            self.save_model(model=model, file_path=filename)
            if "Ensemble" not in type(model).__name__:
                cv_models, ensemble_models = self.cv_model_training(model=model,
                                                                    model_name=model_name,
                                                                    X=self.X_train.copy(),
                                                                    y=self.y_train.copy(),
                                                                    groups=self.X_train.copy().index,
                                                                    plotter=plotter)

            if (self.task == "binary_classification") or (self.task == "multi_class_classification"):

                if not hasattr(model, "classes_"):
                    if "TabNet" in type(model).__name__:
                        # train model
                        model.fit(
                            X_train=self.X_train.values,
                            y_train=self.y_train.values,
                            eval_metric=['auc'],
                            max_epochs=self.optimization_iter,
                            patience=10,
                            batch_size=32,
                            virtual_batch_size=32,
                            num_workers=self.n_cpus,
                            weights=1,
                            drop_last=True
                        )
                    else:
                        # train model
                        model = model.fit(self.X_train, self.y_train)

                if "TabNet" in type(model).__name__:
                    y_pred_full = cross_val_predict(model,
                                                    self.X_train.values.copy(),
                                                    self.y_train.values.copy(),
                                                    n_jobs=self.n_cpus,
                                                    cv=self.cross_val_splits,
                                                    # fit_params={"scoring":self.scoring},
                                                    groups=self.X_train.copy().index,
                                                    method="predict")
                else:
                    y_pred_full = cross_val_predict(model,
                                                    self.X_train.copy(),
                                                    self.y_train.copy(),
                                                    n_jobs=self.n_cpus,
                                                    cv=self.cross_val_splits,
                                                    # fit_params={"scoring":self.scoring},
                                                    groups=self.X_train.copy().index,
                                                    method="predict")
                

                simple_model_name = self.get_simple_model_name(model_name)

                if simple_model_name != "":
                    plotter.plot_conv_matrix(title=str(type(model).__name__) + "_on_train_prediction",
                                            y=self.y_train,
                                            y_pred=y_pred_full,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "/")
                else:
                    plotter.plot_conv_matrix(title=str(type(model).__name__) + "_on_train_prediction",
                                            y=self.y_train,
                                            y_pred=y_pred_full,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/CV/" + model_name + "/")

                if not os.path.exists(self.plot_save_dir + "/confusion_matrix/CV/"):
                    os.makedirs(self.plot_save_dir + "/confusion_matrix/CV/")

                if simple_model_name != "":
                    if not os.path.exists(self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "/"):
                        os.makedirs(self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "/")
                else:
                    if not os.path.exists(self.plot_save_dir + "/confusion_matrix/CV/" + model_name + "/"):
                        os.makedirs(self.plot_save_dir + "/confusion_matrix/CV/" + model_name + "/")

        else:
            model = self.load_model(filename)

            print("Found trained model!")
            self.logger.info("Found optimized model at " + filename + "!")
            cv_models, ensemble_models = self.cv_model_training(model=model,
                                                                model_name=model_name,
                                                                X=self.X_train.copy(),
                                                                y=self.y_train.copy(),
                                                                groups=self.X_train.copy().index,
                                                                plotter=plotter)

            if len(self.X_val) == 0:
                self.eval_model(model=model,
                                X_test=self.X_test.copy(),
                                y_test=self.y_test.copy(),
                                X_val=None,
                                y_val=None,
                                model_name="trained_cv_" + model_name)
            else:
                self.eval_model(model=model,
                                X_test=self.X_test.copy(),
                                y_test=self.y_test.copy(),
                                X_val=self.X_val.copy(),
                                y_val=self.y_val.copy(),
                                model_name="trained_cv_" + model_name)

        return cv_models

    def train_model(self, clf, label):
        """
        Train model without cross-validation
        :param clf: Classification model
        :param label: Label in the data
        """

        if self.optimize:
            if not os.path.exists(self.model_save_dir + "/optimized/" + type(clf).__name_ + "/"):
                os.makedirs(self.model_save_dir + "/optimized/" + type(clf).__name_ + "/")

            filename = self.model_save_dir + "/optimized/" + type(
                clf).__name_ + "/" + self.RunID + "_optimized_" + type(clf).__name_ + '.sav'
        else:
            if not os.path.exists(self.model_save_dir + "/trained/" + type(clf).__name_ + "/"):
                os.makedirs(self.model_save_dir + "/trained/" + type(clf).__name_ + "/")

            filename = self.model_save_dir + "/trained/" + type(clf).__name_ + "/" + self.RunID + "_trained_" + type(
                clf).__name_ + '.sav'

        if not os.path.exists(filename):

            if self.optimize:

                self.logger.info("Optimizing " + type(model).__name__ + " ...")
                print(3 * "#", "Optimizing " + type(model).__name__, 3 * "#")

                # optimize on total training dataset
                clf, train_auc, val_auc, test_auc = self.get_best_model(X=self.X_train.copy(),
                                                                        y=self.y_train.copy(),
                                                                        model=clf,
                                                                        out_dir=self.model_save_dir + "/CV/optimized/" + type(
                                                                            clf).__name_ + "/",
                                                                        X_val=None,
                                                                        y_val=None,
                                                                        filename=filename,
                                                                        model_name=type(clf).__name_)

            else:
                if "TabNet" in type(clf).__name__:
                    clf.fit(self.X_train.values, self.y_train.values)
                else:
                    clf = clf.fit(self.X_train, self.y_train)

        else:
            print("Found trained model!")
            clf = self.load_model(filename)

            self.logger.info("Found optimized model at " + filename + "!")

        if self.optimize:
            if len(self.X_val) == 0:
                self.eval_model(model=clf,
                                X_test=self.X_test.copy(),
                                y_test=self.y_test.copy(),
                                X_val=None,
                                y_val=None,
                                model_name="optimized_" + type(clf).__name_)
            else:
                self.eval_model(model=clf,
                                X_test=self.X_test.copy(),
                                y_test=self.y_test.copy(),
                                X_val=self.X_val.copy(),
                                y_val=self.y_val.copy(),
                                model_name="optimized_" + type(clf).__name_)

        else:
            if len(self.X_val) == 0:
                self.eval_model(model=clf,
                                X_test=self.X_test.copy(),
                                y_test=self.y_test.copy(),
                                X_val=None,
                                y_val=None,
                                model_name="trained_" + type(clf).__name_)
            else:
                self.eval_model(model=clf,
                                X_test=self.X_test.copy(),
                                y_test=self.y_test.copy(),
                                X_val=self.X_val.copy(),
                                y_val=self.y_val.copy(),
                                model_name="trained_" + type(clf).__name_)

        if type(clf).__name__ == "KNeighborsClassifier":
            # No predict_proba available for Lasso
            y_pred = clf.predict(self.X_val.values)
        elif "TabNet" in type(clf).__name__:
            # No predict_proba available for Lasso
            y_pred = clf.predict(self.X_val.values)
        elif "Ensemble" in str(type(clf).__name__):
            if len(clf.clfs) > 1:
                if self.check_tabnet_in_ensemble(model):
                    y_pred = clf.predict(self.X_val.values)
        else:
            y_pred = clf.predict(self.X_val)

        if self.task == "binary_classification":
            auc = roc_auc_score(self.y_val, y_pred)
        elif self.task == "multi_class_classification":
            auc = roc_auc_score(self.y_val, y_pred, multi_class='ovr')
        else:
            self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
            raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
        

        print("AUROC: %0.2f [%s]" % (auc, label))
        self.logger.info("AUROC: %0.2f [%s]" % (auc, label))

        # save model
        filename = self.model_save_dir + "/" + self.RunID + "_trained_" + str(type(clf).__name__) + "_" + str(
            self.rand_state) + '.sav'
        self.save_model(model=clf, file_path=filename)

        # Plot decision region of trained model
        plotter = PerformancePlotter(output_path=self.out_folder,
                                     X=self.X_train.copy(),
                                     y=self.y_train.copy(),
                                     y_pred=y_pred,
                                     RunID=self.RunID,
                                     error=self.error,
                                     logger=self.logger)

        # plotter.plot_decision_regions_from_features(model=clf)

        # Plot Convolution Matrix
        if type(clf).__name__ == "KNeighborsClassifier":
            # No predict_proba available for Lasso
            y_pred = clf.predict_proba(self.X_val.values)

        elif "TabNet" in type(clf).__name__:
            # No predict_proba available for Lasso
            y_pred = clf.predict(self.X_val.values)
        elif "Ensemble" in str(type(clf).__name__):
            if len(clf.clfs) > 1:
                if self.check_tabnet_in_ensemble(model):
                    y_pred = clf.predict(self.X_val.values)
        else:
            y_pred = clf.predict_proba(self.X_val)

        plotter.y = self.y_val
        plotter.y_pred = y_pred

        if (self.task == "binary_classification") or (self.task == "multi_class_classification"):
            plotter.plot_conv_matrix(title=str(type(clf).__name__ + "_" + str(self.RunID)),
                                     label=clf.classes_,
                                     path=self.plot_save_dir + "/confusion_matrix/")

    def get_decision_regions(self, model, filetitle, plotter: PerformancePlotter = None):
        """
        Plot decision regions for every classifier to get an idea of performance
        :return:
        """

        if plotter is None:
            plotter = PerformancePlotter(output_path=self.out_folder,
                                         X=self.X_train.copy(),
                                         y=self.y_train.copy(),
                                         RunID=self.RunID,
                                         error=self.error,
                                         logger=self.logger)

        performance = pd.DataFrame({"Features": [], "Performance": []})
        best_feature_files = []

        if "TabNet" in type(model).__name__:
            pass
        else:
            # 1. get all csv files with best selected features
            best_feature_files = glob.glob(self.best_selected_features_folder_path + "/Best_Selected_features_*.csv")
        if len(best_feature_files) > 0:
            feature_not_found = False

            # 2. train model on this features and save result
            for best_feature_file in best_feature_files:
                with open(best_feature_file) as f:
                    best_features = f.read().splitlines()

                for feature in best_features:
                    if feature not in self.X_train.columns:
                        feature_not_found = True

                if not feature_not_found:
                    X_train_best_feature_set = self.X_train.copy()[best_features]

                    scores = cross_val_score(model,
                                             X_train_best_feature_set,
                                             self.y_train.copy(),
                                             groups=self.X_train.index,
                                             n_jobs=self.n_cpus,
                                             cv=self.cv,
                                             scoring=self.scoring)

                    tmp = pd.DataFrame({"Features": [best_features], "Performance": [scores.mean()]})
                    performance = pd.concat([performance, tmp], ignore_index=True)
                else:
                    self.error.warning(
                        "Could not find selected best features in feature space. Check feature formating and filtering.")
                    return

            # 3. select best features for all collections of best features
            best_best_features = performance.loc[performance['Performance'].idxmax(), "Features"]

            model = clone(model)
            if type(model).__name__ == "XGBClassifier":
                model = model.fit(self.X_train.copy()[best_best_features].values, self.y_train.copy().values)
            else:
                model = model.fit(self.X_train.copy()[best_best_features], self.y_train.copy())

            title = "Trained " + type(model).__name__
            filename = self.plot_save_dir + '/Decision_region_plots/Decision_region_' + filetitle + ".png"

            if not os.path.exists(self.plot_save_dir + "/Decision_region_plots/"):
                os.makedirs(self.plot_save_dir + "/Decision_region_plots/")

            # Plot decision region of trained model
            plotter.plot_decision_regions_from_features(model=model,
                                                        X=self.X_train.copy()[best_best_features],
                                                        y=self.y_train.copy(),
                                                        title=title,
                                                        filename=filename)

        else:
            print("Could not plot decision regions. No selected features found.")
            self.error.warning("Could not plot decision regions. No selected features found.")

    def check_tabnet_in_ensemble(self, ensemble):
        """
        Checking if TabNet is in Ensemble models
        :param ensemble: EnsembleVoteClassifier
        :return (bool): True if TabNet is in Ensemble
        """
        tabnet_included = False
        if len(ensemble.clfs) > 0:
            for model in ensemble.clfs:
                if "TabNet" in type(model).__name__:
                    tabnet_included = True
                    break
                elif "Ensemble" in type(model).__name__:
                    for mod in model.clfs:
                        if "TabNet" in type(mod).__name__:
                            tabnet_included = True
                            break
        else:
            self.error.warning("No models included in Ensemble.")
            print("No models included in Ensemble.")

        return tabnet_included

    def test(self, models, model_name):
        """
        Test trained models on the hould out test set
        """

        # if cross validation get AUROC plot for each and all models together
        # get test AUROC
        # bootstrap test AUROC
        if isinstance(models, list):

            # Test pretrained models
            model_filename = self.model_save_dir + "/pretrained_models/" + self.RunID + "_stable_pretrained_" + model_name + '.sav'

            if os.path.exists(model_filename):
                print("Found trained model!")
                pretrained_model = self.load_model(model_filename)

                if not hasattr(pretrained_model, "classes_"):
                    self.error.warning("Loaded model does not seem to be trained. Please check: " + model_filename)
                    print("Warning: Loaded model does not seem to be trained. Please check: " + model_filename)

                # self.plotter.plot_AUC(model=pretrained_model, X_test=self.X_test, y_test=self.y_test,
                #                      model_name=model_name + "_test")

            else:
                print("Can not find pretrained model for testing. " + model_filename)
                self.error.warning("Can not find pretrained model for testing. " + model_filename)

            # Test cross validation models
            if self.use_cross_validation:
                print("Test cross validation models of " + model_name)
                self.logger.info("Test cross validation models of " + model_name)

                # check in the folders of cv models for every model
                cv_model_files = glob.glob(
                    self.model_save_dir + "/CV/" + model_name + "/" + self.RunID + "_cv_trained_fold_*.sav")

                print("Found {} trained models. ".format(str(len(cv_model_files))))
                self.logger.info("Found {} trained models. ".format(str(len(cv_model_files))))

                # load these modelse and put them into a list
                tprs = []
                aucs = []
                mean_tpr = []
                mean_auc = []
                std_auc = 0.0

                mean_fpr = np.linspace(0, 1, 100)

                fig, ax = plt.subplots()

                fold = 0
                for cv_model_path in cv_model_files:
                    cv_model = self.load_model(cv_model_path)

                    if not hasattr(cv_model, "classes_"):
                        self.error.warning("Loaded model does not seem to be trained. Please check: " + cv_model_path)
                        print("Warning: Loaded model does not seem to be trained. Please check: " + cv_model_path)

                    if "TabNet" in type(cv_model).__name__:
                        # Predict
                        y_pred = cv_model.predict_proba(self.X_test.values)
                    elif "Ensemble" in str(type(cv_model).__name__):
                        if len(cv_model.clfs) > 1:
                            if self.check_tabnet_in_ensemble(cv_model):
                                y_pred = cv_model.predict_proba(self.X_test.values)
                    else:
                        # Predict
                        y_pred = cv_model.predict_proba(self.X_test)

                    viz = self.plotter.plot_cv_AUC(y=self.y_test, y_pred=y_pred[:, 1], fold=fold, ax=ax)

                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(viz.roc_auc)

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = auc(mean_fpr, mean_tpr)
                    std_auc = np.std(aucs)

                    fold += 1

                ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0)
                ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

                ax.plot(
                    mean_fpr,
                    mean_tpr,
                    color="b",
                    label=r"Mean Test ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, float(std_auc)),
                    lw=2,
                    alpha=0.8,
                )

                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

                ax.fill_between(
                    mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color="grey",
                    alpha=0.2,
                    label=r"$\pm$ 1 std. dev.",
                )

                ax.set(
                    xlim=[-0.05, 1.05],
                    ylim=[-0.05, 1.05],
                    title=model_name + "\nReceiver operating characteristic curve",
                )
                ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0)

                simple_model_name = self.get_simple_model_name(model_name)

                if not os.path.exists(self.plot_save_dir + "/AUC"):
                    os.makedirs(self.plot_save_dir + "/AUC", exist_ok=True)

                if simple_model_name != "":
                    if not os.path.exists(self.plot_save_dir + "/AUC/" + simple_model_name):
                        os.makedirs(self.plot_save_dir + "/AUC/" + simple_model_name, exist_ok=True)
                else:
                    if not os.path.exists(self.plot_save_dir + "/AUC/" + model_name):
                        os.makedirs(self.plot_save_dir + "/AUC/" + model_name, exist_ok=True)

                if simple_model_name != "":
                    plt.savefig(self.plot_save_dir + "/AUC/" + simple_model_name + "/" + str(
                        type(model).__name__) + "_" + self.RunID + "_" + str(round(mean_auc, 2)) + "_cv_test_AUROC.png")
                else:
                    plt.savefig(self.plot_save_dir + "/AUC/" + model_name + "/" + str(
                        type(model).__name__) + "_" + self.RunID + "_" + str(round(mean_auc, 2)) + "_cv_test_AUROC.png")

                plt.clf()
                plt.close()

            # for model in models:
            # Check if model has been trained
            # if assert hasattr(model, "classes_"):

    def get_simple_model_name(self, model_name):
        """
        Get simple model name from string
        """

        simple_model_name = ""
        if "Best_Ensemble" not in model_name:
            if isinstance(self.model, list):
                model_names = [type(m).__name__ for m in self.model]
            else:
                model_names = [type(self.model).__name__]

            
            for model_n in model_names:
                if model_n in model_name:
                    simple_model_name = model_n
                    break
        else:
            simple_model_name = "Best_Ensemble_Combination"
        
        return simple_model_name

    def eval_model(self, model=None, X_test=None, y_test=None, X_val=None, y_val=None, model_name=None, val_pred=None):
        """
        Getting Evaluation parameters from a fitted model
        :param model: Model to evaluate
        :param X_test: Test set
        :param y_test: Test set label
        :param X_val: Validation set
        :param y_val: Validation set label
        :param model_name: Name of model
        :param val_pred: prediction of validation already present
        :return: pd.DataFrame with all evaluation parameters (AUC, F1, balanced ACC, as well as bootstrapped values
        """
        if model is None:
            print(str(model_name), "No model available to evaluate!")
            self.error.warning(str(model_name) + " No model available to evaluate!")
            return None

        if model_name is None:
            model_name = type(model).__name__

        # Predict with pre-trained Model
        if "Ensemble" in type(model).__name__:
            if len(model.clfs) > 0:
                if "Ensemble" in type(model.clfs[0]).__name__:
                    print("ENSEMBLING",type(model).__name__,  type(model.clfs[0]).__name__, type(model.clfs[0].clfs[0]).__name__)
                if self.check_tabnet_in_ensemble(model):
                    y_pred = model.predict(X_test.values)
                    y_predict_proba = model.predict_proba(X_test.values)[:, 1]

                else:
                    y_pred = model.predict(X_test)
                    y_predict_proba = model.predict_proba(X_test)[:, 1]
            else:
                self.error.warning("No models in Ensemble.")
                print("No models in Ensemble.")
                return None
            
            ensemble_model_name = PerformancePlotter.extract_ensemble_model_name(ensemble_string=model_name)

            if not ensemble_model_name is None:
                ensemble_model_name_short = ensemble_model_name.replace("SoftEnsemble_","")

                # get all fold val AUC from the model of interest:
                models = self.prediction_summary.copy().loc[self.prediction_summary["Models"].str.contains(ensemble_model_name_short), :]
                val_auc_fold_models, test_auc_fold_models, test_auc_ensemble_models = PerformancePlotter.get_AUC_values_per_models(df=models)
                # TODO val_auc_fold_models as dict calculate the mean and give it to the parameter val_pred
                

        elif "TabNet" in type(model).__name__:
            y_pred = model.predict(X_test.values)
            y_predict_proba = model.predict_proba(X_test.values)[:, 1]
            if X_val is not None:
                y_pred_val = model.predict(X_val.values)
                y_predict_proba_val = model.predict_proba(X_val.values)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_predict_proba = model.predict_proba(X_test)[:, 1]
            if X_val is not None:
                y_pred_val = model.predict(X_val)
                y_predict_proba_val = model.predict_proba(X_val)[:, 1]

        # Average Precision
        ap = average_precision_score(y_test, y_predict_proba)

        # AUROC
        if self.task == "binary_classification":
            auroc = roc_auc_score(y_test, y_predict_proba)
        elif self.task == "multi_class_classification":
            auroc = roc_auc_score(y_test, y_predict_proba, multi_class='ovr')
        else:
            self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
            raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
        
        true_labels = y_test.to_list()
        sample_ids = X_test.index.to_list()
        if X_val is not None:
            true_labels_val = y_val.to_list()
            sample_ids_val = X_val.index.to_list()

        sample_prediction_summary = pd.DataFrame.from_dict({"ID":sample_ids,
                                                            "Label":true_labels,
                                                            "Prediction": y_pred,
                                                            "Prediction_Proba": y_predict_proba,
                                                            })
        if X_val is not None:
            sample_val_prediction_summary = pd.DataFrame.from_dict({"ID":sample_ids_val,
                                                                    "Label":true_labels_val,
                                                                    "Prediction": y_pred_val,
                                                                    "Prediction_Proba": y_predict_proba_val,
                                                                    })

        # Plot prediction probability distribution
        if not os.path.exists(self.out_folder + "/plots/prediction_prob/"):
            os.makedirs(self.out_folder + "/plots/prediction_prob/")

        simple_model_name = self.get_simple_model_name(model_name)
        
        if simple_model_name != "":
            if not os.path.exists(self.out_folder + "/plots/prediction_prob/" + simple_model_name + "/"):
                os.makedirs(self.out_folder + "/plots/prediction_prob/" + simple_model_name + "/")

            PerformancePlotter.plot_probability_distribution(df=sample_prediction_summary, 
                                                            label_column="Label", 
                                                            proba_column="Prediction_Proba", 
                                                            save_path=self.out_folder + "/plots/prediction_prob/" + simple_model_name + "/", 
                                                            model_name=model_name)

        else:

            PerformancePlotter.plot_probability_distribution(df=sample_prediction_summary, 
                                                            label_column="Label", 
                                                            proba_column="Prediction_Proba", 
                                                            save_path=self.out_folder + "/plots/prediction_prob/", 
                                                            model_name=model_name)


        if not os.path.exists(self.out_folder + "/prediction_prob/"):
            os.makedirs(self.out_folder + "/prediction_prob/")

        if simple_model_name != "":
            if not os.path.exists(self.out_folder + "/prediction_prob/" + simple_model_name + "/"):
                os.makedirs(self.out_folder + "/prediction_prob/" + simple_model_name + "/")

            if "on_fold_val" in str(model_name):
                if X_val is not None:
                    sample_val_prediction_summary.to_csv(self.out_folder + "/prediction_prob/" + simple_model_name + "/prediction_prob_" + str(model_name) + ".csv")
                
                test_model_name = model_name
                test_model_name = test_model_name.replace("on_fold_val","fold_test")
                sample_prediction_summary.to_csv(self.out_folder + "/prediction_prob/" + simple_model_name + "/prediction_prob_" + str(test_model_name) + ".csv")
            else:
                if X_val is not None:
                    sample_val_prediction_summary.to_csv(self.out_folder + "/prediction_prob/prediction_prob_" + str(model_name) + "_on_val.csv")
                
                sample_prediction_summary.to_csv(self.out_folder + "/prediction_prob/" + simple_model_name + "/prediction_prob_" + str(model_name) + ".csv")
        else:
            if "on_fold_val" in str(model_name):
                if X_val is not None:
                    sample_val_prediction_summary.to_csv(self.out_folder + "/prediction_prob/prediction_prob_" + str(model_name) + ".csv")
                
                test_model_name = model_name
                test_model_name = test_model_name.replace("on_fold_val","fold_test")
                sample_prediction_summary.to_csv(self.out_folder + "/prediction_prob/prediction_prob_" + str(test_model_name) + ".csv")
            else:
                if X_val is not None:
                    sample_val_prediction_summary.to_csv(self.out_folder + "/prediction_prob/prediction_prob_" + str(model_name) + "_on_val.csv")
                
                sample_prediction_summary.to_csv(self.out_folder + "/prediction_prob/prediction_prob_" + str(model_name) + ".csv")

        # Balanced Accuracy
        bal_acc = balanced_accuracy_score(y_test, y_pred)  # , adjusted=True)

        # F score
        f1 = f1_score(y_test, y_pred)

        # Bootstrapping
        sample_statistic, lower, upper, auc_mean, auc_std = self.bootstrap_auc(y_test, y_predict_proba)
        sens_sample_statistic, sens_lower, sens_upper, sens_mean, sens_std = self.bootstrap_sensitivity(y_test, y_pred)
        sp_sample_statistic, sp_lower, sp_upper, sp_mean, sp_std = self.bootstrap_specificity(y_test, y_pred)
        f1_sample_statistic, f1_lower, f1_upper, f1_mean, f1_std = self.bootstrap_f1(y_test, y_pred)

        if val_pred is None:
            if X_val is not None:
                # Predict with pre-trained Model
                if "TabNet" in type(model).__name__:
                    val_y_pred = model.predict(X_val.values)
                    val_y_predict_proba = model.predict_proba(X_val.values)[:, 1]
                elif "Ensemble" in type(model).__name__:
                    if len(model.clfs) > 0:
                        if self.check_tabnet_in_ensemble(model):
                            val_y_pred = model.predict(X_val.values)
                            val_y_predict_proba = model.predict_proba(X_val.values)[:, 1]
                else:
                    val_y_pred = model.predict(X_val)
                    val_y_predict_proba = model.predict_proba(X_val)[:, 1]

                # Average Precision
                val_ap = average_precision_score(y_val, val_y_pred)

                # AUROC
                if self.task == "binary_classification":
                    val_auroc = roc_auc_score(y_val, val_y_predict_proba)
                elif self.task == "multi_class_classification":
                    val_auroc = roc_auc_score(y_val, val_y_predict_proba, multi_class='ovr')
                else:
                    self.error.error("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                    raise ValueError("Unknown Task! Please specify binary_classification, multi_class_classification or regression")
                
                # Balanced Accuracy
                val_bal_acc = balanced_accuracy_score(y_val, val_y_pred)  # , adjusted=True)

                # F score
                val_f1 = f1_score(y_val, val_y_pred)
            else:
                val_ap = 0.
                val_auroc = 0.
                val_bal_acc = 0.
                val_f1 = 0.
        else:
            val_auroc = val_pred
            val_bal_acc = 0.
            val_f1 = 0.
            val_ap = 0.

        model_parameter_entry = pd.DataFrame.from_dict({"Models": [model_name],
                                                        "Params": [model.get_params()],
                                                        "Val_AUC": [val_auroc],
                                                        "Test_AUC": [auroc]
                                                        })

        prediction_summary_entry = pd.DataFrame.from_dict({"Models": [model_name],
                                                           "Val_AUC": [val_auroc],
                                                           "Test_AUC": [auroc],
                                                           "Val_F1": [val_f1],
                                                           "Test_F1": [f1],
                                                           "Val_ACC": [val_bal_acc],
                                                           "Test_ACC": [bal_acc],
                                                           "Val_AP": [val_ap],
                                                           "Test_AP": [ap],
                                                           "Bootstrap_AUC": [auc_mean],
                                                           "Bootstrap_AUC_std": [auc_std],
                                                           "Bootstrap_AUC_upper": [upper],
                                                           "Bootstrap_AUC_lower": [lower],
                                                           "Bootstrap_Sensif": [sens_mean],
                                                           "Bootstrap_Sensif_std": [sens_std],
                                                           "Bootstrap_Sensif_upper": [sens_upper],
                                                           "Bootstrap_Sensif_lower": [sens_lower],
                                                           "Bootstrap_Specif": [sp_mean],
                                                           "Bootstrap_Specif_std": [sp_std],
                                                           "Bootstrap_Specif_upper": [sp_upper],
                                                           "Bootstrap_Specif_lower": [sp_lower],
                                                           "Bootstrap_F1": [f1_mean],
                                                           "Bootstrap_F1_std": [f1_std],
                                                           "Bootstrap_F1_upper": [f1_upper],
                                                           "Bootstrap_F1_lower": [f1_lower],
                                                           })
        self.logger.info("Model: " + model_name + " Test AUC: " + str(auroc) + " Test F1: " + str(f1) + " Test AP: " + str(ap))
        print("Model: " + model_name + " Test AUC: " + str(auroc) + " Test F1: " + str(f1) + " Test AP: " + str(ap))
        
        simple_model_name = self.get_simple_model_name(model_name)

        if simple_model_name != "":
            if not os.path.exists(self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "_test_set_performance/"):
                os.makedirs(self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "_test_set_performance/",exist_ok=True)

            if not os.path.exists(self.plot_save_dir + "/confusion_matrix/" + simple_model_name + "_test_set_performance/"):
                os.makedirs(self.plot_save_dir + "/confusion_matrix/" + simple_model_name + "_test_set_performance/",exist_ok=True)


        if simple_model_name != "":
            if "fold" in model_name:
                self.plotter.plot_conv_matrix(title=model_name + "_on_test_set",
                                            y=y_test,
                                            y_pred=y_pred,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/CV/" + simple_model_name + "_test_set_performance/")
            else:
                self.plotter.plot_conv_matrix(title=model_name + "_on_test_set",
                                            y=y_test,
                                            y_pred=y_pred,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/" + simple_model_name + "_test_set_performance/")

        else:
            if "fold" in model_name:
                self.plotter.plot_conv_matrix(title=model_name + "_on_test_set",
                                            y=y_test,
                                            y_pred=y_pred,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/CV/" + model_name + "_test_set_performance/")
            else:
                self.plotter.plot_conv_matrix(title=model_name + "_on_test_set",
                                            y=y_test,
                                            y_pred=y_pred,
                                            label=model.classes_,
                                            path=self.plot_save_dir + "/confusion_matrix/" + model_name + "_test_set_performance/")

        
        simple_model_name = self.get_simple_model_name(model_name)
        
        if simple_model_name != "":
            if not os.path.exists(self.plot_save_dir + "/AUC/" + simple_model_name + "/"):
                    os.makedirs(self.plot_save_dir + "/AUC/" + simple_model_name + "/",exist_ok=True)

            self.plotter.plot_AUC(model=model, X_test=X_test, y_test=y_test, model_name=model_name,
                                out_path=self.plot_save_dir + "/AUC/" + simple_model_name)
        else:
            self.plotter.plot_AUC(model=model, X_test=X_test, y_test=y_test, model_name=model_name,
                                out_path=self.plot_save_dir + "/AUC/" + model_name)

        self.model_parameter = pd.concat([self.model_parameter, model_parameter_entry])
        self.model_parameter = self.model_parameter.drop_duplicates(subset=["Models", "Test_AUC", "Val_AUC"])
        self.model_parameter.to_csv(self.out_folder + "/model_parameters.csv")

        self.prediction_summary = pd.concat([self.prediction_summary, prediction_summary_entry])
        self.prediction_summary = self.prediction_summary.drop_duplicates(
            subset=["Models", "Test_AUC", "Test_ACC", "Test_F1", "Test_AP"])
        self.prediction_summary.to_csv(self.out_folder + "/Prediction_summary.csv")
        # return prediction_summary_entry

    def get_stable_parameter_range(self, model):
        """
        Get range where to search for the which parameter in which range are you searching for
        """
        if type(model).__name__ == "Lasso":
            param_name = "alpha"
            param_range = range(1, 10, 1)
        elif (type(model).__name__ == "SVC") or (type(model).__name__ == "LinearSVC"):
            param_name = "C"
            param_range = np.logspace(-2.3, -1.3, 10) * float(len(self.X) * np.sqrt(len(self.X_train)))
        elif "TabNet" in type(model).__name__:
            param_name = 'n_d'
            param_range = range(10, 200, 40)
        else:
            param_name = "n_estimators"
            param_range = range(100, 1000, 100)

        return param_name, param_range

    def perform_stable_pretraining(self, model, overview_filename):
        """
        Perform pretraining for estimating the stable model performance based on the size of the model on cross validation
        """

        y_stable = 0.0
        os.makedirs(self.model_save_dir + "/pretrained_models", exist_ok=True)
        model_filename = self.model_save_dir + "/pretrained_models/" + self.RunID + "_stable_pretrained_" + type(model).__name__ + '.sav'
        
        if not os.path.exists(model_filename):
            
            self.logger.info("Performing pretraining " + str(type(model).__name__))

            param_name, param_range = self.get_stable_parameter_range(model)

            # check if model has been optimized already
            if not os.path.exists(model_filename):
                # loader.start()
                x_max, y_max, x_stable, y_stable = self.get_stable_model_size(model, param_name, param_range)

                tmp = pd.DataFrame({"Model": [type(model).__name__],
                                    "Parameter_name": [param_name],
                                    "Stable_Parameter_value": [x_stable],
                                    "Stable_performance_AUROC": [y_stable],
                                    "Max_Parameter_value": [x_max],
                                    "Max_performance_AUROC": [y_max]})

                if not tmp["Model"].values in self.pre_training_results["Model"].values:
                    self.pre_training_results = pd.concat([self.pre_training_results, tmp], ignore_index=True)

                if type(model).__name__ == "Lasso":
                    model.set_params(alpha=x_stable)
                elif (type(model).__name__ == "SVC") or (type(model).__name__ == "LinearSVC"):
                    model.set_params(C=x_max)  # performance increases constantly
                elif "TabNet" in type(model).__name__:
                    model.set_params(n_d=x_stable)
                else:
                    model.set_params(n_estimators=x_stable)

                self.stable_models.append(clone(model))
                self.get_decision_regions(model=model,
                                          filetitle="stable_" + type(model).__name__ + "_" + self.RunID)

                # Train Pretrained Model
                if "TabNet" in type(model).__name__:
                    model.fit(
                        X_train=self.X_train.values,
                        y_train=self.y_train.values,
                        eval_set=[(self.X_train.values, self.y_train.values)],
                        eval_name=['train'],
                        eval_metric=['auc'],
                        max_epochs=200,
                        patience=10,
                        batch_size=32,
                        virtual_batch_size=32,
                        num_workers=self.n_cpus,
                        weights=1,
                        drop_last=True)
                else:
                    model = model.fit(self.X_train, self.y_train)

                # save model
                self.save_model(model=model, file_path=model_filename)

                if self.shap_analysis:
                    self.plotter.plot_shap(model=model,
                                           X=self.X_train,
                                           y=self.y_train,
                                           output_path=self.shap_plot_save_dir + "/" + "stable_" + type(
                                               model).__name__,
                                           seed=self.rand_state)

                if len(self.X_val) == 0:
                    self.eval_model(model=model,
                                    X_test=self.X_test,
                                    y_test=self.y_test,
                                    X_val=None,
                                    y_val=None,
                                    val_pred=y_stable,
                                    model_name="stable_" + type(model).__name__)
                else:
                    self.eval_model(model=model,
                                    X_test=self.X_test,
                                    y_test=self.y_test,
                                    X_val=self.X_val,
                                    y_val=self.y_val,
                                    model_name="stable_" + type(model).__name__)

                self.logger.info("Save trained stable {} model.".format(type(model).__name__))

                # save model
                self.save_model(model=model, file_path=model_filename)

                # loader.stop()

                self.logger.info(
                    "Stable {} model performance with {} estimators and {} AUROC.".format(str(type(model).__name__),
                                                                                          str(x_stable),
                                                                                          str(round(y_stable, 3))))
                print(
                    "Stable {} model performance with {} estimators and {} AUROC.".format(str(type(model).__name__),
                                                                                          str(x_stable),
                                                                                          str(round(y_stable, 3))))
            else:
                print("Found trained model!")
                self.logger.info("Found trained model at " + str(model_filename))

                self.s.append(clone(model))
                model = self.load_model(model_filename)

                # x_max, y_max, x_stable, y_stable = self.get_stable_model_size(model, param_name, param_range)
                if len(self.X_val) == 0:
                    self.eval_model(model=model,
                                    X_test=self.X_test,
                                    y_test=self.y_test,
                                    X_val=None,
                                    y_val=None,
                                    val_pred=y_stable,
                                    model_name="stable_" + type(model).__name__)
                else:
                    self.eval_model(model=model,
                                    X_test=self.X_test,
                                    y_test=self.y_test,
                                    X_val=self.X_val,
                                    y_val=self.y_val,
                                    model_name="stable_" + type(model).__name__)

            self.pre_training_results.to_csv(overview_filename)

        else:
            print("Pretraining already performed.")

    def create_ensemble_models(self, list_of_models=None, fit_estimators=False):
        """
        Create ensemble model
        :param list_of_models: List of models to ensemble
        :param fit_estimators: Train models or not (takes fitted estimator in ensemble if False)
        :return EnsembleVoteClassifier: Soft voting, Hard voting
        """

        corrected_models = False
        corrected_list_of_models = []

        if corrected_models:
            list_of_models = corrected_list_of_models

        if not list_of_models is None:
            ensemble_soft = EnsembleVoteClassifier(clfs=list_of_models, voting='soft',
                                                   fit_base_estimators=fit_estimators)  # , use_clones=False)
            ensemble_hard = EnsembleVoteClassifier(clfs=list_of_models, voting='hard',
                                                   fit_base_estimators=fit_estimators)  # , use_clones=False)
        elif len(list_of_models) > 1:
            self.error.error(
                "For ensembling please give valid number of models. Number of given models for ensembling: " + str(
                    len(list_of_models)))
            print("For ensembling please give valid number of models. Number of given models for ensembling: " + str(
                len(list_of_models)))
            return None, None
            # raise ValueError("For ensembling please give valid number of models.")
        else:
            self.error.error("For ensembling please give models to ensemble. No list of models is given!")
            print("For ensembling please give models to ensemble. No list of models is given!")
            return None, None
            # raise ValueError("For ensembling please give models to ensemble.")

        return ensemble_soft, ensemble_hard

    def ensembling(self, list_of_models, model_name, out_folder, fit_estimators, get_ensambled_model_names = False):
        """
        Perform Ensembling of models in a list. Models can be fitted already.
        :param list_of_models: list of models to ensemble
        :param model_name: names of ensemble model
        :param out_folder: Folder to save the models
        :param fit_estimators: Fit the models or not
        :param get_ensambled_model_names: Get the names of the models in the ensemble
        """

        model_names_in_list = []
        ensemble_model_name = ""

        if out_folder.endswith("/"):
            out_folder = out_folder[:-1]

        os.makedirs(out_folder, exist_ok=True)

        for model in list_of_models:
            name_of_model = type(model).__name__

            if "Ensemble" in name_of_model:
                for e_model in model.clfs:
                    e_model_name = type(e_model).__name__

                    model_names_in_list.append(e_model_name)
            else:
                model_names_in_list.append(name_of_model)

        if len(set(model_names_in_list)) > 1:
            self.error.warning(
                "Ensembling of different models need to apply carefully based on different configuration. Ensembling: " + str(
                    set(model_names_in_list)))
            print(
                "Ensembling of different models need to apply carefully based on different configuration. Ensembling: " + str(
                    set(model_names_in_list)))
            
        else:
            ensemble_model_name = str(list(set(model_names_in_list))[0])

        ensemble_soft, ensemble_hard = self.create_ensemble_models(list_of_models=list_of_models,
                                                                   fit_estimators=fit_estimators)
        if not ensemble_soft is None:
            if self.check_tabnet_in_ensemble(ensemble_soft):
                ensemble_soft.fit(self.X_train.values, self.y_train.values)
                ensemble_hard.fit(self.X_train.values, self.y_train.values)

                self.eval_model(model=ensemble_soft,
                                X_test=self.X_test,
                                y_test=self.y_test,
                                model_name="SoftEnsemble_" + model_name + "_" + ensemble_model_name)

                self.eval_model(model=ensemble_hard,
                                X_test=self.X_test,
                                y_test=self.y_test,
                                model_name="HardEnsemble_" + model_name + "_" + ensemble_model_name)

            else:
                ensemble_soft = ensemble_soft.fit(self.X_train, self.y_train)
                ensemble_hard = ensemble_hard.fit(self.X_train, self.y_train)

                self.eval_model(model=ensemble_soft,
                                X_test=self.X_test,
                                y_test=self.y_test,
                                model_name="SoftEnsemble_" + model_name + "_" + ensemble_model_name)

                self.eval_model(model=ensemble_hard,
                                X_test=self.X_test,
                                y_test=self.y_test,
                                model_name="HardEnsemble_" + model_name + "_" + ensemble_model_name)

            ensemble_soft_out_folder = out_folder + "/SoftEnsemble_" + model_name + "_" + ensemble_model_name + "/"
            ensemble_hard_out_folder = out_folder + "/HardEnsemble_" + model_name + "_" + ensemble_model_name + "/"

            ensemble_soft_filename = ensemble_soft_out_folder + self.RunID + "_SoftEnsemble_" + model_name + '.sav'
            ensemble_hard_filename = ensemble_hard_out_folder + self.RunID + "_HardEnsemble_" + model_name + '.sav'

            os.makedirs(ensemble_soft_out_folder, exist_ok=True)
            os.makedirs(ensemble_hard_out_folder, exist_ok=True)
            
            self.save_model(model=ensemble_soft, file_path=ensemble_soft_filename)
            self.save_model(model=ensemble_hard, file_path=ensemble_hard_filename)
            
        if get_ensambled_model_names:
             return ensemble_soft, ensemble_hard, ensemble_model_name
        else:
            return ensemble_soft, ensemble_hard
    
    @staticmethod
    def polish_model_name(name):
        """
        Function to polish ensemble model names
        :param str: name in format nameModelClass_ModelName_
        """

        # Extract the base model name after the last "_"
        model_name = name.split("_")[-1]
        base_name = name.split("_")[0]
        # Return "SoftEnsemble_" + base model name
        return f"{base_name}_{model_name}"

    @staticmethod
    def select_top_models(mean_auroc_dict, top_n=3):
        """
        Selects the top N models based on mean validation AUROC.

        Parameters:
        - mean_auroc_dict (dict): Dictionary of model names and their mean validation AUROC.
        - top_n (int): Number of top models to select.

        Returns:
        - list: Top N models sorted by mean AUROC.
        """
        # Sort models by mean AUROC in descending order
        sorted_models = sorted(mean_auroc_dict.items(), key=lambda x: x[1][0], reverse=True)

        # Select top N models
        top_models = [model[0] for model in sorted_models[:top_n]]

        return top_models
    
    def get_ensemble_model_paths(self, selected_models):
        """
        Generates paths for the selected top models.

        Parameters:
        - selected_models (list): List of selected model names.

        Returns:
        - dict: Dictionary of model names and their corresponding paths.
        """

        if not self.out_folder.endswith("/"):
            base_path = self.out_folder + "/"
        else:
            base_path = self.out_folder
        
        model_paths = {}
        for model in selected_models:
            path = base_path + f"models/CV/optimized/{model}*/SoftEnsemble_{model}*_cv_optimized_{model}/"
            filenames = glob.glob(path + "*_cv_optimized.sav")

            if len(filenames) > 1:
                print(f"Found {len(filenames)} models: " + filenames)

            if len(filenames) == 0:
                self.error.error(f"No ensembling model found for {model} in {path}.")
                print(f"No ensembling model found for {model} in {path}.")
                continue
                # raise ValueError(f"No ensembling model found for {model} in {path}.")

            for filename in filenames:    
                model_paths[model] = filename

        return model_paths

    def plot_feature_importances(self, folder_path, output_folder):
        """
        Loads all feature importance CSV files, aggregates feature importances across folds for each model,
        and saves plots to the specified output folder:
        - Feature importance for each fold (if values exist).
        - Mean feature importance across all folds for each model.
        """
        os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
        model_feature_importances = {}  # Dictionary to store feature importances per model
        csv_files = glob.glob(os.path.join(folder_path, "optimized_*_fold_*_feature_importances.csv"))

        for file in csv_files:
            filename = os.path.basename(file)
            model_name = filename.split('_fold_')[0].replace("optimized_", "")
            fold_number = filename.split('_fold_')[1].split('_')[0]  # Extract fold number

            df = pd.read_csv(file)
            feature_col = df.columns[0]  # First column (feature names)
            importance_col = df.columns[1]  # Second column (importance values)

            # Check if importance values exist
            if df[importance_col].isna().all():
                print(f"Warning: No feature importance values found for {filename}, skipping plot.")
                continue

            # Clean feature names by removing '_zscore' if present
            df[feature_col] = df[feature_col].str.replace("_zscore", "", regex=True)

            if model_name not in model_feature_importances:
                model_feature_importances[model_name] = []
            
            model_feature_importances[model_name].append(df.set_index(feature_col)[importance_col])

            # Save feature importance plot for this fold
            plt.figure(figsize=(12, max(6, len(df) / 5)))  # Increase width to accommodate long titles
            df.set_index(feature_col)[importance_col].sort_values(ascending=True).plot(kind='barh')
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Features")
            plt.title(f"{model_name} - Feature Importance (Fold {fold_number})", fontsize=12)
            plt.tight_layout()  # Ensure layout fits
            plt.savefig(os.path.join(output_folder, f"{model_name}_fold_{fold_number}_importance.png"), bbox_inches='tight')
            plt.close()

        # Save mean feature importance for each model
        for model_name, importance_list in model_feature_importances.items():
            importance_df = pd.concat(importance_list, axis=1).fillna(0)
            mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
            
            plt.figure(figsize=(12, max(6, len(mean_importance) / 5)))  # Increase width for long titles
            mean_importance.sort_values(ascending=True).plot(kind='barh')
            plt.xlabel("Mean Feature Importance Score")
            plt.ylabel("Features")
            plt.title(f"{model_name} - Mean Feature Importance Across Folds", fontsize=12)
            plt.tight_layout()  # Ensure layout fits
            plt.savefig(os.path.join(output_folder, f"{model_name}_mean_importance.png"), bbox_inches='tight')
            plt.close()

    def train(self):
        """
        The main method to train a classifier.
        Procedure:
        1. Split data
        2. Load models
        3. Get the stable and efficient model size (n_estimators ...)
        4. Optimize Model Hyperparameter in CV fashion
        5. Train model with optimized hyperparameter and eval on val set
        6. Test model on test set
        7. Save Model
        8. Apply an Ensemble of all models on the test set
        """

        # 1. Split data
        # 2. Load models
        # 3. Get the stable and efficient model size (n_estimators ...)
        # 4. Optimize Model Hyperparameter in CV fashion
        # 5. Train model with optimized hyperparameter and eval on val set
        # 6. Test model on test set
        # 7. Save Model
        # 8. Apply an Ensemble of all models on the test set

        y_stable = None
        x_stable = None

        # only want to predict with a trained model
        if self.predict_only:
            # set seed
            self.random_control()

            # format input data
            self.data = self.config_input_data(data=self.data.copy())

            if self.data.index.name != "ID":
                if "ID" in self.data.columns:
                    self.data.set_index("ID", inplace=True)
                else:
                    self.error.error("ID is missing in input data.")
                    raise ValueError("ID is missing in input data.")

            imbalance = self.predict_class_imbalance(self.data, self.Prediction_Label)
            if imbalance["is_significant"]:
                print(f"Warning - Significant class imbalance detected!")
                self.error.warning(f"Significant class imbalance detected!")

            self.X = self.data.loc[:, self.data.columns != self.Prediction_Label]
            
            self.X = self.X.loc[:,~self.X.columns.duplicated()]

            self.feature_format_check()

            self.y = self.data[self.Prediction_Label]

            # Prediction only takes all the samples from the input data to predict and not training
            self.X_train = pd.DataFrame()
            self.y_train = pd.DataFrame()

            self.X_test = self.X
            self.y_test = self.y
            
             # Load trained model from model_path
            self.generate_model_instance()

            self.verify_task()

            self.verify_scoring_matrics()

            # predict only with trained model
            self.plotter = PerformancePlotter(output_path=self.plot_save_dir,
                                            X=self.X_test,
                                            y=self.y_test,
                                            RunID=self.RunID,
                                            error=self.error,
                                            logger=self.logger)

            for model in self.model:
                model_name = type(model).__name__

                self.eval_model(model=model,
                            X_test=self.X_test,
                            y_test=self.y_test,
                            X_val=None,
                            y_val=None,
                            model_name="predict_only_" + model_name)
                        
        else:
            # generate train test val sets
            self.split_data()
            
            # save train and test data for feature differences between model trainings
            self.X_train_global = self.X_train
            self.X_test_global = self.X_test
            self.X_val_global = self.X_val

            # get a model if there is no model specified and no parameters are given create a new model
            self.generate_model_instance()

            self.plotter = PerformancePlotter(output_path=self.plot_save_dir,
                                            X=self.X_train,
                                            y=self.y_train,
                                            RunID=self.RunID,
                                            error=self.error,
                                            logger=self.logger)

            if isinstance(self.model, list):
                print("Training", len(self.model), "different models ... ")


            if self.plot_save_dir is None:
                self.plot_save_dir = self.out_folder + "/plots"
                self.logger.info(
                    "Model save directory not specified. Using default directory: {}".format(self.plot_save_dir))

                if not os.path.exists(self.plot_save_dir):
                    os.makedirs(self.plot_save_dir,exist_ok=True)

            if self.shap_plot_save_dir is None:
                self.shap_plot_save_dir = self.plot_save_dir + "/SHAP"

            if not os.path.exists(self.shap_plot_save_dir):
                os.makedirs(self.shap_plot_save_dir,exist_ok=True)


            # 3. Get the stable and efficient model size (n_estimators ...)
            if self.stable_pretraining:
                overview_filename = self.out_folder + "/Pretrained_model_overview_" + str(self.RunID) + '.csv'

                if os.path.exists(str(overview_filename)):
                    self.pre_training_results = pd.read_csv(overview_filename, index_col=0)

                self.stable_models = []

                for model in self.model:
                    print(3 * "#", "Pretraining {}".format(type(model).__name__))
                    self.perform_stable_pretraining(model, overview_filename)

            # Training model/s
            trained_models = {}
            if isinstance(self.model, list):

                if self.model_names is None:
                    self.model_names = []

                    for model in self.model:
                        model_name = type(model).__name__
                        self.model_names.append(model_name)

                if self.ensemble:
                    self.ensemble_df = pd.DataFrame({"Model_name": [], "File_path": [], "Type": []})

                for clf, label in zip(self.model, self.model_names):
                    # if "Ensemble" not in label:
                    print(3 * "#", "Training", label, 3 * "#")
                    if self.use_cross_validation:
                        trained_models[label] = self.train_cv(model=clf, model_name=label, plotter=self.plotter,
                                                            optimize=self.optimize)
                    else:
                        trained_models[label] = self.train_model(clf=clf, label=label)
            else:
                model_name = str(type(self.model).__name__)
                print(3 * "#", "Training", model_name, 3 * "#")

                if self.use_cross_validation:
                    trained_models[model_name] = self.train_cv(model=self.model, model_name=model_name,
                                                            plotter=self.plotter, optimize=self.optimize)
                else:
                    trained_models[model_name] = self.train_model(clf=self.model, label=model_name)


            val_auc_fold_models, test_auc_fold_models, test_auc_ensemble_models = PerformancePlotter.get_AUC_values_per_models(self.prediction_summary.copy(), number_of_folds=self.cross_val_splits)
            val_auc_fold_models_mean = {model: [np.mean(values)] for model, values in val_auc_fold_models.items()} 

            # Get Best Model
            best_model = ModelTrainer.select_top_models(mean_auroc_dict=val_auc_fold_models_mean, top_n=1) 
            

            best_model_path = self.get_ensemble_model_paths(selected_models=best_model)
            best_model_name = best_model[0]
            model = self.load_model(best_model_path[best_model_name])

            integrated_model_name = ""
            if "Ensemble" in type(model).__name__:
                if integrated_model_name == "":
                    integrated_model_name = type(model.clfs[0]).__name__ 
                else:
                    integrated_model_name += "_" + type(model.clfs[0]).__name__
            else:
                integrated_model_name += type(model).__name__ + "_"

            print(f"Selected Best model based on validation AUROC: {integrated_model_name}")
            self.logger.info(f"Selected Best model based on validation AUROC: {integrated_model_name}")

            if self.ensemble_best_models:
                # Getting the top models based on the validation AUROC
                top_models = ModelTrainer.select_top_models(mean_auroc_dict=val_auc_fold_models_mean, top_n=self.ensemble_best_n_models)      
                self.logger.info(f"Top models based on validation AUROC: {top_models}")

                self.logger.info("Ensembling the top models ...")

                # ensembling the top models
                model_paths = self.get_ensemble_model_paths(selected_models=top_models)

                # load models
                models_4_ensemble = []
                integrated_model_names = ""
                for model_name in model_paths:
                    
                    model = self.load_model(model_paths[model_name])
                    if "Ensemble" in type(model).__name__:
                        if integrated_model_names == "":
                            integrated_model_names = type(model.clfs[0]).__name__ 
                        else:
                            integrated_model_names += "_" + type(model.clfs[0]).__name__
                    else:
                        integrated_model_names += type(model).__name__ + "_"

                    print("Included model:",type(model).__name__)
                    models_4_ensemble.append(model)

                print("Ensembling the top models: ", models_4_ensemble)
                if None in models_4_ensemble:
                    print("Error in loading models for ensembling.")

                # ensembling models
                ensemble_soft, ensemble_hard = self.ensembling(list_of_models=models_4_ensemble, 
                                                                model_name= "Best_Ensemble_" + integrated_model_names, 
                                                                out_folder=self.model_save_dir + "/CV/optimized/best_ensemble/", 
                                                                fit_estimators=False, 
                                                                get_ensambled_model_names=False)
                
                self.eval_model(model=ensemble_soft, 
                                X_test=self.X_test, 
                                y_test=self.y_test, 
                                X_val=None, 
                                y_val=None, 
                                model_name="Best_Ensemble_" + integrated_model_names)

            # check all models and create final plots to evaluate

            # get all Ensembled approaches
            enemble_df = self.prediction_summary.copy()[self.prediction_summary.copy()['Models'].str.contains("SoftEnsemble_", case=False, na=False)]
            stable_df = self.prediction_summary.copy()[self.prediction_summary.copy()['Models'].str.contains("stable_", case=False, na=False)]

            # Apply the function to all model names
            enemble_df.loc[:,"Models"] = [ModelTrainer.polish_model_name(name=name) for name in enemble_df["Models"].to_list()]

            # plot Performance distribution for all Ensembl models with Concidence interval
            PerformancePlotter.plot_model_auc_ci_vertical(df=enemble_df,
                                                        filename=self.out_folder + "/plots/Ensemble_CI_dist_plot.png")

            PerformancePlotter.plot_model_auc_ci_vertical(df=stable_df,
                                                        filename=self.out_folder + "/plots/Stable_CI_dist_plot.png")

            val_auc_fold_models, test_auc_fold_models, test_auc_ensemble_models = PerformancePlotter.get_AUC_values_per_models(self.prediction_summary.copy(), number_of_folds=self.cross_val_splits)
            best_model, best_AUC  = PerformancePlotter.plot_val_auc_distribution(val_auc_dict=val_auc_fold_models, save_path=self.out_folder + "/plots/CV_Val_AUC_distribution.png")
            
            print(f"Ensemble Test Performance of best model {best_model} with Val AUC: {round(best_AUC,3)}: {test_auc_ensemble_models[best_model]}")
            self.logger.info(f"Ensemble Test Performance of best model {best_model} with Val AUC: {round(best_AUC,3)}: {test_auc_ensemble_models[best_model]}")        
            

        val_auc_fold_models, test_auc_fold_models, test_auc_ensemble_models = PerformancePlotter.get_AUC_values_per_models(self.prediction_summary.copy(), number_of_folds=self.cross_val_splits)
        
        PerformancePlotter.plot_summary_auroc(val_auc_fold_models=val_auc_fold_models, 
                                              test_auc_fold_models=test_auc_fold_models, 
                                              test_auc_ensemble_models=test_auc_ensemble_models, 
                                              data=self.prediction_summary.copy(), 
                                              output_path=self.out_folder + "/plots/")

        
        subfolders = [ f.path for f in os.scandir(self.out_folder + "/feature_importance") if f.is_dir() ]
        for folder in subfolders:
            model_folder_name = os.path.basename(folder)
            # Plot feature impotances
            self.plot_feature_importances(folder_path=folder, output_folder=self.out_folder + "/plots/feature_importances/" + model_folder_name + "/")

        print("RPTK Prediction Done!")
        self.logger.info("#### RPTK Prediction Done! ####")
