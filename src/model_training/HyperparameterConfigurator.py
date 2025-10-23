import numpy as np
import pandas as pd
import glob
import os
import optuna
import math

from rptk.src.config.Log_generator_config import LogGenerator


class HyperparameterConfigurator:
    """
    Class to give sets of hyperparameters for each model to optimize. Main function is get_hyperparameters to get the correct set of hyperparmeters with the correct optimizer
    :param model: Model you want hyperparaeters for
    :param optimizer_lib: Algorithm you want to optimize hyperparameter with (sklearn, or optuna)
    :param X: dataset without label for parameter estimation
    :param logger: info logger for logging processing
    :param error: error logger for logging error messages
    :param out_path: Path to output files
    :param interested_hyperparameter: list of hyperparameter names which are interesting for optimization
    :param extended_parameter_set: bool if you want to use more hyperparameter values (smaller interval)
    :param train_all_at_once: bool if you want to train the model size (num trees) first (false) or with all parameters together (true)
    """

    def __init__(self,
                 model=None,
                 optimizer_lib=None,
                 X=None,
                 logger=None,
                 error=None,
                 out_path="",
                 interested_hyperparameter=None,
                 extended_parameter_set=False,
                 train_all_at_once=True,
                 RunID=None):

        if interested_hyperparameter is None:
            interested_hyperparameter = []

        self.model = model
        self.optimizer_lib = optimizer_lib
        self.X = X
        self.logger = logger
        self.error = error
        self.out_path = out_path
        self.interested_hyperparameter = interested_hyperparameter
        self.extended_parameter_set = extended_parameter_set
        self.train_all_at_once = train_all_at_once
        self.RunID = RunID

        # define internal variables
        self.fit_params = {}

        self.available_optimizer = ["sklearn", "optuna"]

        self.available_models = ["RandomForestClassifier",
                                 "GradientBoostingClassifier",
                                 "LGBMClassifier",
                                 "XGBClassifier",
                                 "TabNetClassifier"]

        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.out_path + "/RPTK_hyperparameter_" + self.RunID + ".log",
                logger_topic="RPTK Hyperparameter Config"
            ).generate_log()

        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.out_path + "/RPTK_hyperparameter_" + self.RunID + ".err",
                logger_topic="RPTK Hyperparameter Config error"
            ).generate_log()

    def optuna_archive(self, model_name=None):
        """
        Configure hyperparameter with using optuna optimizers
        :param model_name: name of the model to select the parameters for
        :return hyperparameter: dict with set of hyperparameter
        """

        if model_name is None:
            model_name = self.get_model_name(model=self.model)

        hyperparameter = {}

        if model_name == "RandomForestClassifier":
            self.fit_params["eval_metric"] = "roc_auc"
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),  # Number of Trees for the forest
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "max_samples": trial.suggest_float('max_samples', 0.2, 0.8, step=0.01),
                        "max_features": trial.suggest_categorical("max_features",
                                                                  ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.2, 0.8, step=0.1),
                        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                        "class_weight": trial.suggest_categorical("class_weight",
                                                                  [None, "balanced", "balanced_subsample"]),
                        "criterion": trial.suggest_categorical("gini", "entropy"),
                    }
                else:
                    hyperparameter = {
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "max_samples": trial.suggest_float('max_samples', 0.2, 0.8, step=0.01),
                        "max_features": trial.suggest_categorical("max_features",
                                                                  ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.2, 0.8, step=0.1),
                        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                        "class_weight": trial.suggest_categorical("class_weight",
                                                                  [None, "balanced", "balanced_subsample"]),
                        "criterion": trial.suggest_categorical("gini", "entropy"),
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                    }
                else:
                    hyperparameter = {
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 10, 30),  # Max depth of a tree
                    }

        elif model_name == "GradientBoostingClassifier":
            self.fit_params["eval_metric"] = "roc_auc"
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),  # Number of Trees for the forest
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "validation_fraction": trial.suggest_float('validation_fraction', 0.1, 0.4, step=0.01),
                        "max_features": trial.suggest_categorical("max_features",
                                                                  ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001),
                        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
                        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 0, 100),
                        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes",
                                                                    [self.generate_max_leaf_nodes(self.X)]),
                        "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
                        "init": trial.suggest_categorical("init", [None, self.model]),
                        "warm_start": trial.suggest_categorical("warm_start", [False, True]),
                        "tol": trial.suggest_float("tol", 1e-4, 0.3, step=0.01),
                    }
                else:
                    hyperparameter = {
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "validation_fraction": trial.suggest_float('validation_fraction', 0.1, 0.4, step=0.01),
                        "max_features": trial.suggest_categorical("max_features",
                                                                  ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001),
                        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
                        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 0, 100),
                        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes",
                                                                    [self.generate_max_leaf_nodes(self.X)]),
                        "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
                        "init": trial.suggest_categorical("init", [None, self.model]),
                        "warm_start": trial.suggest_categorical("warm_start", [False, True]),
                        "tol": trial.suggest_float("tol", 1e-4, 0.3, step=0.01),
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001)
                    }
                else:
                    hyperparameter = {
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001)
                    }

        elif model_name == "LGBMClassifier":
            self.fit_params = {"eval_metric": "auc"}
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),  # number of boosted trees to fit
                        'num_leaves': trial.suggest_int('num_leaves', 20, 70),  # max number of leaves in one tree
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', int(self.X.shape[0] * 0.1),
                                                              int(self.X.shape[0] * 0.2)),  # min 10% or 20% of the data
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.025, step=0.001),
                        'max_bin': trial.suggest_int('max_bin', 40, 100),
                        # max number of bins that feature values will be bucketed in
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0, step=0.1),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.2, step=0.01),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.2, step=0.01),
                        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.4, step=0.01),

                        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 11),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.99, step=0.1),
                        'verbosity': -1,
                    }
                else:
                    hyperparameter = {
                        'num_leaves': trial.suggest_int('num_leaves', 20, 70),  # max number of leaves in one tree
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', int(self.X.shape[0] * 0.1),
                                                              int(self.X.shape[0] * 0.2)),  # min 10% or 20% of the data
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.025, step=0.001),
                        'max_bin': trial.suggest_int('max_bin', 40, 100),
                        # max number of bins that feature values will be bucketed in
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0, step=0.1),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.2, step=0.01),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.2, step=0.01),
                        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.4, step=0.01),

                        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 11),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.99, step=0.1),
                        'verbosity': -1,
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', int(self.X.shape[0] * 0.1),
                                                              int(self.X.shape[0] * 0.2)),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, step=0.01),
                        'num_leaves': trial.suggest_int('num_leaves', 40, 80),
                        'verbosity': -1,
                        'first_metric_only': True
                    }
                else:
                    hyperparameter = {
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', int(self.X.shape[0] * 0.1),
                                                              int(self.X.shape[0] * 0.2)),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, step=0.01),
                        'num_leaves': trial.suggest_int('num_leaves', 40, 80),
                        'verbosity': -1,
                        'first_metric_only': True
                    }
        elif model_name == "XGBClassifier":
            self.fit_params["eval_metric"] = "roc_auc"
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),  # number of boosted trees to fit
                        'booster': trial.suggest_categorical('booster', ["gbtree", "gblinear"]),
                        # Select the type of model to run at each iteration
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                        'gamma': trial.suggest_int('gamma', 0, 4),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
                        'max_delta_step': trial.suggest_int('max_delta_step', 0, 3),
                        'subsample': trial.suggest_float('subsample', 0.6, 0.8, step=0.1),
                        # 'sampling_method':["uniform","gradient_based"],
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8, step=0.1),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.8, step=0.1),
                        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.1),
                        'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                        'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                        'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
                    }
                else:
                    hyperparameter = {
                        'booster': trial.suggest_categorical('booster', ["gbtree", "gblinear"]),
                        # Select the type of model to run at each iteration
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                        'gamma': trial.suggest_int('gamma', 0, 4),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
                        'max_delta_step': trial.suggest_int('max_delta_step', 0, 3),
                        'subsample': trial.suggest_float('subsample', 0.6, 0.8, step=0.1),
                        # 'sampling_method':["uniform","gradient_based"],
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8, step=0.1),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.8, step=0.1),
                        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.1),
                        'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                        'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                        'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                        'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                    }
                else:
                    hyperparameter = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                        'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                    }

        elif model_name == "TabNetClassifier":
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_d': trial.suggest_int('n_d', 100, 200),
                        'n_a': trial.suggest_int('n_a', 2, 100),
                        'n_steps': trial.suggest_int('n_steps', 3, 10),  # Number of steps in the architecture
                        'gamma': trial.suggest_float('gamma', 1.0, 2.0, step=0.1),
                        # This is the coefficient for feature reusage in the masks.
                        'n_independent': trial.suggest_int('n_independent', 1, 4),
                        # Number of independent Gated Linear Units layers at each step.
                        'n_shared': trial.suggest_int('n_shared', 1, 4),
                        # Number of shared Gated Linear Units at each step
                        'momentum': trial.suggest_float('momentum', 0.02, 0.2, step=0.01),
                        'lambda_sparse': trial.suggest_float('lambda_sparse', 0.001, 0.1, step=0.0001),
                        'mask_type': trial.suggest_categorical('mask_type', ['entmax', "sparsemax"]),
                        'lr': trial.suggest_float('lr', 0.005, 0.025, step=0.001),
                    }
                else:
                    hyperparameter = {
                        'n_a': trial.suggest_int('n_a', 2, 100),
                        'n_steps': trial.suggest_int('n_steps', 3, 10),  # Number of steps in the architecture
                        'gamma': trial.suggest_float('gamma', 1.0, 2.0, step=0.1),
                        # This is the coefficient for feature reusage in the masks.
                        'n_independent': trial.suggest_int('n_independent', 1, 4),
                        # Number of independent Gated Linear Units layers at each step.
                        'n_shared': trial.suggest_int('n_shared', 1, 4),
                        # Number of shared Gated Linear Units at each step
                        'momentum': trial.suggest_float('momentum', 0.02, 0.2, step=0.01),
                        'lambda_sparse': trial.suggest_float('lambda_sparse', 0.001, 0.1, step=0.0001),
                        'mask_type': trial.suggest_categorical('mask_type', ['entmax', "sparsemax"]),
                        'lr': trial.suggest_float('lr', 0.005, 0.025, step=0.001),
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_d': trial.suggest_int('n_d', 100, 200),
                        'n_a': trial.suggest_int('n_a', 2, 100),
                        'lr': trial.suggest_float('lr', 0.005, 0.025, step=0.001),
                    }
                else:
                    hyperparameter = {
                        'n_a': trial.suggest_int('n_a', 2, 100),
                        'lr': trial.suggest_float('lr', 0.005, 0.025, step=0.001),
                    }

        return hyperparameter

    def sklearn_archive(self, model=None):
        """
        Configure hyperparameters with using the sklearn optimizers
        :param model: model to get hyperparameter for
        :return hyperparameters: dict with hyperparameter
        """

        hyperparameter = {}

        if model is None:
            model = self.model

        model_name = self.get_model_name(model=model)

        if model_name == "RandomForestClassifier":
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        ## numerical ##
                        "n_estimators": list(range(100, 1000, 100)),  # Number of Trees for the forest
                        "ccp_alpha": [0.001, 0.01, 0.02, 0.03],
                        "max_depth": list(range(5, self.round_down_to_nearest_10(len(self.X) / 2), 1)),
                        # Max depth of a tree
                        "max_features": list(range(self.round_down_to_nearest_10(self.X.shape[1] / 2),
                                                   self.round_down_to_nearest_10(self.X.shape[1]), 5)),
                        # Number of features to consider for the best split
                        "max_samples": [None, 0.2, 0.4, 0.6, 0.8],
                        # If bootstrap is True, the number of samples to draw from X to train each base estimator.
                        "min_samples_split": [2, 0.2, 0.4, 0.6, 0.8, 1.0],
                        # The minimum number of samples required to split an internal node
                        "min_samples_leaf": [1, 0.2, 0.3, 0.4],
                        # Minimum number of samples required at a leaf node. --> smoothing the mode

                        # categorical
                        "criterion": ["gini", "entropy"],
                        "bootstrap": [True, False],
                        # Bootstrap samples when building trees or use whole dataset to build each tree.
                        "class_weight": [None, "balanced", "balanced_subsample"]
                        # “balanced” adjust weights inversely proportional to class frequencies balanced_subsample” do “balanced” on the bootstrap sample for every tree grown.
                    }
                else:
                    hyperparameter = {
                        ## numerical ##
                        "ccp_alpha": [0.001, 0.01, 0.02, 0.03],
                        "max_depth": list(range(5, self.round_down_to_nearest_10(len(self.X) / 2), 1)),
                        # Max depth of a tree
                        "max_features": list(range(self.round_down_to_nearest_10(self.X.shape[1] / 2),
                                                   self.round_down_to_nearest_10(self.X.shape[1]), 5)),
                        # Number of features to consider for the best split
                        "max_samples": [None, 0.2, 0.4, 0.6, 0.8],
                        # If bootstrap is True, the number of samples to draw from X to train each base estimator.
                        "min_samples_split": [2, 0.2, 0.4, 0.6, 0.8, 1.0],
                        # The minimum number of samples required to split an internal node
                        "min_samples_leaf": [1, 0.2, 0.3, 0.4],
                        # Minimum number of samples required at a leaf node. --> smoothing the mode

                        # categorical
                        "criterion": ["gini", "entropy"],
                        "bootstrap": [True, False],
                        # Bootstrap samples when building trees or use whole dataset to build each tree.
                        "class_weight": [None, "balanced", "balanced_subsample"]
                        # “balanced” adjust weights inversely proportional to class frequencies balanced_subsample” do “balanced” on the bootstrap sample for every tree grown.
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        ## numerical ##
                        "n_estimators": [100, 400, 800, 1000],
                        "ccp_alpha": [0.001, 0.01, 0.05, 0.1],
                        'max_depth': [None, 2, 5, 7, 10],  # Max depth of a tree
                        "max_samples": [0.2, 0.5, 0.8],
                        "max_features": ["sqrt", None, 0.2, 0.6, 1.0],
                        # categorical
                        "criterion": ["gini", "entropy"],
                    }
                else:
                    hyperparameter = {
                        ## numerical ##
                        "ccp_alpha": [0.001, 0.01, 0.05, 0.1],
                        'max_depth': [None, 2, 5, 7, 10],  # Max depth of a tree
                        "max_samples": [0.2, 0.5, 0.8],
                        "max_features": ["sqrt", None, 0.2, 0.6, 1.0],
                        # categorical
                        "criterion": ["gini", "entropy"],
                    }
        elif model_name == "GradientBoostingClassifier":
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        ## numerical ##
                        "n_estimators": list(range(100, 800, 100)),  # Number of Trees for the forest
                        "ccp_alpha": [0.0, 0.001, 0.01, 0.02, 0.03],
                        # Complexity parameter for Minimal Cost-Complexity Pruning (subtree with largest cost complexity that is smaller than ccp_alpha will be chosen)
                        "validation_fraction": [0.1, 0.2, 0.3, 0.4],
                        # The proportion of training data to set aside as validation set for early stopping
                        "learning_rate": [0.005, 0.01, 0.02, 0.025],
                        # Learning rate shrinks the contribution of each tree
                        "subsample": [1.0, 0.2, 0.4, 0.6, 0.8],
                        # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
                        "n_iter_no_change": [None, 20, 40, 60, 80, 100],
                        # number of iterations after val loss is not improving
                        "tol": [1e-4, 0.1, 0.2, 0.3],
                        # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations
                        'max_depth': [None, 2, 4, 6, 8, 10],  # Max depth of a tree
                        "max_features": ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0],
                        "max_leaf_nodes": self.generate_max_leaf_nodes(self.X),

                        # categorical
                        "loss": ["log_loss", "exponential"],  # The loss function to be optimized.
                        "criterion": ["friedman_mse", "squared_error"],
                        "warm_start": [False, True],
                        # Reuse solution of the previous call to fit and add more estimators to the ensemble or fit new forest
                        "init": [None, model]
                    }
                else:
                    hyperparameter = {
                        ## numerical ##
                        "ccp_alpha": [0.0, 0.001, 0.01, 0.02, 0.03],
                        # Complexity parameter for Minimal Cost-Complexity Pruning (subtree with largest cost complexity that is smaller than ccp_alpha will be chosen)
                        "validation_fraction": [0.1, 0.2, 0.3, 0.4],
                        # The proportion of training data to set aside as validation set for early stopping
                        "learning_rate": [0.005, 0.01, 0.02, 0.025],
                        # Learning rate shrinks the contribution of each tree
                        "subsample": [1.0, 0.2, 0.4, 0.6, 0.8],
                        # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
                        "n_iter_no_change": [None, 20, 40, 60, 80, 100],
                        # number of iterations after val loss is not improving
                        "tol": [1e-4, 0.1, 0.2, 0.3],
                        # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations
                        'max_depth': [None, 2, 4, 6, 8, 10],  # Max depth of a tree
                        "max_features": ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0],
                        "max_leaf_nodes": self.generate_max_leaf_nodes(self.X),

                        # categorical
                        "loss": ["log_loss", "exponential"],  # The loss function to be optimized.
                        "criterion": ["friedman_mse", "squared_error"],
                        "warm_start": [False, True],
                        # Reuse solution of the previous call to fit and add more estimators to the ensemble or fit new forest
                        "init": [None, model]
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        ## numerical ##
                        "n_estimators": [100, 300, 600, 1000],  # Number of Trees for the forest
                        "ccp_alpha": [0.0, 0.01, 0.03],
                        # Complexity parameter for Minimal Cost-Complexity Pruning (subtree with largest cost complexity that is smaller than ccp_alpha will be chosen)
                        "validation_fraction": [0.1, 0.2, 0.4],
                        # The proportion of training data to set aside as validation set for early stopping
                        "learning_rate": [0.005, 0.02, 0.03],
                        # Learning rate shrinks the contribution of each tree
                        "subsample": [1.0, 0.2, 0.5, 0.8],
                        # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
                        "n_iter_no_change": [None, 20, 60, 100],
                        # number of iterations after val loss is not improving
                        "tol": [1e-4, 0.1, 0.3],
                        # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations
                        'max_depth': [None, 4, 8, 12],  # Max depth of a tree
                        # categorical
                        "loss": ["log_loss", "exponential"],  # The loss function to be optimized.
                        # Reuse solution of the previous call to fit and add more estimators to the ensemble or fit new forest
                        "init": [None, model]
                    }
                else:
                    hyperparameter = {
                        ## numerical ##
                        "ccp_alpha": [0.0, 0.01, 0.03],
                        # Complexity parameter for Minimal Cost-Complexity Pruning (subtree with largest cost complexity that is smaller than ccp_alpha will be chosen)
                        "validation_fraction": [0.1, 0.2, 0.4],
                        # The proportion of training data to set aside as validation set for early stopping
                        "learning_rate": [0.005, 0.02, 0.03],
                        # Learning rate shrinks the contribution of each tree
                        "subsample": [1.0, 0.2, 0.5, 0.8],
                        # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
                        "n_iter_no_change": [None, 20, 60, 100],
                        # number of iterations after val loss is not improving
                        "tol": [1e-4, 0.1, 0.3],
                        # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations
                        'max_depth': [None, 4, 8, 12],  # Max depth of a tree
                        # categorical
                        "loss": ["log_loss", "exponential"],  # The loss function to be optimized.
                        # Reuse solution of the previous call to fit and add more estimators to the ensemble or fit new forest
                        "init": [None, model]
                    }
        elif model_name == "LGBMClassifier":
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_estimators': list(range(100, 800, 100)),  # number of boosted trees to fit
                        'min_data_in_leaf': [20, 30, 40],
                        # minimal number of data in one leaf. Can be used to deal with over-fitting
                        'num_iterations': [100, 150, 200],
                        'max_bin': [40, 80, 100],  # max number of bins that feature values will be bucketed in
                        'lambda_l1': [0.0, 0.2, 0.4],
                        'lambda_l2': [0.0, 0.2, 0.4],
                        'min_gain_to_split': [0.0, 0.2, 0.4],
                        'boosting_type': ['gbdt', 'dart', 'rf'],
                        'num_leaves': sp.stats.randint(2, 1001),
                        'subsample_for_bin': sp.stats.randint(10, 1001),
                        'min_split_gain': sp.stats.uniform(0, 5.0),
                        'min_child_weight': sp.stats.uniform(1e-6, 1e-2),
                        'reg_alpha': sp.stats.uniform(0, 1e-2),
                        'reg_lambda': sp.stats.uniform(0, 1e-2),
                        'tree_learner': ['data', 'feature', 'serial', 'voting'],
                        'bagging_freq': [1, 2, 5, 8, 11],
                        'bagging_fraction': [0.4, 0.6, 0.99],
                        'feature_fraction': sp.stats.uniform(1e-3, 0.99),
                        'learning_rate': sp.stats.uniform(1e-6, 0.99),
                        'max_depth': sp.stats.randint(1, 501),
                    }
                else:
                    hyperparameter = {
                        'n_estimators': list(range(100, 800, 100)),  # number of boosted trees to fit
                        'min_data_in_leaf': [20, 30, 40],
                        # minimal number of data in one leaf. Can be used to deal with over-fitting
                        'num_iterations': [100, 150, 200],
                        'max_bin': [40, 80, 100],  # max number of bins that feature values will be bucketed in
                        'lambda_l1': [0.0, 0.2, 0.4],
                        'lambda_l2': [0.0, 0.2, 0.4],
                        'min_gain_to_split': [0.0, 0.2, 0.4],
                        'boosting_type': ['gbdt', 'dart', 'rf'],
                        'num_leaves': sp.stats.randint(2, 1001),
                        'subsample_for_bin': sp.stats.randint(10, 1001),
                        'min_split_gain': sp.stats.uniform(0, 5.0),
                        'min_child_weight': sp.stats.uniform(1e-6, 1e-2),
                        'reg_alpha': sp.stats.uniform(0, 1e-2),
                        'reg_lambda': sp.stats.uniform(0, 1e-2),
                        'tree_learner': ['data', 'feature', 'serial', 'voting'],
                        'bagging_freq': [1, 2, 5, 8, 11],
                        'bagging_fraction': [0.4, 0.6, 0.99],
                        'feature_fraction': sp.stats.uniform(1e-3, 0.99),
                        'learning_rate': sp.stats.uniform(1e-6, 0.99),
                        'max_depth': sp.stats.randint(1, 501),
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        "n_estimators": [100, 300, 600, 1000],
                        'num_leaves': [20, 30],  # max number of leaves in one tree
                        'max_depth': [-1, 5, 7, 10, 15],
                        'min_data_in_leaf': [20, 30, 40],
                        # minimal number of data in one leaf. Can be used to deal with over-fitting
                        'num_iterations': [100, 150, 200],
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                        'max_bin': [40, 80, 100],  # max number of bins that feature values will be bucketed in
                        'feature_fraction': [0.4, 0.6, 0.8, 1.0],
                        'lambda_l1': [0.0, 0.2, 0.4],
                        'lambda_l2': [0.0, 0.2, 0.4],
                        'min_gain_to_split': [0.0, 0.2, 0.4]
                    }
                else:
                    hyperparameter = {
                        'num_leaves': [20, 30],  # max number of leaves in one tree
                        'max_depth': [-1, 5, 7, 10, 15],
                        'min_data_in_leaf': [20, 30, 40],
                        # minimal number of data in one leaf. Can be used to deal with over-fitting
                        'num_iterations': [100, 150, 200],
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                        'max_bin': [40, 80, 100],  # max number of bins that feature values will be bucketed in
                        'feature_fraction': [0.4, 0.6, 0.8, 1.0],
                        'lambda_l1': [0.0, 0.2, 0.4],
                        'lambda_l2': [0.0, 0.2, 0.4],
                        'min_gain_to_split': [0.0, 0.2, 0.4]
                    }

        elif model_name == "XGBClassifier":
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_estimators': list(range(100, 800, 100)),
                        'booster': ["gbtree", "gblinear"],  # Select the type of model to run at each iteration
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                        'gamma': [0, 2, 4],
                        'max_depth': [6, 4, 8],
                        'min_child_weight': [1, 2, 3],
                        'max_delta_step': [0, 1, 2, 3],
                        'subsample': [1, 0.7, 0.8],
                        # 'sampling_method':["uniform","gradient_based"],
                        'colsample_bytree': [1, 0.5, 0.8],
                        'colsample_bylevel': [1, 0.5, 0.8],
                        'colsample_bynode': [1, 0.5, 0.8],
                        'lambda': [1, 0.5, 0.8],
                        # L2 regularization term on weights. Increasing will make model more conservative.
                        'alpha': [1, 0.5, 0.8],
                        # L1 regularization term on weights. Increasing will make model more conservative.
                        'tree_method': ['auto', 'exact', 'approx', 'hist']
                    }
                else:
                    hyperparameter = {
                        'booster': ["gbtree", "gblinear"],  # Select the type of model to run at each iteration
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                        'gamma': [0, 2, 4],
                        'max_depth': [6, 4, 8],
                        'min_child_weight': [1, 2, 3],
                        'max_delta_step': [0, 1, 2, 3],
                        'subsample': [1, 0.7, 0.8],
                        # 'sampling_method':["uniform","gradient_based"],
                        'colsample_bytree': [1, 0.5, 0.8],
                        'colsample_bylevel': [1, 0.5, 0.8],
                        'colsample_bynode': [1, 0.5, 0.8],
                        'lambda': [1, 0.5, 0.8],
                        # L2 regularization term on weights. Increasing will make model more conservative.
                        'alpha': [1, 0.5, 0.8],
                        # L1 regularization term on weights. Increasing will make model more conservative.
                        'tree_method': ['auto', 'exact', 'approx', 'hist']
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_estimators': list(range(100, 800, 100)),
                        'booster': ["gbtree", "gblinear"],  # Select the type of model to run at each iteration
                        'learning_rate': [0.005, 0.025],
                        'gamma': [0, 2, 4],
                        'max_depth': [6, 4, 8],
                        'lambda': [1, 0.5, 0.8],
                        # L2 regularization term on weights. Increasing will make model more conservative.
                        'alpha': [1, 0.5, 0.8],
                        # L1 regularization term on weights. Increasing will make model more conservative.
                    }
                else:
                    hyperparameter = {
                        'booster': ["gbtree", "gblinear"],  # Select the type of model to run at each iteration
                        'learning_rate': [0.005, 0.025],
                        'gamma': [0, 2, 4],
                        'max_depth': [6, 4, 8],
                        'lambda': [1, 0.5, 0.8],
                        # L2 regularization term on weights. Increasing will make model more conservative.
                        'alpha': [1, 0.5, 0.8],
                        # L1 regularization term on weights. Increasing will make model more conservative.
                    }

        elif model_name == "TabNetClassifier":
            if self.extended_parameter_set:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_d': [2, 4, 6, 8, 16, 24, 32, 64, 128],
                        # Width of the decision prediction layer. Bigger more capacity to model with the risk of overfitting.
                        'n_a': [2, 4, 6, 8, 16, 24, 32, 64, 128],
                        # Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice.
                        'lr': [0.005, 0.01, 0.02, 0.025],
                        'n_steps': [3, 4, 5, 6, 7, 8, 9, 10],  # Number of steps in the architecture
                        'gamma': [1.0, 1.2, 1.5, 2.0],  # This is the coefficient for feature reusage in the masks.
                        'n_independent': [2, 3, 4, 1],  # Number of independent Gated Linear Units layers at each step.
                        'n_shared': [2, 3, 4, 1],  # Number of shared Gated Linear Units at each step
                        'momentum': [0.02, 0.06, 0.08, 0.2],
                        'lambda_sparse': [0, 0.000001, 0.0001, 0.001, 0.01, 0.1],
                        'scheduler_params': [{"step_size": 50, "gamma": 0.9}, {"step_size": 10, "gamma": 0.9},
                                             {"step_size": 30, "gamma": 0.9}],
                        'mask_type': ['entmax', "sparsemax"],
                        'optimizer_params': [dict(lr=2e-2), dict(lr=0.1), dict(lr=0.4)]
                    }
                else:
                    hyperparameter = {
                        # Width of the decision prediction layer. Bigger more capacity to model with the risk of overfitting.
                        'n_a': [2, 4, 6, 8, 16, 24, 32, 64, 128],
                        # Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice.
                        'lr': [0.005, 0.01, 0.02, 0.025],
                        'n_steps': [3, 4, 5, 6, 7, 8, 9, 10],  # Number of steps in the architecture
                        'gamma': [1.0, 1.2, 1.5, 2.0],  # This is the coefficient for feature reusage in the masks.
                        'n_independent': [2, 3, 4, 1],  # Number of independent Gated Linear Units layers at each step.
                        'n_shared': [2, 3, 4, 1],  # Number of shared Gated Linear Units at each step
                        'momentum': [0.02, 0.06, 0.08, 0.2],
                        'lambda_sparse': [0, 0.000001, 0.0001, 0.001, 0.01, 0.1],
                        'scheduler_params': [{"step_size": 50, "gamma": 0.9}, {"step_size": 10, "gamma": 0.9},
                                             {"step_size": 30, "gamma": 0.9}],
                        'mask_type': ['entmax', "sparsemax"],
                        'optimizer_params': [dict(lr=2e-2), dict(lr=0.1), dict(lr=0.4)]
                    }
            else:
                if self.train_all_at_once:
                    hyperparameter = {
                        'n_d': [2, 6, 32, 64],
                        # Width of the decision prediction layer. Bigger more capacity to model with the risk of overfitting.
                        'n_a': [2, 6, 32, 64],
                        # Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice.
                        'lr': [0.005, 0.02, 0.03],
                        'n_steps': [3, 4, 8, 10],  # Number of steps in the architecture
                        'gamma': [1.0, 1.5, 2.0],  # This is the coefficient for feature reusage in the masks.
                        'n_independent': [1, 3],  # Number of independent Gated Linear Units layers at each step.
                        'n_shared': [1, 3],  # Number of shared Gated Linear Units at each step
                        'momentum': [0.01, 0.1, 0.2],
                        'lambda_sparse': [0, 0.001, 0.1],
                        'scheduler_params': [{"step_size": 50, "gamma": 0.9}, {"step_size": 10, "gamma": 0.9}]
                    }
                else:
                    hyperparameter = {
                        'n_a': [2, 6, 32, 64],
                        # Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice.
                        'lr': [0.005, 0.02, 0.03],
                        'n_steps': [3, 4, 8, 10],  # Number of steps in the architecture
                        'gamma': [1.0, 1.5, 2.0],  # This is the coefficient for feature reusage in the masks.
                        'n_independent': [1, 3],  # Number of independent Gated Linear Units layers at each step.
                        'n_shared': [1, 3],  # Number of shared Gated Linear Units at each step
                        'momentum': [0.01, 0.1, 0.2],
                        'lambda_sparse': [0, 0.001, 0.1],
                        'scheduler_params': [{"step_size": 50, "gamma": 0.9}, {"step_size": 10, "gamma": 0.9}]
                    }
        else:
            self.error.error("{} not defined for hyperparameter selection! ".format(model_name) +
                             "Please select a model from the list of integrated models: {}".format
                             (self.available_models))

        return hyperparameter

    def round_down_to_nearest_10(self, num):
        """
        Round numbers down to the last 10
        :param num: number to round
        :return: rounded number
        """

        # takes a number and rounds it down to the nex number with %10 = 0
        return math.floor(num / 10) * 10

    def generate_max_leaf_nodes(self, df):
        """
        Generate max leaf nodes based on parameter from data
        :param df: data (training data with features
        :return: max number of leaf nodes for hyperparameter
        """

        max_leaf_nodes = list(
            range(int(self.round_down_to_nearest_10(len(df) / 2)), int(self.round_down_to_nearest_10(len(df))), 10))
        max_leaf_nodes.append(None)

        return max_leaf_nodes

    def hyperparameter_archive(self, model):
        """
        Store hyperparameter sets for each model depending on the model name
        :param model: model we select parameter collection for
        :return: set of hyperparameter for the model
        """

        all_available_hyperparameter = {}

        if self.optimizer_lib == "sklearn":
            all_available_hyperparameter = self.sklearn_archive(model=model)
        elif self.optimizer_lib == "optuna":
            all_available_hyperparameter = self.optuna_archive(model_name=self.get_model_name(model=model))
        else:
            self.error.error("{} optimizer lib not defined or not supported!".format(self.optimizer_lib) +
                             "Please select an integrated optimizer lib: {}".format(self.available_optimizer))

        return all_available_hyperparameter

    def get_model_name(self, model=None):
        """
        Get model name from input model
        :param model: model we need name for
        :return model_name:
        """
        if model is None:
            model = self.model

        model_name = type(model).__name__

        return model_name

    def get_hyperparameters(self, model=None):
        """
        Get Hyperparameter according to the model parameters
        :param model: model we need hyperparameters for
        :return hyperparameters: dict with set of hyperparameters
        """

        hyperparameters = {}

        if model is None:
            model = self.model

        model_name = self.get_model_name(model=model)

        if model_name in self.available_models:
            if self.optimizer_lib in self.available_optimizer:
                hyperparameters = self.hyperparameter_archive(model=model)
            else:
                self.error.error("Optimization library {} not supported! ".format(self.optimizer_lib) +
                                 "Pleas use one of the supported libraries: {}".format(self.available_optimizer))
        else:
            self.error.error("Model {} not in available models! ".format(model_name) +
                             "Please provide one of the available models {}.".format(self.available_models))

        return hyperparameters
