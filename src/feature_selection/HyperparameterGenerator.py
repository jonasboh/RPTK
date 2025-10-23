import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import random
import statistics
import multiprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from time import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from optuna.trial import Trial


class GridHyperParameterGenerator:
    def __init__(self,
                 model,
                 x,
                 y,
                 unsup_pretraining: bool = False,
                 n_features: int = 10,
                 extended_parameter_set: bool = False,  # If more hyperparemter shoudl be used for optimization
                 use_optuna: bool = False,  # Use Optuna for optimization
                 trial: Trial = Trial):

        self.model = model
        self.X = x
        self.y = y
        self.unsup_pretraining = unsup_pretraining
        self.n_features = n_features
        self.extended_parameter_set = extended_parameter_set
        self.use_optuna = use_optuna
        self.trial = trial

    ## Optuna does not work in this setting
    def generate_hyperparameter_set(self):

        self.hyperparameter = {}

        if type(self.model).__name__ == "RandomForestClassifier":

            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {
                        "n_estimators": self.trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        "ccp_alpha": self.trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': self.trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "max_samples": self.trial.suggest_float('max_samples', 0.2, 0.8, step=0.01),
                        "max_features": self.trial.suggest_categorical("max_features",
                                                                  ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "criterion": self.trial.suggest_categorical("gini", "entropy"),
                    }
                else:
                    self.hyperparameter = {

                        ## numerical ##
                        ## "n_estimators": list(range(100, 800, 100)),  # Number of Trees for the forest
                        "ccp_alpha": [0.0, 0.001, 0.01, 0.02, 0.03],
                        # Complexity parameter for Minimal Cost-Complexity Pruning (subtree with largest cost complexity that is smaller than ccp_alpha will be chosen)

                        'max_depth': [None, 2, 4, 6, 8, 10],  # Max depth of a tree
                        "max_samples": [0.2, 0.4, 0.6, 0.8, None],
                        # If bootstrap is True, the number of samples to draw from X to train each base estimator.
                        "max_features": ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0],
                        # Number of features to consider for the best split

                        # detailed parameter
                        # "max_depth": list(range(5, round_down_to_nearest_10(len(X_Train)/2), 1)), # Max depth of a tree
                        # "max_features": list(range(round_down_to_nearest_10(X_Train.shape[1]/2), round_down_to_nearest_10(X_Train.shape[1]), 5)), # Number of features to consider for the best split
                        ###"max_leaf_nodes": generate_max_leaf_nodes(X_Train), # Grow trees with max_leaf_nodes in best-first fashion.
                        # "max_samples": [None, 0.2, 0.4, 0.6, 0.8], # If bootstrap is True, the number of samples to draw from X to train each base estimator.

                        # "min_samples_split": [2, 0.2, 0.4, 0.6, 0.8, 1.0], # The minimum number of samples required to split an internal node
                        # "min_samples_leaf": [1, 0.2, 0.3, 0.4], # Minimum number of samples required at a leaf node. --> smoothing the mode
                        ###"min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.4], # Minimum weighted fraction of the sum of weights (of all the input samples) required to be at a leaf node
                        ###"min_impurity_decrease": [0.0, 0.1, 0.2, 0.3], # Split node if split induces a decrease of the impurity greater than or equal to this value

                        # categorical
                        "criterion": ["gini", "entropy"]
                        ,
                        # "log_loss"], # The function to measure the quality of a split Note: This parameter is tree-specific
                        # "bootstrap": [True, False], # Bootstrap samples when building trees or use whole dataset to build each tree.
                        ##"oob_score": [False, True], # Use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
                        ##"warm_start": [False, True], # Reuse solution of the previous call to fit and add more estimators to the ensemble or fit new forest
                        # "class_weight": [None, "balanced", "balanced_subsample"] # “balanced” adjust weights inversely proportional to class frequencies balanced_subsample” do “balanced” on the bootstrap sample for every tree grown.
                    }
            else:
                if self.use_optuna:
                    self.hyperparameter = {
                        "n_estimators": self.trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        'max_depth': self.trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                    }
                self.hyperparameter = {
                    "n_estimators": list(range(100, 800, 200)),  # Number of Trees for the forest
                    'max_depth': [None, 2, 4, 8, 10],  # Max depth of a tree
                }

        elif type(self.model).__name__ == "LinearSVC":

            if self.X.shape[0] > self.X.shape[1]:
                dual = False
            else:
                dual = True
            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {
                        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                        #"loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                        "dual": [dual],
                        "tol": trial.suggest_float("tol", 1e-6, 1e-2, step=1e-1),
                        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                    }
                else:
                    self.hyperparameter = {
                        "penalty": ["l1", "l2"],
                        #"loss": ["hinge", "squared_hinge"],
                        "dual": [dual],
                        "tol": [1e-6, 1e-4, 1e-2],
                        "class_weight": [None, "balanced"],
                        "fit_intercept":[True, False],
                    }
            else:
                if self.use_optuna:
                    self.hyperparameter = {
                        "penalty": self.trial.suggest_categorical("penalty", ["l1", "l2"]),
                        #"loss": self.trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                        "dual": [dual],
                        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                    }
                else:
                    self.hyperparameter = {
                        "penalty": ["l1"],
                        #"loss": ["hinge", "squared_hinge"],
                        "dual": [dual],
                        "class_weight": [None, "balanced"],
                    }

        elif type(self.model).__name__ == "SelectKBest":

            # No optuna optimization! -> No Effect?!
            if self.extended_parameter_set:
                self.hyperparameter = {
                    "score_func": ["f_classif", "chi2","SelectFdr","SelectFpr"],
                    "k": [self.n_features]
                }
            else:
                self.hyperparameter = {
                    "score_func": ["f_classif","chi2"],
                    "k": [self.n_features]
                }

        elif type(self.model).__name__ == "Lasso":

            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {

                        "alpha": trial.suggest_int("alpha", -5, 100),
                        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                        "max_iter": trial.suggest_int("max_iter", 500, 1500),
                        "tol": trial.suggest_float("tol", 1e-10, 1e-1, step=1e-1),
                    }
                else:
                    self.hyperparameter = {
                        "alpha": [np.logspace(-5, 1, num=100)],
                        "fit_intercept": [True, False],
                        "max_iter": [500, 1000, 1500],
                        "tol": [1e-6, 1e-4, 1e-2],
                    }

            else:
                if self.use_optuna:
                    self.hyperparameter = {

                        "alpha": trial.suggest_int("alpha", -5, 100),
                    }
                else:
                    self.hyperparameter = {
                        "alpha": [np.logspace(-5, 1, num=100)]
                    }
        elif type(self.model).__name__ == "GradientBoostingClassifier":
            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.03, step=0.01),
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "validation_fraction": trial.suggest_float('validation_fraction', 0.1, 0.4, step=0.01),
                        "max_features": trial.suggest_categorical("max_features",
                                                                  ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001),
                        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
                        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 0, 100),
                        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [generate_max_leaf_nodes(self.X)]),
                        "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
                        "init": trial.suggest_categorical("init", [None, self.model]),
                        "warm_start": trial.suggest_categorical("warm_start", [False, True]),
                        "tol": trial.suggest_float("tol", 1e-4, 0.3, step=0.01),
                    }
                else:
                    self.hyperparameter = {

                        ## numerical ##
                        # "n_estimators": list(range(100, 800, 100)), # Number of Trees for the forest
                        "ccp_alpha": [0.0, 0.001, 0.01, 0.02, 0.03],
                        # Complexity parameter for Minimal Cost-Complexity Pruning (subtree with largest cost complexity that is smaller than ccp_alpha will be chosen)
                        "validation_fraction": [0.1, 0.2, 0.3, 0.4],
                        # The proportion of training data to set aside as validation set for early stopping
                        "learning_rate": [0.005, 0.01, 0.02, 0.025],  # Learning rate shrinks the contribution of each tree
                        "subsample": [1.0, 0.2, 0.4, 0.6, 0.8],
                        # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
                        "n_iter_no_change": [None, 20, 40, 60, 80, 100],
                        # number of iterations after val loss is not improving
                        # "tol": [1e-4,0.1,0.2,0.3], # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations

                        'max_depth': [None, 2, 4, 6, 8, 10],  # Max depth of a tree
                        "max_features": ["sqrt", None, 0.2, 0.4, 0.6, 0.8, 1.0],
                        # Number of features to consider for the best split
                        "max_leaf_nodes": generate_max_leaf_nodes(self.X),
                        # Grow trees with max_leaf_nodes in best-first fashion. ,

                        # "min_samples_split": [2, 0.2, 0.4, 0.6, 0.8, 1.0], # The minimum number of samples required to split an internal node
                        # "min_samples_leaf": [1, 0.2, 0.3, 0.4], # Minimum number of samples required at a leaf node. --> smoothing the mode
                        # "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.4], # Minimum weighted fraction of the sum of weights (of all the input samples) required to be at a leaf node
                        # "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3], # Split node if split induces a decrease of the impurity greater than or equal to this value

                        # categorical
                        # "loss":["log_loss","exponential"], # The loss function to be optimized.
                        "criterion": ["friedman_mse", "squared_error"],
                        # The function to measure the quality of a split Note: This parameter is tree-specific
                        # "bootstrap": [True, False], # Bootstrap samples when building trees or use whole dataset to build each tree.
                        # "oob_score": [False, True], # Use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
                        "warm_start": [False, True],
                        # Reuse solution of the previous call to fit and add more estimators to the ensemble or fit new forest
                        "init": [None, self.model]
                    }
            else:
                if self.use_optuna:
                    self.hyperparameter = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),  # Number of Trees for the forest
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  # Max depth of a tree
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, .025, step=0.001)
                    }
                else:
                    self.hyperparameter = {
                        "learning_rate": [0.005, 0.01, 0.025],  # Learning rate shrinks the contribution of each tree
                        'max_depth': [None, 2, 4, 8, 10],  # Max depth of a tree
                    }

        elif type(self.model).__name__ == "LGBMClassifier":
            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {
                        'num_leaves': trial.suggest_int('num_leaves', 20, 30),  # max number of leaves in one tree
                        'max_depth': trial.suggest_int('num_leaves', -1, 15),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 40),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.025, step=0.001),
                        'max_bin': trial.suggest_int('max_bin',40, 100),  # max number of bins that feature values will be bucketed in
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0, step=0.1),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.2, step=0.01),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.2, step=0.01),
                        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.4, step=0.01),

                        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 11),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.99, step=0.1),
                    }
                else:
                    self.hyperparameter = {
                        'num_leaves': [20, 30],  # max number of leaves in one tree
                        'max_depth': [-1, 5, 7, 10, 15],
                        'min_data_in_leaf': [20, 30, 40],
                        # minimal number of data in one leaf. Can be used to deal with over-fitting
                        # 'num_iterations':[100,150,200],
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                        'max_bin': [40, 80, 100],  # max number of bins that feature values will be bucketed in
                        # 'path_smooth':[0.2,0.5,0.6],
                        'feature_fraction': [0.4, 0.6, 0.8, 1.0],
                        'lambda_l1': [0.0, 0.2, 0.4],
                        'lambda_l2': [0.0, 0.2, 0.4],
                        'min_gain_to_split': [0.0, 0.2, 0.4],

                        'boosting_type': ['gbdt', 'dart', 'rf'],
                        # 'num_leaves': sp.stats.randint(2, 1001),
                        # 'subsample_for_bin': sp.stats.randint(10, 1001),
                        # 'min_split_gain': sp.stats.uniform(0, 5.0),
                        # 'min_child_weight': sp.stats.uniform(1e-6, 1e-2),
                        # 'reg_alpha': sp.stats.uniform(0, 1e-2),
                        # 'reg_lambda': sp.stats.uniform(0, 1e-2),
                        # 'tree_learner': ['data', 'feature', 'serial', 'voting' ],
                        # 'application': ['binary', 'cross-entropy'],
                        'bagging_freq': [1, 2, 5, 8, 11],
                        'bagging_fraction': [0.4, 0.6, 0.99],
                        # 'feature_fraction': sp.stats.uniform(1e-3, 0.99),
                        # 'learning_rate': sp.stats.uniform(1e-6, 0.99),
                        # 'max_depth': sp.stats.randint(1, 501),
                    }
            else:
                if self.use_optuna:
                    self.hyperparameter = {
                        'max_depth': trial.suggest_int('num_leaves', -1, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.025, step=0.001),
                    }
                else:
                    self.hyperparameter = {
                        'max_depth': [-1, 5, 10, 15],
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                    }


        elif type(self.model).__name__ == "XGBClassifier":
            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {
                        'booster': trial.suggest_categorical('booster', ["gbtree", "gblinear"]),  # Select the type of model to run at each iteration
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
                    self.hyperparameter = {
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
                if self.use_optuna:
                    self.hyperparameter = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.025, step=0.001),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'lambda': trial.suggest_float('lambda', 0.6, 1.0, step=0.1),
                        'alpha': trial.suggest_float('alpha', 0.6, 1.0, step=0.1),
                    }
                else:
                    self.hyperparameter = {
                        'learning_rate': [0.005, 0.01, 0.02, 0.025],
                        'max_depth': [6, 4, 8],
                        'lambda': [1, 0.5, 0.8], # L2 regularization term on weights. Increasing will make model more conservative.
                        'alpha': [1, 0.5, 0.8],  # L1 regularization term on weights. Increasing will make model more conservative.
                    }


        elif type(self.model).__name__ == "TabNetClassifier":
            if self.extended_parameter_set:

                if self.use_optuna:
                    self.hyperparameter = {
                        'n_d': trial.suggest_int('n_d', 100, 200),
                        'n_a': trial.suggest_int('n_a', 2, 100),
                        'n_steps': trial.suggest_int('n_steps', 3, 10),  # Number of steps in the architecture
                        'gamma': trial.suggest_float('gamma', 1.0, 2.0, steps=0.1),  # This is the coefficient for feature reusage in the masks.
                        'n_independent': trial.suggest_int('n_independent', 1, 4),  # Number of independent Gated Linear Units layers at each step.
                        'n_shared': trial.suggest_int('n_shared', 1, 4),  # Number of shared Gated Linear Units at each step
                        'momentum': trial.suggest_float('momentum', 0.02, 0.2, steps=0.01),
                        'lambda_sparse': trial.suggest_float('lambda_sparse', 0.001, 0.1, steps=0.0001),
                        'mask_type': trial.suggest_int('mask_type', ['entmax', "sparsemax"]),
                        'lr': trial.suggest_float('lr', 0.005, 0.025, steps=0.001),
                    }
                else:
                    self.hyperparameter = {
                        'n_d': [8, 16, 24, 32, 64, 128, 6, 4, 2],
                        # Width of the decision prediction layer. Bigger more capacity to model with the risk of overfitting.
                        'n_a': [8, 16, 24, 32, 64, 128, 6, 4, 2],
                        # Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice.
                        ## 'optimizer_params':[dict(lr=0.005),dict(lr=0.01),dict(lr=0.02),dict(lr=0.025)],
                        'lr': [0.005, 0.01, 0.02, 0.025],
                        'n_steps': [3, 4, 5, 6, 7, 8, 9, 10],  # Number of steps in the architecture
                        'gamma': [1.0, 1.2, 1.5, 2.0],  # This is the coefficient for feature reusage in the masks.
                        'n_independent': [2, 3, 4, 1],  # Number of independent Gated Linear Units layers at each step.
                        'n_shared': [2, 3, 4, 1],  # Number of shared Gated Linear Units at each step
                        'momentum': [0.02, 0.06, 0.08, 0.2],
                        'lambda_sparse': [0, 0.000001, 0.0001, 0.001, 0.01, 0.1],
                        # 'scheduler_params':[{"step_size":50,"gamma":0.9},{"step_size":10,"gamma":0.9},{"step_size":30,"gamma":0.9}],
                        'mask_type': ['entmax', "sparsemax"],
                        # 'optimizer_params':[dict(lr=2e-2),dict(lr=0.1),dict(lr=0.4)]
                    }
            else:
                if self.use_optuna:
                    self.hyperparameter = {
                        'n_d': trial.suggest_int('n_d', 100, 200),
                        'n_a': trial.suggest_int('n_a', 2, 100),
                        'lr': trial.suggest_float('lr', 0.005, 0.025, steps=0.001),
                    }
                else:
                    self.hyperparameter = {
                        'lr': [0.005, 0.01, 0.02, 0.025],
                        'n_d': [8, 16, 128, 6, 2],
                        'n_a': [8, 16, 128, 6,  2],
                    }
        elif type(self.model).__name__ == "KNeighborsClassifier":
            if self.extended_parameter_set:
                if self.use_optuna:
                    self.hyperparameter = {
                        'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
                        'p': trial.suggest_int('p', 1, 10)
                    }
                else:
                    self.hyperparameter = {
                        'n_neighbors': [3, 5, 8, 9, 10],
                        'leaf_size': [10, 20, 30, 40, 50, 60,70, 80],
                        'p': [1, 4, 6, 7, 8]
                    }
            else:
                if self.use_optuna:
                    self.hyperparameter = {
                        'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
                        'p': trial.suggest_int('p', 1, 10)
                    }
                else:
                    self.hyperparameter = {
                        'n_neighbors': [3, 5, 8, 10],
                        'leaf_size': [10, 20, 40, 60, 80],
                        'p': [1, 4, 6, 8]
                    }
        else:
            raise ValueError(
                f"Hyperparameter optimization for {type(self.model).__name__} not implemented!"
            )
        return self.hyperparameter

def round_down_to_nearest_10(num):
    # takes a number and rounds it down to the nex number with %10 = 0
    return math.floor(num / 10) * 10

def generate_max_leaf_nodes(df):
    max_leaf_nodes = list(range(int(round_down_to_nearest_10(len(df) / 2)), int(round_down_to_nearest_10(len(df))), 10))
    max_leaf_nodes.append(None)

    return max_leaf_nodes

#class OptunaHyperparameterGenerator:
#    def __init__(self,
#                 model,
#
#                 ):

#        self.model = model

