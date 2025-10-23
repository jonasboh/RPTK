# +
from sklearn.model_selection import (
    TimeSeriesSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import glob
import os
from tqdm import tqdm
import multiprocessing
# import lightgbm as lgb

from sklearn.model_selection import validation_curve
from random import randint

import torch
import optuna

import scipy as sp

import math
import time
import random

from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV, LeaveOneGroupOut, LeavePGroupsOut, \
    ParameterGrid, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, RocCurveDisplay, confusion_matrix, \
    auc, roc_curve, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from sklearn.model_selection import ParameterSampler

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from pytorch_tabnet.tab_model import TabNetClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.nipy_spectral
# cmap_cv = plt.cm.coolwarm

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import random

from rptk.src.model_training.HyperparameterConfigurator import HyperparameterConfigurator

#from rptk.src.CV_Generator import *
#from rptk.src.Oversampler import *
#from rptk.src.Normal_Dist_Simulator import *


# +
class Optimizer:
    """
       Class for grid search optimization and improvements by oversampling & data simulation

       Objects in class
       ----------
       Object from class Optimizer

       Functions in Class
       -------
       RandomSearchCV_iteration: Make grid search by randomly select hyperparameter combinations
       Report_best_RandomSearchCV_iteration: Report best grid search iteration by AUC
       Optimize_with_simulation: Improve Grid serach by adding simulated data and performing oversampling (BorderlineSMOTE)
       Repeated_optimize: Use false detection to imporve performance by simulation and upsampling repeatedly

    """

    def __init__(self,
                 X=pd.DataFrame(),
                 y=pd.Series(),
                 num_sample_simulations=0,
                 prefix="",
                 random_grid_search=True,  # Use Random Grid search Else use GridSearchCV
                 start_grid_iter=100,
                 end_grid_iter=300,
                 int_grid_iter=100,
                 X_val=pd.DataFrame(),
                 y_val=pd.DataFrame(),
                 groups=None,
                 setting=None,
                 debug_grid=np.nan,
                 num_std=1,
                 n_train=1,
                 false_df=pd.DataFrame(),
                 oversample=True,
                 label="Label",
                 index="ID",
                 fit_params=None,
                 num_simulations=1,
                 best_setting_num=None,
                 prefit=True,
                 verbose=1,
                 cv=GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=123),
                 model=RandomForestClassifier(random_state=123),
                 best_num_estimators=100,
                 out_file="test.png",
                 out_dir=".",
                 cpu=1):

        if groups is None:
            groups = []
        if setting is None:
            setting = {}
        if fit_params is None:
            fit_params = {}
        self.x = X.copy()
        self.y = y.copy()

        if groups == None:
            groups = []
        if setting == None:
            setting = {}
        if fit_params == None:
            fit_params = {}

        if (len(X_val) != 0) & (len(y_val) != 0):
            self.x_val = X_val.copy()
            self.y_val = y_val.copy()
        else:
            self.x_val = None
            self.y_val = None

        if len(false_df) != 0:
            if "Prediction" in false_df.columns:
                false_df.drop(['Prediction'], axis=1, inplace=True)
            self.false = false_df.copy()
        else:
            self.false = pd.DataFrame()

        if len(groups) == 0:
            groups = self.x.index.values
        self.groups = groups

        self.num_simulations = num_simulations
        self.num_simu_samples = num_sample_simulations
        self.prefix = prefix
        self.cv = cv  # Crossvalidator
        self.model = model
        self.start = start_grid_iter
        self.end = end_grid_iter
        self.iter = int_grid_iter
        self.out_file = out_file
        self.out_dir = out_dir
        self.fit_params = fit_params  # Parameter for model training
        self.best_num_estimators = best_num_estimators  # Set best_num_estimators = 0 to get optimum num_estimators from settings
        self.index = index
        self.label = label
        self.best_setting_num = best_setting_num  # if avg_settings = False take this number of best model parameter for mean settings (if None -> average from best seetings from each iteration; if False take the best estimator from grid search
        self.oversample = oversample  # Oversampling of dataset before optimization
        # if len(setting) == 0:
        #    setting
        self.verbose = verbose  # Print statements and detailed output
        self.num_std = num_std  # times standard deviation treshold for data simulation
        self.n_train = n_train  # number of how often the model needs to get trained
        self.setting = setting  # Set of hyperparameters to optimize
        self.debug_grid = debug_grid  # Raise error value when grid serach or replace non calculated models
        self.prefit = prefit  # show model performance before simulation
        self.grid_iteration = 1
        self.random_grid_search = random_grid_search
        self.cpu = cpu

    def annot_max(self, x, y, ax=None):
        """
        Annotate the x,y point with the max AUROC
        """

        xmax = x[np.argmax(y)]
        ymax = y.max()

        text = "Max AUROC: x={:.3f}, y={:.3f}".format(xmax, ymax)

        if not ax:
            ax = plt.gca()

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")

        if self.n_train == 1:
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.88, 0.89), **kw)

        return xmax, ymax
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def get_stable_idx(self, arr):
        """
        Get the index of the longest stable plateau in an array
        :param arr: array to get stable index from
        """
        
        most_trees_stable_idx = 0
        
        # get differences in array
        diff_test = np.diff(arr)
        # get gradient in array
        plateaus = np.sign(np.around(diff_test, 2))

        idx_pairs = np.where(np.diff(np.hstack(([False], plateaus == 0, [False]))))[0].reshape(-1, 2)
        
        if len(idx_pairs) == 0:
            nearest_0_plateau = self.find_nearest(plateaus, 0)
            idx_pairs = np.where(np.diff(np.hstack(([False], plateaus == nearest_0_plateau, [False]))))[0].reshape(-1, 2)
        
        # get longest stable index with min num_estimators
        min_stable = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 0]
        max_stable = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 1]

        stable = min_stable + int((max_stable - min_stable) / 2)

        stable_index = 0

        # get min number of trees from plateau
        most_trees_stable_idx = stable_index + stable

        return most_trees_stable_idx

    def annot_stable(self, x, y, ax=None):
        """
        Annotate the plot with the stable AUROC
        :param x (pd.DataFrame): Values training was based on
        :param y (pd.Series): label training was based on
        :param ax: ax from matplotlib
        """

        x_stable = x[self.get_stable_idx(y)]
        y_stable = y[self.get_stable_idx(y)]

        text = "Stable AUROC: x={:.3f}, y={:.3f}".format(x_stable, y_stable)

        if not ax:
            ax = plt.gca()

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=40")

        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="left", va="bottom")

        if self.n_train == 1:
            ax.annotate(text, xy=(x_stable, y_stable), xytext=(0.20, 0.21), **kw)

        return x_stable, y_stable

    def train_and_get_params_from_grid(self, X, y, group, parameters, gclf, iterations, iter, fit_params=None):
        """
        Train model with given parameters and return model and parameters
        """
        if fit_params is None:
            fit_params = {}

        para = {}

        if len(fit_params) == 0:
            # if self.verbose == 1:
            # TODO: logging
            #    print("Warning: No Fitting Settings! Perform training without settings!")
            if type(self.model).__name__ == "TabNetClassifier":
                grid_model = gclf.fit(X.values, y, groups=group)
            else:
                grid_model = gclf.fit(X, y, groups=group)
        else:
            if type(self.model).__name__ == "TabNetClassifier":
                grid_model = gclf.fit(X.values, y, groups=group, **fit_params)
            else:
                grid_model = gclf.fit(X, y, groups=group, **fit_params)

        df = pd.DataFrame(gclf.cv_results_)
        iterations[iter] = df['mean_test_roc_auc'].loc[df["rank_test_roc_auc"] == 1].drop_duplicates(
            keep='first').values

        # TODO: this might be problematic for other models which do not have get_params() function
        if type(self.model).__name__ == "RandomForestClassifier":
            para = {"iter": iter,
                    "params": gclf.best_estimator_.get_params(),
                    "AUC": df['mean_test_roc_auc'].loc[df["rank_test_roc_auc"] == 1].drop_duplicates(
                        keep='first').values[0]}

        parameters.append(para)

        return grid_model, parameters, iterations

    def calc_max_papra_iter(self):
        """
        Calculate max number of iterations for grid search
        """
        # Calculate max combinations of parameters
        max_iters = 1
        for i in self.setting:
            max_iters = max_iters * len(i)

        return max_iters

    def RandomSearchCV_iteration(self, X, y, group, parameter, fit_params=None):
        """
        Make a grid search by randomly selecting Hyperparameter combinations
        """

        # use defined fit params if given in as init of the object
        if fit_params is None:
            fit_params = {}

        if fit_params == {} and self.fit_params != {}:
            fit_params = self.fit_params

        # Do several grid searches to investigate how many iterations are neede to get the optimal performance (global optimum)
        grid_models = {}
        iterations = {}
        parameters = []
        grid_model = ""
        rfc_best_trained = ""

        start = time.time()

        # look from Start - End
        if type(self.model).__name__ == "TabNetClassifier":

            # if self.tabnet_unsup_pretraining != np.nan:
            # parameter['from_unsupervised'] = [self.tabnet_unsup_pretraining]

            for i in range(self.start, self.end, self.iter):
                grid = list(ParameterSampler(parameter,
                                             n_iter=i,
                                             random_state=123))

                iter_ = 0
                search_results = pd.DataFrame(columns=["iter", "params", "AUC"])

                # go through each parameter setting
                for params in tqdm(grid):
                    params['n_a'] = params['n_d']  # n_a=n_d always per the paper
                    tabnet = create_tabnet()

                    tabnet.set_params(**params)

                    tabnet.fit(X_train=X.values, y_train=y, **fit_params)

                    train_roc_auc, val_roc_auc = Optimizer.calc_train_val_auc(self,
                                                                              tabnet,
                                                                              X,
                                                                              y,
                                                                              self.x_val,
                                                                              self.y_val)
                    search_results_tmp = {"iter": iter_,
                                          "params": params,
                                          "AUC": val_roc_auc}

                    parameters.append(search_results_tmp)

                    iter_ += 1

                search_results = pd.DataFrame(parameters)

                # search_results.sort_values(by=["AUC"], inplace = True)
                best_para = search_results.loc[search_results['AUC'] == search_results['AUC'].max()]
                iterations[i] = search_results.loc[search_results['AUC'] == search_results['AUC'].max()]
                tabnet = create_tabnet()

                # print(best_para)
                # print(dtype(best_para))
                # print(dtype(best_para["params"]))
                # print(best_para["params"].values)
                # print(dtype(best_para["params"].values))
                # print(best_para["params"].values[0])
                # print(dtype(best_para["params"].values[0]))

                grid_model = tabnet.set_params(**best_para["params"].values[0])
        elif (type(self.model).__name__ == "LGBMClassifier"):

            # Do several grid searches to investigate how many iterations are neede to get the optimal performance (global optimum)

            for i in range(self.start, self.end, self.iter):
                grid = list(ParameterSampler(parameter,
                                             n_iter=i,
                                             random_state=123))

                iter_ = 0
                search_results = pd.DataFrame(columns=["iter", "params", "AUC"])

                lgbtm = ""

                # go through each parameter setting
                for params in tqdm(grid):
                    # params['n_a'] = params['n_d'] # n_a=n_d always per the paper
                    lgbtm = self.model
                    lgbtm.set_params(**params)

                    lgbtm = lgbtm.fit(X, y, **fit_params)

                    train_roc_auc, val_roc_auc = Optimizer.calc_train_val_auc(self,
                                                                              lgbtm,
                                                                              X,
                                                                              y,
                                                                              self.x_val,
                                                                              self.y_val)
                    search_results_tmp = {"iter": iter_,
                                          "params": params,
                                          "AUC": val_roc_auc}

                    parameters.append(search_results_tmp)

                    iter_ += 1

                search_results = pd.DataFrame(parameters)

                # search_results.sort_values(by=["AUC"], inplace = True)
                best_para = search_results.loc[search_results['AUC'] == search_results['AUC'].max()]
                iterations[i] = search_results.loc[search_results['AUC'] == search_results['AUC'].max()]

                grid_model = lgbtm.set_params(**best_para["params"].values[0])

        elif (type(self.model).__name__ == "XGBClassifier"):

            # Do several grid searches to investigate how many iterations are neede to get the optimal performance (global optimum)

            for i in range(self.start, self.end, self.iter):
                grid = list(ParameterSampler(parameter,
                                             n_iter=i,
                                             random_state=123))

                iter_ = 0
                search_results = pd.DataFrame(columns=["iter", "params", "AUC"])
                xgboost = self.model
                # go through each parameter setting
                for params in tqdm(grid):
                    # params['n_a'] = params['n_d'] # n_a=n_d always per the paper

                    xgboost.set_params(**params)

                    if len(fit_params) == 0:
                        xgboost.fit(X, y)
                    else:
                        # xgboost.fit(X, y, **fit_params,eval_set=[(self.x_val,self.y_val)])
                        xgboost.fit(X, y,
                                    verbose=True,
                                    # early_stopping_rounds=10,
                                    eval_metric='rmse')  # ,
                        # eval_set=[(self.x_val,self.y_val)])

                    train_roc_auc, val_roc_auc = Optimizer.calc_train_val_auc(self,
                                                                              xgboost,
                                                                              X,
                                                                              y,
                                                                              self.x_val,
                                                                              self.y_val)
                    search_results_tmp = {"iter": iter_,
                                          "params": params,
                                          "AUC": val_roc_auc}

                    parameters.append(search_results_tmp)

                    iter_ += 1

                search_results = pd.DataFrame(parameters)

                # search_results.sort_values(by=["AUC"], inplace = True)
                best_para = search_results.loc[search_results['AUC'] == search_results['AUC'].max()]
                iterations[i] = search_results.loc[search_results['AUC'] == search_results['AUC'].max()]

                grid_model = xgboost.set_params(**best_para["params"].values[0])

        else:
            if self.random_grid_search:

                max_iter = self.calc_max_papra_iter()

                # Do not more iterations as possible parameter combinations
                if self.end > max_iter:
                    self.end = max_iter

                for i in tqdm(range(self.start, self.end, self.iter)):
                    gclf = RandomizedSearchCV(self.model,
                                              parameter,
                                              verbose=0,
                                              n_jobs=self.cpu,
                                              n_iter=i,
                                              scoring=["balanced_accuracy", "roc_auc"],
                                              cv=self.cv,
                                              refit="balanced_accuracy",
                                              random_state=123,
                                              error_score=self.debug_grid
                                              )

                    grid_model, parameters, iterations = self.train_and_get_params_from_grid(X=X,
                                                                                             y=y,
                                                                                             group=group,
                                                                                             parameters=parameters,
                                                                                             gclf=gclf,
                                                                                             iterations=iterations,
                                                                                             iter=i,
                                                                                             fit_params=fit_params)
                    grid_models[parameters[len(parameters) - 1]["AUC"]] = grid_model
            else:
                gclf = GridSearchCV(self.model,
                                    parameter,
                                    verbose=0,
                                    n_jobs=self.cpu,
                                    scoring=["balanced_accuracy", "roc_auc"],
                                    cv=self.cv,
                                    refit="balanced_accuracy",
                                    error_score=self.debug_grid
                                    )
                i = 1

                grid_model, parameters, iterations = self.train_and_get_params_from_grid(X=X,
                                                                                         y=y,
                                                                                         group=group,
                                                                                         parameters=parameters,
                                                                                         gclf=gclf,
                                                                                         iterations=iterations,
                                                                                         iter=i,
                                                                                         fit_params=fit_params)
                grid_models[round(parameters["AUC"], 3)] = grid_model

        end = time.time()
        if self.verbose == 1:
            print("time:", round((end - start) / 60, 2), "minutes")

        # save the result to CSV
        best_df = pd.DataFrame(parameters)
        best_df.to_csv(self.out_file, index=False)

        smallest_best_iteration, best_auc, best_parameters = self.report_best_RandomSearchCV_iteration(
            parameters=parameters)

        self.best_grid_model = grid_models[best_auc]

        # Look at the performance of the best model
        if (type(self.model).__name__ == "RandomForestClassifier"):
            rfc_best_trained, false, y_pred_proba, y_pred_true = cross_val(model=self.best_grid_model.best_estimator_,
                                                                           cv=self.cv,
                                                                           X=X,
                                                                           y=y,
                                                                           group=group,
                                                                           verbose=1,
                                                                           simulation=True,
                                                                           reduced_output=True,
                                                                           out_path=self.out_dir)

        return parameters, iterations, self.best_grid_model, y_pred_proba, y_pred_true

    def get_best_grid_model(self):
        return self.best_grid_model

    def report_best_RandomSearchCV_iteration(self, parameters, title="Grid_Search_Iteration"):
        """
        Plot the best iteration of the grid search
        """

        df_i = pd.DataFrame(parameters)

        if df_i.shape[0] > 1:
            plot = df_i.plot(x="iter",
                             y="AUC",
                             colormap='viridis',
                             legend=None)

            plt.xlabel("Iteration")
            plt.ylabel("Best test mean AUROC")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(self.out_dir + str(type(self.model).__name__) + "_" + title + '.png')

        smallest_best_iteration = df_i.drop_duplicates(subset="iter")["iter"].loc[
            df_i.drop_duplicates(subset="iter")["AUC"] == df_i.drop_duplicates(subset="iter")["AUC"].max()].values[0]
        best_auc = df_i.drop_duplicates(subset="iter")["AUC"].loc[
            df_i.drop_duplicates(subset="iter")["AUC"] == df_i.drop_duplicates(subset="iter")["AUC"].max()].values[0]

        # if self.verbose == 1:
        print("Max AUC with", smallest_best_iteration, "iteration and", round(best_auc, 3), "AUROC")

        # get the best parameters
        best_parameters = df_i.loc[df_i["iter"] == smallest_best_iteration]["params"].values[0]

        # if df_i.shape[1] > 1:
        #    fig = plot.get_figure()
        #    fig.savefig(self.out_file)
        # plt.close()

        return smallest_best_iteration, best_auc, best_parameters

    def get_best_parameter_model(self, best_parameters, best_auc, final_grid_model):

        best_mean_parameter = {}

        # 3. Train model with best setting
        if self.best_setting_num == None:
            best_mean_parameter = best_average_setting(best_parameters, best_auc)
        elif type(self.best_setting_num) == dict:
            pass
        elif type(self.best_setting_num) == int:
            if type(final_grid_model).__name__ == "TabNetClassifier":
                best_mean_parameter = best_average_setting(best_parameters, best_auc, self.best_setting_num)
            else:
                best_mean_parameter = best_average_setting(final_grid_model.cv_results_, best_auc,
                                                           self.best_setting_num)
        elif self.best_setting_num == False:
            if type(final_grid_model).__name__ == "TabNetClassifier":
                best_mean_parameter = best_average_setting(best_parameters, best_auc, self.best_setting_num)
            else:
                best_rfc = final_grid_model.best_estimator_
        else:
            print(
                "Error: Please set best_setting_num Parameter accordingly [None,int,False]. (None for mean parameter of grid iterationg, int for mean of best final grid iteration, False for only best estimator of last grid iteration.")

        if len(best_mean_parameter) != 0:
            if self.best_num_estimators == 0:
                if type(self.model).__name__ == "RandomForestClassifier":
                    best_rfc = RandomForestClassifier(random_state=123,
                                                      n_jobs=self.cpu,
                                                      verbose=0,
                                                      bootstrap=True,
                                                      oob_score=True,
                                                      **best_mean_parameter)

                if type(self.model).__name__ == "GradientBoostingClassifier":
                    best_rfc = GradientBoostingClassifier(random_state=123,
                                                          **best_mean_parameter)

                if type(self.model).__name__ == "LGBMClassifier":
                    best_rfc = lgb.LGBMClassifier(random_state=123,
                                                  **best_mean_parameter)

                if type(self.model).__name__ == "TabNetClassifier":
                    best_rfc = TabNetClassifier(seed=123,
                                                scheduler_params={"step_size": 10,  # how to use learning rate scheduler
                                                                  "gamma": 0.9},
                                                optimizer_fn=torch.optim.Adam,
                                                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                                verbose=0,
                                                **best_mean_parameter)

                if type(self.model).__name__ == "XGBClassifier":
                    best_rfc = XGBClassifier(random_state=123,
                                             **best_mean_parameter)


            elif type(self.model).__name__ == "RandomForestClassifier":
                best_rfc = RandomForestClassifier(random_state=123,
                                                  n_jobs=self.cpu,
                                                  verbose=0,
                                                  bootstrap=True,
                                                  oob_score=True,
                                                  n_estimators=self.best_num_estimators,
                                                  **best_mean_parameter)

            elif type(self.model).__name__ == "GradientBoostingClassifier":
                best_mean_parameter["n_estimators"] = int(self.best_num_estimators)
                best_rfc = GradientBoostingClassifier(random_state=123,
                                                      **best_mean_parameter)

            elif type(self.model).__name__ == "LGBMClassifier":
                best_mean_parameter["n_estimators"] = int(self.best_num_estimators)
                best_rfc = lgb.LGBMClassifier(random_state=123,
                                              # n_estimators=int(self.best_num_estimators)
                                              **best_mean_parameter)

            elif type(self.model).__name__ == "TabNetClassifier":
                best_rfc = TabNetClassifier(seed=123,
                                            scheduler_params={"step_size": 10, "gamma": 0.9},
                                            optimizer_fn=torch.optim.Adam,
                                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                            verbose=0,
                                            **best_mean_parameter)

            elif type(self.model).__name__ == "XGBClassifier":
                best_mean_parameter["n_estimators"] = int(self.best_num_estimators)
                best_rfc = XGBClassifier(random_state=123,
                                         **best_mean_parameter)

        else:
            print("Warning: Parameter optimization not completed!")

        return best_rfc

    def optimize_with_simulation(self):

        best_mean_parameter = {}
        best_rfc = self.model

        # 1. Simulate Data
        if "Prediction" in self.false.columns:
            self.false.drop(["Prediction"], axis=1, inplace=True)

        X_Train_simulator = Normal_Dist_Simulator(self.x,
                                                  self.y,
                                                  self.false,
                                                  self.label,
                                                  self.num_simu_samples,
                                                  self.prefix,
                                                  self.num_std)

        X_Train_sim, y_Train_sim, meta = X_Train_simulator.simulate_data_by_label()

        # 2. Optimize hyperparameters on Data

        group_sim = X_Train_sim.index.values

        start = time.time()

        best_parameters, best_iterations, final_grid_model = self.RandomSearchCV_iteration(
                                                                                            X=X_Train_sim,
                                                                                            y=y_Train_sim,
                                                                                            group=group_sim,
                                                                                            parameter=generate_hyperparameter_set(
                                                                                                self.model,
                                                                                                X_Train_sim),
                                                                                            fit_params=self.fit_params)

        end = time.time()
        if self.verbose == 1:
            print("time:", round((end - start) / 60, 2), "minutes")

        smalles_best_iteration, best_auc = self.report_best_RandomSearchCV_iteration(parameters=best_parameters,
                                                                                     title="Grid Search Iteration " + str(
                                                                                         self.grid_iteration)
                                                                                     )

        # if self.verbose == 1:
        # print("Best Iteration at ",smalles_best_iteration ,"with",best_auc,"AUROC")

        # Get best model from grid search    
        best_rfc = Optimizer.get_best_parameter_model(self, best_parameters, best_auc, final_grid_model)

        X_train, X_test, y_train, y_test, y_pred_acc, best_rfc_bsmote_sim_sim, false_train_sim, y_pred_proba, y_pred_true = cross_val(
            model=best_rfc,
            cv=self.cv,
            X=X_Train_sim,
            y=y_Train_sim,
            group=group_sim,
            settings=self.fit_params)

        # false_train_sim = eval_results(best_rfc,
        #                             X_Train_sim,
        #                             y_Train_sim,
        #                             "Radiomics Standard BSMOTE oversampling and simulated grid search",
        #                              False)

        return X_Train_sim, y_Train_sim, best_rfc, false_train_sim, best_mean_parameter, meta, final_grid_model

    def calc_train_val_auc(self, model, x, y, x_val, y_val):

        # y_pred_train = model.predict(x)
        # y_pred_val = model.predict(x_val)

        if type(self.model).__name__ == "TabNetClassifier":
            y_pred_train = model.predict_proba(x.values)[:, 1]
            y_pred_val = model.predict_proba(x_val.values)[:, 1]
        else:
            y_pred_train = model.predict_proba(x)[:, 1]
            y_pred_val = model.predict_proba(x_val)[:, 1]

        auc_train = roc_auc_score(y, y_pred_train)
        auc_val = roc_auc_score(y_val, y_pred_val)

        # fpr_train, tpr_train, _ = roc_curve(y, y_pred_train, pos_label=1)
        # fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val, pos_label=1)

        # auc_train = auc(fpr_train, tpr_train)
        # auc_val = auc(fpr_val, tpr_val)

        return auc_train, auc_val

    def calc_train_val_acc(self, model, x, y, x_val, y_val):

        if type(self.model).__name__ == "TabNetClassifier":
            y_pred_train = model.predict(x.values)
            y_pred_val = model.predict(x_val.values)
        else:
            y_pred_train = model.predict(x)
            y_pred_val = model.predict(x_val)

        train_acc = accuracy_score(y_pred_train, y)
        val_acc = accuracy_score(y_pred_val, y_val)

        return train_acc, val_acc

    def repeated_optimize(self):
        """
        Function to repeat optimization

        Parameters
        ----------
        Object from class Optimizer

        Returns
        -------
        Simulated Data: X and y
        False predictions after training
        Parameters of best model
        Distribution of means to catch the effect of data simulation
        """

        meta_data = pd.DataFrame()
        model_storer = pd.DataFrame()

        X_sim = pd.DataFrame()
        y_sim = pd.DataFrame()
        rfc = ""  # RandomForestClassifier(random_state=123)
        false_sim = ""
        paras = ""
        meta = ""
        final_grid_model = ""
        untrained_model = self.model

        print("#### Check Performance before Simulation ####")

        # If we do not have false predictions
        if self.false.empty == True:
            if self.verbose == 1:
                print("No data detected for simulation. Training model for data generation!")

            X_tr, X_te, y_tr, y_te, y_pred, cl, false_train, y_pred_proba, y_pred_true = cross_val(model=self.model,
                                                                                                   cv=self.cv,
                                                                                                   X=self.x,
                                                                                                   y=self.y,
                                                                                                   group=self.groups,
                                                                                                   verbose=self.verbose,
                                                                                                   settings=self.fit_params)
            if (self.x_val == None) and (self.y_val == None):
                print("No validation data detected. Using left out split data from cross-validation for validation!")
                self.x_val = X_te
                self.y_val = y_te
                self.x = X_tr
                self.y = y_tr
                self.groups = self.x.index.values

            self.false = false_train

            if type(self.model).__name__ == "TabNetClassifier":
                # reset model
                self.model = untrained_model

                # Check Grid search without simulation
        if self.prefit == True:
            if type(self.model).__name__ == "TabNetClassifier":
                fit = self.model.fit(self.x.values, self.y)
            else:
                self.model = self.model.fit(self.x, self.y)

            auc_train, auc_val = Optimizer.calc_train_val_auc(self, self.model, self.x, self.y, self.x_val, self.y_val)
            acc_train, acc_val = Optimizer.calc_train_val_acc(self, self.model, self.x, self.y, self.x_val, self.y_val)

            data = {"iteration": [-2], "model": [self.model], "train_AUC": [auc_train], "val_AUC": [auc_val],
                    "train_ACC": [acc_train],
                    "val_ACC": [acc_val]}  # Take all best model configurations of eacht optimization iteration

            # Create DataFrame
            model_storer = pd.DataFrame(data)

            best_parameters, best_iterations, final_grid_model = Optimizer.RandomSearchCV_iteration(self,
                                                                                                    X=self.x,
                                                                                                    y=self.y,
                                                                                                    group=self.groups,
                                                                                                    parameter=generate_hyperparameter_set(
                                                                                                        self.model,
                                                                                                        self.x),
                                                                                                    fit_params=self.fit_params)

            smalles_best_iteration, best_auc = self.report_best_RandomSearchCV_iteration(parameters=best_parameters,
                                                                                         title="Pre Simulation")

            if type(self.model).__name__ == "TabNetClassifier":
                fit = final_grid_model.fit(self.x.values, self.y)
                best_model = final_grid_model
                auc_train, auc_val = Optimizer.calc_train_val_auc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)
                acc_train, acc_val = Optimizer.calc_train_val_acc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)

            elif type(self.model).__name__ == "LGBMClassifier":
                fit = final_grid_model.fit(self.x.values, self.y)
                best_model = final_grid_model
                auc_train, auc_val = Optimizer.calc_train_val_auc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)
                acc_train, acc_val = Optimizer.calc_train_val_acc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)

            elif type(self.model).__name__ == "XGBClassifier":
                fit = final_grid_model.fit(self.x.values, self.y)
                best_model = final_grid_model
                auc_train, auc_val = Optimizer.calc_train_val_auc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)
                acc_train, acc_val = Optimizer.calc_train_val_acc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)
            else:
                self.model = self.model.fit(self.x, self.y)
                best_model = final_grid_model.best_estimator_
                auc_train, auc_val = Optimizer.calc_train_val_auc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)
                acc_train, acc_val = Optimizer.calc_train_val_acc(self, best_model, self.x, self.y, self.x_val,
                                                                  self.y_val)

            data = {"iteration": [-1],
                    "model": [best_model],
                    "train_AUC": [auc_train],
                    "val_AUC": [auc_val],
                    "train_ACC": [acc_train],
                    "val_ACC": [acc_val]}

            data_df = pd.DataFrame(data)
            model_storer = pd.concat([model_storer, data_df], ignore_index=True)

            # Get best model from grid search    
            best_rfc = Optimizer.get_best_parameter_model(self, best_parameters, best_auc, final_grid_model)

            X_tr, X_te, y_tr, y_te, y_pred, cl, false_train = cross_val(model=best_rfc,
                                                                        cv=self.cv,
                                                                        X=self.x,
                                                                        y=self.y,
                                                                        group=self.groups,
                                                                        verbose=self.verbose,
                                                                        settings=self.fit_params)

            # reset model
            # self.model = untrained_model

        print("#### Start Simulation ####")
        for i in range(self.num_simulations):

            self.grid_iteration = i
            # print(self.x)
            # Oversampling
            if self.oversample:
                # if the classes are already balanced skip oversampling
                if self.y.value_counts()[0] != self.y.value_counts()[1]:
                    if self.verbose == 1:
                        print("Input Data imbalanced! Apply Oversampling...")
                    train_Oversampler = Oversampler(self.x,
                                                    self.y,
                                                    1.0,
                                                    i,
                                                    self.label,
                                                    self.index,
                                                    self.verbose)

                    # if we have a minimum of sample numbers
                    if ((self.y.value_counts()[0] > 3) & (self.y.value_counts()[1] > 3)):
                        X_resampled, y_resampled = train_Oversampler.BSMOTE_Oversampling()

                        self.x = X_resampled
                        self.y = y_resampled
                    else:
                        if self.verbose == 1:
                            print("Error: Too few samples for Oversampling! Skipping Oversampling!")

                # if we have false predictions
                # print(len(self.false[self.false[self.label]==0])> 0 & len(self.false[self.false[self.label]==1]) > 0)
                if (len(self.false[self.false[self.label] == 0]) > 0) & (
                        len(self.false[self.false[self.label] == 1]) > 0):
                    if self.false[self.label].value_counts()[0] != self.false[self.label].value_counts()[1]:
                        if ((self.false[self.label].value_counts()[0] > 3) & (
                                self.false[self.label].value_counts()[1] > 3)):
                            if self.verbose == 1:
                                print("False positives/negatives Data imbalanced! Apply Oversampling...")

                            X_false = self.false
                            y_false = self.false[self.label]

                            train_Oversampler = Oversampler(X_false,
                                                            y_false,
                                                            1.0,
                                                            i,
                                                            self.label,
                                                            self.index)

                            try:
                                X_resampled, y_resampled = train_Oversampler.BSMOTE_Oversampling()

                                self.false = X_resampled
                                self.false[self.label] = y_resampled
                            except:
                                print("Warning: Oversampling Failed - Check your Data!")
            # Data Simulation
            if (len(self.false[self.false[self.label] == 0]) > 0) & (len(self.false[self.false[self.label] == 1]) > 0):

                # Take false preditions for oversampling
                if ((self.false[self.label].value_counts()[0] >= 2) & (self.false[self.label].value_counts()[1] >= 2)):
                    X_sim, y_sim, rfc, false_sim, paras, meta, final_grid_model = Optimizer.optimize_with_simulation(
                        self)

                    meta = meta.rename(('{}_' + str(i)).format)
                    meta_data = pd.concat([meta_data, meta])

                    self.x = X_sim
                    self.y = y_sim
                    self.false = false_sim
                    new_prefix = int(self.prefix) - 1
                    self.prefix = str(new_prefix)

                # Few samples with label 0 in fals preditions
                elif ((len(self.false[self.false[self.label] == 0]) < 2) & (
                        len(self.false[self.false[self.label] == 1]) > 2)):
                    # add values from total dataset with only label 0 
                    additional = self.x[self.y == 0]
                    additional.loc[:, self.label] = 0
                    # additional[self.label] = 0
                    complete = pd.concat([self.false, additional])
                    self.false = complete

                    X_sim, y_sim, rfc, false_sim, paras, meta, final_grid_model = Optimizer.optimize_with_simulation(
                        self)

                    meta = meta.rename(('{}_' + str(i)).format)
                    meta_data = pd.concat([meta_data, meta])

                    self.x = X_sim
                    self.y = y_sim
                    self.false = false_sim
                    new_prefix = int(self.prefix) - 1
                    self.prefix = str(new_prefix)

                # less samples with label 1 in fals preditions
                elif ((len(self.false[self.false[self.label] == 0]) > 2) & (
                        len(self.false[self.false[self.label] == 1]) < 2)):
                    # add values from total dataset with only label 1
                    additional = self.x[self.y == 1]
                    additional.loc[:, self.label] = 1
                    # additional[self.label] = 1
                    complete = pd.concat([self.false, additional])
                    self.false = complete

                    X_sim, y_sim, rfc, false_sim, paras, meta, final_grid_model = Optimizer.optimize_with_simulation(
                        self)

                    meta = meta.rename(('{}_' + str(i)).format)
                    meta_data = pd.concat([meta_data, meta])

                    self.x = X_sim
                    self.y = y_sim
                    self.false = false_sim
                    new_prefix = int(self.prefix) - 1
                    self.prefix = str(new_prefix)

                elif ((len(self.false[self.false[self.label] == 0]) < 2) & (
                        len(self.false[self.false[self.label] == 1]) < 2)):
                    # add values from total dataset with label 0 & 1
                    # add values with label 0
                    additional = self.x[self.y == 0]
                    additional.loc[:, self.label] = 0
                    # additional[self.label] = 0
                    complete = pd.concat([self.false, additional])
                    self.false = complete

                    # add values with label 1
                    additional = self.x[self.y == 1]
                    additional.loc[:, self.label] = 1
                    # additional[self.label] = 1
                    complete = pd.concat([self.false, additional])
                    self.false = complete

                    X_sim, y_sim, rfc, false_sim, paras, meta, final_grid_model = Optimizer.optimize_with_simulation(
                        self)

                    meta = meta.rename(('{}_' + str(i)).format)
                    meta_data = pd.concat([meta_data, meta])

                    self.x = X_sim
                    self.y = y_sim
                    self.false = false_sim
                    new_prefix = int(self.prefix) - 1
                    self.prefix = str(new_prefix)

            else:
                if self.verbose == 1:
                    print("Warning: Too few false samples for Simulation! Use entire Dataset for Simulation!")

                # add values from total dataset with label 0
                additional = self.x[self.y == 0]
                additional.loc[:, self.label] = 0
                # additional[self.label] = 0
                complete = pd.concat([self.false, additional])
                self.false = complete

                # add values from total dataset with label 1
                additional = self.x[self.y == 1]
                additional.loc[:, self.label] = 1
                # additional[self.label] = 1
                complete = pd.concat([self.false, additional])
                self.false = complete

                # make bigger standart deviation for the usage of all data
                prev_std = self.num_std
                self.num_std = 2
                X_sim, y_sim, rfc, false_sim, paras, meta, final_grid_model = Optimizer.optimize_with_simulation(self)

                meta = meta.rename(('{}_' + str(i)).format)
                meta_data = pd.concat([meta_data, meta])

                self.x = X_sim
                self.y = y_sim
                self.false = false_sim
                new_prefix = int(self.prefix) - 1
                self.prefix = str(new_prefix)
                self.num_std = prev_std

            auc_train, auc_val = Optimizer.calc_train_val_auc(self, rfc, X_sim, y_sim, self.x_val, self.y_val)
            acc_train, acc_val = Optimizer.calc_train_val_acc(self, rfc, X_sim, y_sim, self.x_val, self.y_val)
            tmp = {"iteration": [i], "model": [rfc], "train_AUC": [auc_train], "val_AUC": [auc_val],
                   "train_ACC": [acc_train], "val_ACC": [acc_val]}
            tmp_ = pd.DataFrame(tmp)
            model_storer = pd.concat([model_storer, tmp_], ignore_index=True)

        return X_sim, y_sim, model_storer, false_sim, paras, meta_data, final_grid_model

    def generate_hyperparameter_set(self, model, X_Train, tabnet_unsup_pretraining=np.nan):

        all_available_hyperparameter = HyperparameterConfigurator(model=model,
                                                                  optimizer_lib=None,
                                                                  logger=None,
                                                                  error=None,
                                                                  out_path="",
                                                                  interested_hyperparameter=None,
                                                                  extended_parameter_set=None)

        return all_available_hyperparameter


def get_best_setting(grip_parameters, smallest_best_auc):
    best_params = {}

    for iter_ in grip_parameters:
        if iter_["iter"] == smallest_best_auc:
            best_params = iter_["params"]

    return best_params


def best_average_setting(parameters, max_auc=0, first_n=None):
    # make dataframe out of it
    best_params = pd.DataFrame(parameters)
    best_average_params = pd.DataFrame()

    # Take Dataset from Gridsearch with best AUC
    if first_n == None:
        # filter for settings with best auc
        best_params = best_params[best_params['AUC'] == max_auc]
        # get params from best settings
        # best_parameter =  pd.DataFrame(list(best_params["params"]))

    # Take cv_resusts from Grid search and make avergae setting from the best n settings
    else:
        best_params.sort_values(by=["rank_test_roc_auc"], inplace=True)
        best_params = best_params.iloc[:first_n]

    # get params from best settings
    best_parameter = pd.DataFrame(list(best_params["params"]))
    # calc best mean params
    best_average_params = calc_averge_param(best_parameter)

    return best_average_params


def calc_averge_param(dict_):
    mean_para = {}
    sum_para = pd.DataFrame(dict_)

    # Calculate the average parameter setting
    for i in sum_para:
        # set object parameter to categorical parameter
        if sum_para[i].dtype == "object":
            if i != "init":
                if None == sum_para[i].values.all():
                    mean_para[i] = None
                    continue
                elif None == sum_para[i].values.any():
                    sum_para[i].loc[sum_para[i] == None] = 0
                elif type(sum_para[i]) == "dict":
                    continue

                sum_para[i] = sum_para[i].astype('category')
        if sum_para[i].dtype == "bool":
            mean_para[i] = sum_para[i].value_counts().idxmax()
        if is_categorical(sum_para[i]):
            mean_para[i] = sum_para[i].value_counts().idxmax()
        if sum_para[i].dtype == "float64":
            mean_para[i] = round(sum_para[i].mean(), 2)
        if sum_para[i].dtype == "int64":
            mean_para[i] = int(sum_para[i].mean())

        # for random forest we have some parameter which need to be in a range
        if i == "min_samples_leaf":
            if (mean_para[i] < 0.5 or mean_para[i] == 1):
                continue
            else:
                if (round(mean_para[i], 1) == 0.5 or mean_para[i] < 0.75):
                    mean_para[i] = 0.5
                elif round(mean_para[i], 0) == 1:
                    mean_para[i] = 1
        if i == "min_samples_split":
            if (int(mean_para[i]) == 1):
                mean_para[i] = 0.8
                continue
            if (int(mean_para[i]) > 1):
                mean_para[i] = int(mean_para[i])
                continue
        if i == "max_leaf_nodes":
            mean_para[i] = int(mean_para[i])
        if i == "max_depth":
            mean_para[i] = int(mean_para[i])
        if i == "n_iter_no_change":
            mean_para[i] = int(mean_para[i])
        if i == "tol":
            if mean_para[i] == 0.0:
                mean_para[i] = float(1e-4)
        if i == "learning_rate":
            if mean_para[i] < 0.0:
                print("learning rate too small!", "Set to default!")
                mean_para[i] = 0.1

    return mean_para


def is_categorical(array_like):
    return array_like.dtype.name == 'category'
