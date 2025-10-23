import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go

import collections
import sklearn
import numpy as np
import time
import logging
import os
import shap
import seaborn as sns
import re
from collections import defaultdict
from typing import Optional, List, Tuple, Sequence, Union, Dict


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import RocCurveDisplay, accuracy_score, balanced_accuracy_score, mean_squared_error, \
    roc_auc_score, average_precision_score, f1_score, confusion_matrix, auc

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.plotting import plot_decision_regions
import pickle5 as pickle
from collections import defaultdict

from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.model_training.Optimizer import Optimizer
from rptk.src.feature_extraction.FeatureConsensus import ConsensusFeatureFormatter

class PerformancePlotter:
    """
    Class to generate Performance plots for evaluation of model performance.
    :param output_path (str): Path to the output folder to save the plots.
    :param X (pd.DataFrame): Data where predictions are based on.
    :parma y (pd.Series): Data label for samples in X
    :param y_pred (pd.Series): Data label from samples in unseen test set or validation set
    :param RunID (str): ID for RPTK run
    :param error (logger): logging errors
    :param logger (logger): logging processing
    """

    def __init__(self,
                 output_path,
                 X=None,
                 y=None,
                 y_pred: Optional[pd.Series] = None,
                 RunID: str = None,
                 error: logging.Logger = None,
                 logger: logging.Logger = None,
                 task: str = None):

        self.output_path = output_path
        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.RunID = RunID
        self.error = error
        self.logger = logger
        self.task = task

        self.SEP = r"[_\-]"

        if self.RunID is None:
            self.error.error("No RunID provided! Please define a RunID before running.")
            raise ValueError("No RunID provided! Please define a RunID before running.")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        if self.output_path.endswith("/"):
            self.output_path = self.output_path[:-1]

        if self.logger is None:
            self.logger = LogGenerator(
                log_file_name=self.output_path + "/RPTK_performance_plotter_" + self.RunID + ".log",
                logger_topic="RPTK Performance plotting"
            ).generate_log()

        if self.error is None:
            self.error = LogGenerator(
                log_file_name=self.output_path + "/RPTK_performance_plotter_" + self.RunID + ".err",
                logger_topic="RPTK Performance plotting error"
            ).generate_log()

        # ---- Feature Naming Rules ----
        # Multi-token (sequence) rules — applied first
        self.MIRP_MULTI: list[tuple[Optional[tuple[str, ...] , str], str]] = [
            (("rlnu","norm"), "RLNUNorm"),
            (("glnu","norm"), "GLNUNorm"),
            (("zs","entr"), "ZSEntr"),
            (("zs","var"), "ZoneSizeVariance"),
            (("peak","glob"), "GIPeak"),                 # Global intensity peak
            (("zd","var"), "ZDVar"),                     # Zone distance variance
            (("diff","entr"), "DiffEntropy"),
            (("integ","int"), "IntegInt"),               # Integrated intensity
            (("gauss","s2.0"), "Gauss"),
            (("s","3.0","g","1.0","l","0.9","t","0.0"), ""), # drop this marker sequence
            (("inv","diff","mom","norm"), "IDMN"),
            (("info","corr2"), "InfoMeaCorr2"),
            (("lgce",), "LGCE"),                         # Low grey level count emphasis
            (("hdlge",), "HDLGLE"),                      # High dependence low grey level emphasis
            (("int","mean","int","init","roi"), "MeanInitRoi"),
            (("diff","i25","i75"), "InterquartileRange"),
            (("clust","prom"), "ClusterProminence"),
            (("zd","entr"), "ZeroDistEntropy"),
            (("inv","var"), "InVar"),
            (("vol","dens","aabb"), "VolumeDensityAABB"),
            (("grad","g"), "GradientMagnitude"),
            (("joint","max"), "JointMax"),
            (("diff","mean"), "DifferenceMean"),
            (("dc","energy"), "JointEnergy"),
            (("int","bb","dim","y","init","roi"), "BoundingBoxDimY_InitRoi"),
            (("mean","d","15"), "Mean"),
            (("Mean","d","15"), "Mean"),
            (("2d","s"), ""),
        ]

        # Single-token rules — whole-token only, case-insensitive
        self.MIRP_SINGLE: dict[str,  Optional[str , None]] = {
            # First-order / aggregations
            "avg": "Mean", "av": "Mean", "mean": "Mean",
            "median": "Median",
            "max": "Max", "min": "Min", "range": "Range",
            "kurt": "Kurtosis", "mode": "Mode", "skew": "Skewness",
            "qcod": "QuantCoefDisp",
            "diff_i25_i75": "InterquartileRange",
            "entropy": "Entropy",
            "var": "Variance",

            # Texture abbreviations
            "invar": "InVar",
            "InverseVariance": "InVar",
            "hde": "DifferenceEntropy",
            "zd_entr": "ZeroDistanceEntropy",
            "ldlge": "LDLGE", "sdlge": "SDLGE", "szlge": "SZLGE", "lzlge": "LZLGE",
            "lrhge": "LRHGE", "ldhge": "LDHGE",
            "glnu": "GLNU", "zdnu": "ZDNU",
            "lgre": "LGLRE", "szhge": "SZHGLE",
            "sphericity": "Sphericity",
            "energy": "Energy",
            "exponential":"Exponential",
            "complexity":"Complexity",
            "lre":"LRE",

            # Preprocessing / transforms / markers
            "sde": "SDE",                  # SmallDistanceEmphasis
            "logarithm": "Log",
            "squareroot": "SquareRoot",
            "square": "Square",
            "delta": "Delta",
            "gabor": "Gabor",
            "laws": "Laws",
            "lbp-2d": "LBP2D",
            "wavelet-HHH": "WaveletHHH",
            "wavelet-HHL": "WaveletHHL",
            "wavelet-HLH": "WaveletHLH",
            "wavelet-LLH": "WaveletLLH",
            "wavelet-HLL": "WaveletHLL",
            "wavelet-LHH": "WaveletLHH",
            "wavelet-LLL": "WaveletLLL",
            "wavelet-LHL": "WaveletLHL",
            "wavelet-hhh": "WaveletHHH",
            "wavelet-hhl": "WaveletHHL",
            "wavelet-hlh": "WaveletHLH",
            "wavelet-llh": "WaveletLLH",
            "wavelet-hll": "WaveletHLL",
            "wavelet-lhh": "WaveletLHH",


            # Abbreviations appearing as standalone tokens to drop
            "fbs": "", "mrg": "", "w25.0": "", "l5s5e5": "", "e5e5e5": "",
            # Lone 'v' marker (voxel/volumetric flag) — drop only if its own token
            "v": "",

            # Misc
            "cov": "Covariance",
            # ROI / scalers
            # "peritumoral": "margin",
            "zscore": "",

            # Families (IBSI-consistent)
            "cm": "GLCM",
            "rlm": "GLRLM",
            "szm": "GLSZM",
            "dzm": "GLDZM",
            "ngt": "NGTDM",
            "ngl": "NGLDM",

            # Other prefixes
            "ih": "IH",
            "morph": "Morph",
            "stat": "Stat",
            "loc": "LI",

            # Keep geometry tokens as is
            "d1": "", "2d": "2D", "3d": "3D",
            "a0.0": "", "a0": "",
        }

        # Multi-token rules — applied first
        self.PRAD_MULTI: list[tuple[Optional[tuple[str, ...] , str], str]] = [
            (("gauss","s2.0"), "Gauss"),
            (("s","3.0","g","1.0","l","0.9","t","0.0"), ""),  # drop this marker sequence
            (("mean","d","15"), "Mean"),
            (("Mean","d","15"), "Mean"),
            (("intesity","based","statistics"), "Firstorder"),
            (("Root","mean","square"), "RootMeanSquare"),
        ]

        # Single-token rules — whole-token only, case-insensitive
        self.PRAD_SINGLE: dict[str, Optional[str | None]] = {
            # Core renames / abbreviations to canonical
            "InverseVariance": "InVar",
            "GrayLevelNonUniformityNormalized": "GLNUNorm",
            "LargeDependenceLowGrayLevelEmphasis": "LDLGE",
            "DependenceEntropy": "DepEntropy",
            "LongRunHighGrayLevelEmphasis": "LRHGE",
            "SmallDependenceEmphasis": "SDE",
            "Idmn": "IDMN",
            "LowGrayLevelEmphasis": "LGE",
            "LowGrayLevelZoneEmphasis":"LGZE",
            "DependenceNonUniformityNormalized":"DNUNorm",
            "QuantileCoeffDispersion":"QuantCoefDisp",
            "zscore": "",
            "glcm": "GLCM",
            "glrlm": "GLRLM",
            "glszm": "GLSZM",
            "gldm": "GLDZM",
            "ngtdm": "NGTDM",
            "ngldm": "NGLDM",
            "firstorder":"Firstorder",
            "7":"",
            "invar":"Invar",

            # Transforms / filters
            "gabor": "Gabor",
            "laws": "Laws",
            "lbp-2d": "LBP2D",
            "energy": "Energy",
            "delta": "Delta",
            "square": "Square",
            "logarithm": "Log",
            "squareroot": "SquareRoot",
            "original":"",
            # "peritumoral": "margin",

            # Wavelets (normalize mixed casings)
            "wavelet-HHH": "WaveletHHH",
            "wavelet-HHL": "WaveletHHL",
            "wavelet-HLH": "WaveletHLH",
            "wavelet-LLH": "WaveletLLH",
            "wavelet-HLL": "WaveletHLL",
            "wavelet-LHH": "WaveletLHH",
            "wavelet-LLL": "WaveletLLL",
            "wavelet-LHL": "WaveletLHL",
            "wavelet-hhh": "WaveletHHH",
            "wavelet-hhl": "WaveletHHL",
            "wavelet-hlh": "WaveletHLH",
            "wavelet-llh": "WaveletLLH",
            "wavelet-hll": "WaveletHLL",
            "wavelet-lhh": "WaveletLHH",

            # Noisy markers to drop when they are standalone tokens
            "e5e5e5": "",
            "l5s5e5": "",
        }




    def plot_sfs_features(self, sfs=None, title: str = "Forward"):
        """
        Plot performance between feature sets.
        :param sfs: SequentialFeatureSelector object with estimator
        :param title: Title of Forward or Backward direction of sequential feature selection
        """
        if sfs is None:
            self.error.warning("missing SequentialFeatureSelector object for plotting!")
            return

         # 1) Replace underscores with spaces
        clean_title = title.replace("_", " ")
        # 2) Remove any standalone 'mlxtend' (case‐insensitive)
        clean_title = re.sub(r'\bmlxtend\b', '', clean_title, flags=re.IGNORECASE)
        # 3) Collapse multiple spaces and strip
        clean_title = " ".join(clean_title.split())

        # This returns an AxesSubplot
        fig, ax = plot_sfs(
            sfs.get_metric_dict(),
            kind='std_dev',
            ylabel="AUROC",
            figsize=(18, 15),
        )

        # 1) y-limits & ticks every 0.1
        ax.set_ylim(0.1, 1.0)
        ax.set_yticks(np.arange(0.1, 1.01, 0.1))

        # 2) larger axis labels
        ax.set_xlabel("Number of Features", fontsize=14, labelpad=8)
        ax.set_ylabel("AUROC (±1 SD)", fontsize=14, labelpad=8)

        # 3) bigger title
        ax.set_title(f"Sequential {clean_title} Selection Results", fontsize=18, pad=12)

        # Enlarge tick labels:
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)

        # optional: dashed grid
        ax.grid(which='major', linestyle='--', alpha=0.7)

        # save & close
        fig.tight_layout()
        outfile = f"{self.output_path}/{self.RunID}_Sequential_{title}_Selection.png"
        fig.savefig(outfile, bbox_inches='tight')
        plt.close(fig)

    def plot_conv_matrix(self, title, label, y=None, y_pred=None, path=None, fold=None):
        """"
        Plot confusion matrix for evaluation of the predictions for a binary classification
        :param title: Title of the plot
        :param y: True labels
        :param y_pred: Predicted labels
        :param label: Label of the plot
        :param path: Path to save the plot
        """

        if path is None:
            path = self.output_path

        if y is None:
            y = self.y

        if y_pred is None:
            y_pred = self.y_pred

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # plot conv matrix for evaluation
        cf_matrix = confusion_matrix(y.values, y_pred)

        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]

        # calculate percentage relative to group
        p1 = int(group_counts[0]) / (int(group_counts[0]) + int(group_counts[1]))
        p2 = int(group_counts[1]) / (int(group_counts[0]) + int(group_counts[1]))
        p3 = int(group_counts[2]) / (int(group_counts[2]) + int(group_counts[3]))
        p4 = int(group_counts[3]) / (int(group_counts[2]) + int(group_counts[3]))

        group_percentages = [p1, p2, p3, p4]

        group_percentages = ["{0:.2%}".format(value) for value in group_percentages]

        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_counts, group_percentages)]

        labels = np.asarray(labels).reshape(2, 2)

        # create the figure
        plt.figure()
        ax = sns.heatmap(cf_matrix,
                         annot=labels,
                         fmt='',
                         cmap='Blues',
                         annot_kws={"fontsize": 12}
                         )

        ax.set_title(title + '\n', fontsize=18)
        ax.set_xlabel('\nPredicted Values', fontsize=12)
        ax.set_ylabel('Actual Values ', fontsize=12)

        # Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(label)
        ax.yaxis.set_ticklabels(label)

        plt.tight_layout()

        if fold is not None:
            # Save the visualization of the Confusion Matrix.
            plt.savefig(path + '/' + self.RunID + '_confusion_matrix_' + str(title) + "_" + str(fold) + '.png',
                        bbox_inches='tight')
        else:
            # Save the visualization of the Confusion Matrix.
            plt.savefig(path + '/' + self.RunID + '_confusion_matrix_' + str(title) + '.png', bbox_inches='tight')

        plt.close()
        # Clear the figure to free memory if plotting multiple figures in a loop or a script
        plt.clf()

    def plot_AUC(self, model, X_test, y_test, model_name, out_path=None):
        """
        Plot AUROC from estimator
        """
        real_model_name = str(type(model).__name__)
        
        fig, ax = plt.subplots(figsize=(10,10))

        if out_path is None:
            out_path = self.output_path

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        
        if "TabNet" in type(model).__name__:
            y_pred = model.predict_proba(X_test.values)[:, 1]
            viz = RocCurveDisplay.from_predictions(
                                    y_pred=y_pred,
                                    y_true=y_test,
                                    name=str(type(model).__name__),
                                    pos_label=1,
                                    lw=5)
            
        elif "Ensemble" in  type(model).__name__:
                if "TabNet" in type(model.clfs[1]).__name__:
                    y_pred = model.predict_proba(X_test.values)[:, 1]
                    viz = RocCurveDisplay.from_predictions(
                                            y_pred=y_pred,
                                            y_true=y_test,
                                            name="Ensemble " + str(type(model.clfs[1]).__name__),
                                            pos_label=1,
                                            lw=5)
                elif "Ensemble" in  type(model.clfs[1]).__name__:
                    if "TabNet" in type(model.clfs[1].clfs[1]).__name__:
                        y_pred = model.predict_proba(X_test.values)[:, 1]
                        viz = RocCurveDisplay.from_predictions(
                                                y_pred=y_pred,
                                                y_true=y_test,
                                                name="Ensemble of Ensemble " + str(type(model.clfs[1].clfs[1]).__name__),
                                                pos_label=1,
                                                lw=5)
                else:
                    if "Ensemble" in str(type(model).__name__):
                        simple_model_name = str(type(model.clfs[0]).__name__)
                        if "RandomForest" in simple_model_name:
                            tmp_model_name = simple_model_name.replace("Classifier", "")
                            real_model_name = "Ensemble " + tmp_model_name
                        else:
                            real_model_name = "Ensemble " + str(type(model.clfs[0]).__name__)
                    else:
                        if "RandomForest" in str(type(model).__name__):
                            real_model_name = str(type(model).__name__).replace("Classifier", "")
                        else:
                            real_model_name = str(type(model).__name__)

                    y_pred = model.predict_proba(X_test)[:, 1]
                    viz = RocCurveDisplay.from_predictions(
                                            y_pred=y_pred,
                                            y_true=y_test,
                                            name=real_model_name,
                                            pos_label=1,
                                            lw=5)
                
                
        else:
            if "RandomForest" in str(type(model).__name__):
                real_model_name = str(type(model).__name__).replace("Classifier", "")
            else:
                real_model_name = str(type(model).__name__)
            viz = RocCurveDisplay.from_estimator(model, 
                                                 X_test, 
                                                 y_test, 
                                                 name=real_model_name,
                                                 pos_label=1, 
                                                 lw=5)

        if "Ensemble" in str(type(model).__name__):
            simple_model_name = str(type(model.clfs[0]).__name__)
            if "RandomForest" in simple_model_name:
                tmp_model_name = simple_model_name.replace("Classifier", "")
                real_model_name = "Ensemble " + tmp_model_name
        else:
            simple_model_name = str(type(model).__name__)
            real_model_name
            
        if "RandomForest" in simple_model_name:
            simple_model_name = simple_model_name.replace("Classifier", "")
            
        # plt.figure(figsize=(20,20))

        viz.plot(ax=ax, linewidth=8, color="deeppink")

        ax.set_title("Test ROC from " + simple_model_name, fontsize=32)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0, fontsize=22)

        #plt.title("Test AUROC from " + simple_model_name, fontsize=18)
        #plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0, fontsize=18)
        #plt.tight_layout()

        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)

        ax.set_xlabel("False Positive Rate", fontsize=26)
        ax.set_ylabel("True Positive Rate", fontsize=26)

        # Tick intervals (here every 0.1)
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))

        #plt.xticks(fontsize=14)
        #plt.yticks(fontsize=14)
        
        #plt.xlabel("False Positive Rate (Positive label: 1)", fontsize=18)
        #plt.ylabel("True Positive Rate (Positive label: 1)", fontsize=18)
    
        fig.savefig(out_path + '/' + model_name + "_test_ROC_curve.png") #, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

    def get_label_counts(self, series):
        """
        Get numbers of counts from binary label counting
        :param series: with value counts from labels 0 and 1
        :return: number of labels 0 and 1
        """
        series_1 = 0
        series_0 = 0

        if 1 in series.index:
            series_1 = series.at[1]

        if 0 in series.index:
            series_0 = series.at[0]

        return series_1, series_0

    def plot_hyperparameter_validation_curve(self, train_score_Num, test_score_Num, model, param_name, param_values):
        """
        Plot validation curve from Cross Validation with mean AUC and standard deviation
        to get a stable performance
        :param train_score_Num (list): training scores
        :param test_score_Num (list): testing scores
        :param model: model which has been fitted to optimize
        :param param_name (str): Name of the parameter to get stable performance from
        :param param_values (list): list of parameters from optimization
        """

        x_stable = 0
        y_stable = 0
        x_max = 0
        y_max = 0

        train_scores_mean = np.mean(train_score_Num, axis=1)
        train_scores_std = np.std(train_score_Num, axis=1)
        test_scores_mean = np.mean(test_score_Num, axis=1)
        test_scores_std = np.std(test_score_Num, axis=1)

        if "TabNet" in type(model).__name__:
            train_scores_mean = np.mean(train_score_Num, axis=0)
            train_scores_std = np.std(train_score_Num, axis=0)
            test_scores_mean = np.mean(test_score_Num, axis=0)
            test_scores_std = np.std(test_score_Num, axis=0)

        plt.title("Validation Curve with " + str(type(model).__name__))
        plt.xlabel(param_name)
        plt.ylabel("AUROC")
        plt.ylim(0.0, 1.1)

        lw = 2

        cv_color = np.random.rand(3, )

        plt.plot(
            param_values, test_scores_mean, label="Cross-validation score", color=cv_color, lw=lw
        )

        plt.fill_between(
            param_values,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color=cv_color,
            lw=lw,
        )

        x_max, y_max = Optimizer().annot_max(x=param_values, y=test_scores_mean)

        if isinstance(Optimizer().get_stable_idx(test_scores_mean), (int, float)):
            if Optimizer().get_stable_idx(test_scores_mean) != 0:
                x_stable, y_stable = Optimizer().annot_stable(x=param_values, y=test_scores_mean)
            else:
                self.error.warning("No Stable Result Found! Please extend the parameter range.")
                print("Warning: No Stable Result! Please extend the parameter range.")
        else:
            x_stable, y_stable = Optimizer().annot_stable(x=param_values, y=test_scores_mean)

        # if self.n_train == 1:
        plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0)
        # Name for plot: name of model, opti or only est opt and number of estimator + date
        plt.tight_layout()

        plt.savefig(self.output_path + '/' + str(type(model).__name__) + "_" + str(x_stable) + "_" + str(
            round(y_stable, 2)) + "_num_est_curve.png", bbox_inches='tight')
        plt.close()
        plt.clf()

        return x_max, y_max, x_stable, y_stable

    @staticmethod
    def plot_summary_auroc(val_auc_fold_models, test_auc_fold_models, test_auc_ensemble_models, data, output_path=None, plot_title=None):
        """
        Plots a summary of AUROC (Area Under the Receiver Operating Characteristic Curve) values 
        for individual fold models, test models, and ensemble models.

        Parameters:
        - val_auc_fold_models: dict, AUC values for validation folds.
        - test_auc_fold_models: dict, AUC values for test folds.
        - test_auc_ensemble_models: dict, AUC values for ensemble models.
        - data: DataFrame, contains information about ensemble models.
        - output_path: str, optional, path to save the plot.

        Returns:
        - Saves a PNG image of the plot if output_path is provided.
        """
        print("test_auc_ensemble_models", test_auc_ensemble_models)
        desired_order = [
                            "RandomForest",
                            "TabNet",
                            "GradientBoosting",
                            "LGBM",
                            "XGB",
                            "SVC",
                        ]

        ordered_test_auc_ensemble_models = {}
        for ordered_model in desired_order:
            for model in test_auc_ensemble_models:
                if str(ordered_model) in str(model):
                    ordered_test_auc_ensemble_models[ordered_model] = test_auc_ensemble_models[model]
                    pass

        ordered_val_auc_fold_models = {}
        for ordered_model in desired_order:
            for model in val_auc_fold_models:
                if str(ordered_model) in str(model):
                    ordered_val_auc_fold_models[ordered_model] = val_auc_fold_models[model]
                    pass

        ordered_test_auc_fold_models = {}
        for ordered_model in desired_order:
            for model in test_auc_fold_models:
                if str(ordered_model) in str(model):
                    ordered_test_auc_fold_models[ordered_model] = test_auc_fold_models[model]
                    pass

        # Convert validation and test AUROC data into a list of dictionaries
        fold_val_data = [{"Model": model, "AUC_Type": "Validation_AUC", "AUC": auc} 
                        for model, aucs in ordered_val_auc_fold_models.items() for auc in aucs]
        
        fold_test_data = [{"Model": model, "AUC_Type": "Test_AUC", "AUC": auc} 
                        for model, aucs in ordered_test_auc_fold_models.items() for auc in aucs]
        
        # Extract ensemble model data
        ensemble_test_data = [{"Model": row["Models"], "AUC_Type": "Ensemble_Test_AUC", 
                            "Test_AUC": row["Test_AUC"], "AUC": row["Bootstrap_AUC"], 
                            "Std": row["Bootstrap_AUC_std"]}
                            for _, row in data[data["Models"].str.contains("SoftEnsemble")].iterrows()]

        correct_ensemble_test_data = []

        # Correct ensemble model names and match AUROC values
        for j in ensemble_test_data:
            for i in ordered_test_auc_ensemble_models:
                for model in desired_order:
                    if model in j["Model"]:
                        if i in j["Model"]:  # If ensemble name matches
                            j["Test_AUC"] = float(j["Test_AUC"])
                            j["AUC"] = float(j["AUC"])
                            j["Std"] = float(j["Std"])
                            #if float(j["Test_AUC"]) == float(test_auc_ensemble_models[i][0]):  # Ensure AUROC values match
                            j["Model"] = i  # Rename model
                            correct_ensemble_test_data.append(j)
                            break

        # Create DataFrame from validation and test data
        plot_data = pd.DataFrame(fold_val_data + fold_test_data)

        plot_data["Model"] = plot_data["Model"].str.replace("Classifier", "", regex=False)

        # Extract unique test model names
        test_auc_models = plot_data[plot_data["AUC_Type"] == "Test_AUC"]["Model"].unique()

        # All model names that appear in either Validation or Test data
        models_present = plot_data["Model"].unique().tolist()

        # Keep only those from desired_order that actually appear
        ordered_categories = [m for m in desired_order if m in models_present]

        # Extract ensemble model names and values as you already do:
        ensemble_models = [entry["Model"] for entry in correct_ensemble_test_data]
        ensemble_aucs = [entry["AUC"] for entry in correct_ensemble_test_data]
        ensemble_errors = [entry["Std"] for entry in correct_ensemble_test_data]
        print("ensemble_models",ensemble_models)
        # Build a lookup for quick indexing
        ensemble_lookup = {m: (auc, err) for m, auc, err in zip(ensemble_models, ensemble_aucs, ensemble_errors)}

        # Keep only categories that both (a) are in ordered_categories and (b) exist in ensemble results
        aligned_ensemble_x = [m for m in ordered_categories if m in ensemble_lookup]
        aligned_ensemble_y = [ensemble_lookup[m][0] for m in aligned_ensemble_x]
        aligned_ensemble_errors = [ensemble_lookup[m][1] for m in aligned_ensemble_x]

        
        # Initialize data list
        data_list = []

        # Add ensemble test AUROC scatter plot
        data_list.append(go.Scatter(
            x=aligned_ensemble_x,
            y=aligned_ensemble_y,
            mode='markers',
            name="Ensemble Test AUROC (CI 95%)",
            error_y=dict(
                type='data',
                array=aligned_ensemble_errors,
                visible=True
            ),
            marker=dict(color='green', size=10, symbol='circle')
        ))
        
        # Create figure
        fig = go.Figure(data=data_list)
        
        # Add Validation AUC
        fig.add_trace(go.Box(
            x=plot_data[plot_data["AUC_Type"] == "Validation_AUC"]["Model"],
            y=plot_data[plot_data["AUC_Type"] == "Validation_AUC"]["AUC"],
            name="Validation AUROC",
            legendgroup="Validation",  # Grouped for legend
            offsetgroup="Validation",  # Grouped for scaling
            line_color='blue',
            boxmean=True,
            opacity=0.6
        ))
        
        
        # Add Test AUC
        fig.add_trace(go.Box(
            x=plot_data[plot_data["AUC_Type"] == "Test_AUC"]["Model"],
            y=plot_data[plot_data["AUC_Type"] == "Test_AUC"]["AUC"],
            name="Test AUROC",
            legendgroup="Test",  # Grouped for legend
            offsetgroup="Test",  # Grouped for scaling
            line_color='orange',
            boxmean=True,
            opacity=0.6
        ))

        # ---- font size controls (change here) ----
        TITLE_SIZE   = 32
        AXIS_LABEL   = 20
        TICK_SIZE    = 18
        LEGEND_TITLE = 18
        LEGEND_TEXT  = 16
        # ------------------------------------------

        # If no custom title was provided
        if plot_title is None:
            plot_title = "Model Performance Evaluation"

        # One unified update_layout call
        fig.update_layout(
            font=dict(size=TICK_SIZE),  # global default font (affects ticks & legend text unless overridden)
            title=dict(
                text=plot_title,
                x=0.5,
                font=dict(size=TITLE_SIZE)
            ),
            xaxis=dict(
                categoryorder="array",
                categoryarray=desired_order,
                title=dict(text="Models", font=dict(size=AXIS_LABEL)),
                tickfont=dict(size=TICK_SIZE)
            ),
            yaxis=dict(
                range=[0, 1.01],
                dtick=0.05,
                tickvals=[i / 10 for i in range(11)],
                ticktext=[f"{i / 10:.1f}" for i in range(11)],
                showgrid=True,
                title=dict(text="AUROC", font=dict(size=AXIS_LABEL)),
                tickfont=dict(size=TICK_SIZE)
            ),
            legend=dict(
                title=dict(text="Evaluation", font=dict(size=LEGEND_TITLE)),
                font=dict(size=LEGEND_TEXT)
            ),
            boxmode="group"
        )

        # (Optional) enforce consistent label sizing inside traces (bar text, etc.)
        # fig.update_traces(textfont_size=TICK_SIZE)
        
        # Save the plot as an image if output_path is provided
        if output_path:
            png_file_path = output_path + '/AUROC_Summary_Plot.png'
            os.makedirs(output_path, exist_ok=True)
            fig.write_image(png_file_path, width=1200, height=800, scale=2)
        else:
            return fig


    def plot_label_distribution(self,
                            training_series=None,
                            testing_series=None,
                            validation_series=None,
                            output_path=None):
        """
        Generates a stacked bar plot for counts of values in training, testing, and validation pandas Series.

        :param output_path: (str) containing the path to save the plot and the filename of the figure.
        :param training_series: (pandas.Series) containing values for training.
        :param testing_series: (pandas.Series) containing values for testing.
        :param validation_series: (pandas.Series) containing values for validation.
        """

        if output_path is None:
            output_path = self.output_path + "/Label_distribution_" + str(self.RunID) + ".png"

        if (training_series is None) or (testing_series is None):
            self.error.error("Missing Arguments for plotting label distribution! Please provide training and testing labels!")
            raise ValueError("Missing Arguments for plotting label distribution! Please provide training and testing labels!")

        # Combine all series to get total labels
        series_list = [training_series, testing_series]
        if validation_series is not None and not validation_series.empty:
            series_list.append(validation_series)
        total_series = pd.concat(series_list, ignore_index=True)

        # Get all unique labels
        labels = sorted(total_series.unique())

        def get_counts(series, labels):
            counts = series.value_counts().reindex(labels, fill_value=0)
            return counts

        # Counting the values in each series
        total_counts = get_counts(total_series, labels)
        training_counts = get_counts(training_series, labels)
        testing_counts = get_counts(testing_series, labels)
        if validation_series is not None and not validation_series.empty:
            validation_counts = get_counts(validation_series, labels)
            datasets = ['Total', 'Training', 'Testing', 'Validation']
        else:
            validation_counts = None
            datasets = ['Total', 'Training', 'Testing']

        # Prepare data for DataFrame
        data = {'Data': datasets}
        for label in labels:
            label_str = str(label)
            data[label_str] = [
                total_counts[label],
                training_counts[label],
                testing_counts[label],
            ]
            if validation_counts is not None:
                data[label_str].append(validation_counts[label])

        df = pd.DataFrame(data)

        # Plotting
        ax = df.plot.bar(
            x='Data',
            stacked=True,
            figsize=(10, 6),
            xlabel='Data splits',
            ylabel='Sample Count',
            title='Count of Labels in Data splits'
        )

        # Adding counts on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='center')

        ax.set_xticklabels(df["Data"], rotation=45)
        ax.figure.tight_layout()

        # Save the plot
        ax.figure.savefig(output_path)
        plt.close(ax.figure)

    def plot_decision_regions_from_features(self, model, X=None, y=None, title=None, filename=""):
        """
        Plot decision regions of the two best features of the dataset and save the plot.
        
        :param model: Model to plot decision regions
        :param X: Dataframe with features
        :param y: Series with labels
        :param title: Title of the plot
        :param filename: Filename of the plot
        """

        if X is None:
            X = self.X

        if y is None:
            y = self.y

        if title is None:
            title = f"Optimized {type(model).__name__} on {self.RunID}"

        self.logger.info(f"Plot Decision Regions for {type(model).__name__}!")

        # **Optimized Figure Size**
        fig, ax = plt.subplots(figsize=(12, 9), dpi=300)

        # **Plot Decision Regions**
        plot_decision_regions(X.values, y.values, clf=model, ax=ax, legend=2)

        # **Enhance Formatting**
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel(X.columns[0], fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel(X.columns[1], fontsize=16, fontweight='bold', labelpad=10)

        # **Adjust Layout for Publication**
        plt.tight_layout()

        # **Save Figure in High Quality**
        if filename == "":
            filename = self.output_path + f'/Decision_region_{type(model).__name__}_{self.RunID}.png'
        else:
            if not filename.endswith(".png"):
                filename += ".png"

            if not filename.startswith(self.output_path):
                filename = self.output_path + '/' + filename

        plt.savefig(filename, bbox_inches='tight', dpi=300)

        # **Close to Prevent Memory Issues**
        plt.close()
        plt.clf()

    def _seq_rule_to_pattern(self, seq: Sequence[str]) -> Tuple[re.Pattern, str]:
        """
        Build a regex pattern that matches an exact token sequence (joined by "_" or "-").
        Captures the leading and trailing separators/anchors so we can preserve them in substitution.

        Example:
            seq = ("rlnu","norm")
            -> regex matches: (^|[_-])rlnu[_-]norm([_-]|$)
            -> replacement template: r'\\1{NEW}\\2'
        """
        inner = self.SEP.join(re.escape(tok) for tok in seq if tok)
        pat = re.compile(rf'(^|{self.SEP}){inner}({self.SEP}|$)', flags=re.IGNORECASE)
        return pat, r'\g<1>{NEW}\g<2>'

    def _single_token_pattern(self, token: str) -> re.Pattern:
        """
        Build a regex that matches a single token (case-insensitive), bounded by
        separators or string boundaries. Ensures we only replace whole tokens, not substrings.
        """
        return re.compile(rf'(^|{self.SEP}){re.escape(token)}({self.SEP}|$)', flags=re.IGNORECASE)

    def _clean_feature_names(self, 
        cols: List[str],
        *,
        multi_rules: Optional[List[Tuple[Union[Sequence[str], str], str]]] = None,
        single_rules: Optional[Dict[str, Optional[str]]] = None,
        remove_substrings: Optional[List[str]] = None,
        remove_regexes: Optional[List[Union[str, re.Pattern]]] = None,  # <-- accept both
        replace_map: Optional[Dict[str, str]] = None,
        collapse_underscores: bool = True,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Clean and normalize radiomics feature names.

        Parameters
        ----------
        cols : list[str]
            Original feature names.
        multi_rules : list[tuple[(Sequence[str] or str), str]], optional
            Rules to replace entire token sequences with a new string.
            Each rule is (("token1","token2",...), "NewName") or ("token1_token2", "NewName").
        single_rules : dict[str, Optional[str]], optional
            Rules to replace or drop single tokens (whole-token matches only).
            Value = None or "" → drop token, otherwise → replacement string.
        remove_substrings : list[str], optional
            Substrings to remove anywhere in the name (raw .replace).
        remove_regexes : list[str], optional
            Regex patterns to remove anywhere in the name.
        replace_map : dict[str,str], optional
            Fixed direct string replacements applied first.
        collapse_underscores : bool, default=True
            Collapse multiple underscores/hyphens, unify separators, strip edges.

        Returns
        -------
        cleaned : list[str]
            Cleaned feature names (deduplicated).
        mapping : dict[str,str]
            Mapping from original → cleaned name.

        Notes
        -----
        Cleaning order:
            1) Apply fixed `replace_map`.
            2) Apply `multi_rules` (token-sequence aware).
            3) Apply `single_rules` (single token aware).
            4) Remove raw substrings.
            5) Remove regex patterns.
            6) Normalize separators and trim.
            7) Replace underscores with spaces.
            8) Deduplicate names by appending "#2", "#3", ...
        """
        # Normalize input configs
        multi_rules = multi_rules or []
        single_rules = single_rules or {}
        remove_substrings = remove_substrings or []
        replace_map = replace_map or {}
        compiled_remove: List[re.Pattern] = []
        for rgx in (remove_regexes or []):                                   # <-- changed
            compiled_remove.append(re.compile(rgx) if isinstance(rgx, str) else rgx)

        # Precompile MULTI rules (token sequences)
        norm_multi: List[Tuple[re.Pattern, str, str]] = []
        for seq, new in multi_rules:
            if isinstance(seq, str):
                parts = tuple(p for p in re.split(self.SEP, seq) if p)
            else:
                parts = tuple(seq)
            pat, repl = self._seq_rule_to_pattern(parts)
            norm_multi.append((pat, repl, new))
        # Sort by length of pattern to prefer more specific sequences
        norm_multi.sort(key=lambda x: -len(x[0].pattern))

        # Precompile SINGLE token rules
        compiled_single: List[Tuple[re.Pattern, Optional[str], str]] = []
        # Sort tokens by length descending to prefer longer matches first
        for tok in sorted(single_rules.keys(), key=len, reverse=True):
            compiled_single.append((self._single_token_pattern(tok), single_rules[tok], tok))

        def apply_token_rules(name: str) -> str:
            """
            Apply all token cleaning rules to a single feature name.
            """
            t = name

            # 1) Fixed replacements
            for old, new in replace_map.items():
                t = t.replace(old, new)

            # 2) Multi-token rules
            for pat, repl, new in norm_multi:
                if new is None or new == "":
                    # Remove the matched sequence
                    t = pat.sub(r'\g<1>\g<2>', t)
                else:
                    # Replace with new name, preserving separators
                    t = pat.sub(repl.replace("{NEW}", new), t)

            # 3) Single-token rules
            for pat, new, tok in compiled_single:
                if new is None or new == "":
                    t = pat.sub(r'\g<1>\g<2>', t)  # drop token
                else:
                    t = pat.sub(rf'\g<1>{new}\g<2>', t)

            # 4) Remove raw substrings
            for sub in remove_substrings:
                t = t.replace(sub, '')

            # 5) Remove regex patterns
            for rgx in compiled_remove:
                t = rgx.sub('', t)

            # 6) Normalize separators
            if collapse_underscores:
                t = re.sub(r'_+', '_', t)
                t = re.sub(r'-+', '-', t)
                t = t.replace('-', '_')  # unify to underscore
            t = t.strip('_ ').strip()

            # 7) Replace underscores with spaces for readability
            t = t.replace('_', ' ')
            t = re.sub(r'\s+', ' ', t).strip()  # <-- collapse spaces
            return t

        # Apply cleaning to all features
        cleaned_raw = [apply_token_rules(c) for c in cols]

        # 8) Deduplicate cleaned names
        counts = defaultdict(int)
        cleaned = []
        for name in cleaned_raw:
            counts[name] += 1
            if counts[name] == 1:
                cleaned.append(name)
            else:
                cleaned.append(f"{name} #{counts[name]}")

        mapping = dict(zip(cols, cleaned))
        return cleaned, mapping

    def plot_shap(self, model, X, y, output_path=None, seed=1234, dataset=None, extractor="PyRadiomics"):
        """
        Plot explainable feature importance plots for getting the feature input into the performance
        :param model: trained model
        :param X: Dataset where the model has been fitted on
        :param y: Label of the data where the model has been fitted on
        :param output_path: Path to outfile
        """

        plt.rcParams.update({
                "font.size": 18,            # base font size
                "axes.titlesize": 20,       # title size
                "axes.labelsize": 18,       # x/y label size
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
                "legend.fontsize": 16,
                "legend.title_fontsize": 18,
            })

        print(3 * "#", "Perform SHAP Analysis", 3 * "#")
        shap_values = None
        if output_path is None:
            output_path = self.output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        if not hasattr(model, "classes_"):
            if "TabNet" in type(model).__name__:
                model.fit(X_train=X.values,
                                y_train=y.values,
                                eval_set=[(X.values, y.values)],
                                eval_name=['train'],
                                eval_metric=['auc'],
                                max_epochs=100,
                                patience=10,
                                batch_size=32,
                                virtual_batch_size=32,
                                num_workers=1,
                                weights=1,
                                drop_last=False)
            else:
                model = model.fit(X, y)

        if "Ensemble" in  type(model).__name__:
            if dataset is None:
                title = "Ensemble " + type(model.clfs[0]).__name__
            else:
                title = f"{extractor} SHAP Values from " + dataset.replace("_"," ") + " \nEnsemble " + type(model.clfs[0]).__name__
            explainer_name = "Ensemble_" + type(model.clfs[0]).__name__
        else:
            if dataset is None:
                title = type(model).__name__
            else:
                title =  f"{extractor} SHAP Values from " + dataset.replace("_"," ") + " \n" + type(model).__name__
            explainer_name = type(model).__name__
        print(title,  type(model).__name__)
        # Check if SHAP has already been processed
        if os.path.exists(output_path + "/" + explainer_name + "_shap_values.pkl"):
            print("Found SHAP Values. Loading ...")
            with open(output_path + "/" + explainer_name + "_shap_values.pkl", "rb") as val_f:
                shap_values = pickle.load(val_f)
            # shap_values = np.load(output_path + "/" + explainer_name + "_shap_values.npy")
        # Explainer
        if os.path.exists(output_path + "/" + explainer_name + "_explainer.pkl"):
            print("Found SHAP Explainer. Loading ...")
            with open(output_path + "/" + explainer_name + "_explainer.pkl", "rb") as f:
                explainer = shap.Explainer.load(f)
        else:
            if shap_values is None:
                # compute SHAP values
                try:
                    if "TabNet" in type(model).__name__:
                        explainer = shap.Explainer(model=model.predict, masker=X.values, seed=seed)
                        
                        if type(explainer).__name__ == "PermutationExplainer":
                            max_num = 2 * int(X.shape[1]) + 1 
                            shap_values = explainer(X.values, max_evals = max_num)
                        else:
                            shap_values = explainer(X.values)
                            
                    elif "Ensemble" in  type(model).__name__:
                        
                        if len(model.clfs) > 0:
                            if "TabNet" in type(model.clfs[1]).__name__:
                                
                                explainer = shap.Explainer(model=model.predict, masker=X.values, seed=seed)
                                
                                if type(explainer).__name__ == "PermutationExplainer":
                                    max_num = 2 * int(X.shape[1]) + 1 
                                    shap_values = explainer(X.values, max_evals = max_num)
                                else:
                                    shap_values = explainer(X.values)
                                    
                            elif "Ensemble" in  type(model.clfs[1]).__name__:
                                if "TabNet" in type(model.clfs[1].clfs[1]).__name__:
                                    explainer = shap.Explainer(model=model.predict, masker=X.values, seed=seed)
                                
                                    if type(explainer).__name__ == "PermutationExplainer":
                                        max_num = 2 * int(X.shape[1]) + 1 
                                        shap_values = explainer(X.values, max_evals = max_num)
                                    else:
                                        shap_values = explainer(X.values)
                                else:
                                    explainer = shap.Explainer(model=model.predict, masker=X, seed=seed)
                        
                                    if type(explainer).__name__ == "PermutationExplainer":
                                        max_num = 2 * int(X.shape[1]) + 1 
                                        shap_values = explainer(X, max_evals = max_num)
                                    else:
                                        shap_values = explainer(X)

                            else:
                                explainer = shap.Explainer(model=model.predict, masker=X, seed=seed)
                        
                                if type(explainer).__name__ == "PermutationExplainer":
                                    max_num = 2 * int(X.shape[1]) + 1 
                                    shap_values = explainer(X, max_evals = max_num)
                                else:
                                    shap_values = explainer(X)
                        else:
                            print("No Models in Ensembling. Can not generate SHAP plots!")
                            self.error.warning("No Models in Ensembling. Can not generate SHAP plots!")
                            return
                    else:
                        explainer = shap.Explainer(model=model.predict, masker=X, seed=seed)
                        
                        if type(explainer).__name__ == "PermutationExplainer":
                            max_num = 2 * int(X.shape[1]) + 1 
                            shap_values = explainer(X, max_evals = max_num)
                        else:
                            shap_values = explainer(X)

                except ValueError:
                    print("Need to adapt the SHAP explainer!")
                    self.logger.info("Need to adapt the SHAP explainer as the max_evals is too low!")

                    # Set to minimum number required
                    max_evals_num = int(2 * len(X.columns) + 1)
                    explainer = shap.Explainer(model=model.predict, masker=X, seed=seed, max_evals=max_evals_num)
                    if "TabNet" in type(model).__name__:
                        shap_values = explainer(X.values)
                    else:
                        shap_values = explainer(X)
                
                try:
                    with open(output_path + "/" + explainer_name + "_explainer.pkl", "wb") as f:
                        explainer.save(f)
                except Exception as ex:
                    print(f"Failed saving SHAP explainer. {ex}")
                    self.error.warning(f"Failed saving SHAP explainer. {ex}")

                try:
                    with open(os.path.join(output_path, f"{explainer_name}_shap_values.pkl"), "wb") as f:
                        pickle.dump(shap_values, f)
                except Exception as ex:
                    print(f"Failed saving SHAP values. {ex}")
                    self.error.warning(f"Failed saving SHAP values. {ex}")
        
        for feature in X.copy().columns.to_list():
            if "peritumoral" in feature:
                new_feature = feature.replace("peritumoral","margin")
                X = X.rename(columns={feature: new_feature})
                feature = new_feature
            if "intensity-based_statistics" in feature:
                print(feature)
                new_feature = feature.replace("intensity-based_statistics","Firstorder")
                X = X.rename(columns={feature: new_feature})
                feature = new_feature
            if "diagnostics" in feature:
                new_feature = feature.replace("diagnostics","Diagnostic")
                X = X.rename(columns={feature: new_feature})
                

        # === Clean feature names for plotting ===
        if extractor == "MIRP":
            multi = self.MIRP_MULTI
            single = self.MIRP_SINGLE
        else:  # assume PyRadiomics
            multi = self.PRAD_MULTI
            single = self.PRAD_SINGLE

        cons = ConsensusFeatureFormatter(df=X.copy(),
                                feature_cols=None, 
                                extractor=extractor,
                                output_path=self.output_path,
                                generate_feature_profile_plot=False,
                                run_id=self.RunID
                            )
        report, X_clean = cons.run(
                            title=self.RunID,
                            return_dataframe=True,
                            rename_mode="rename"           # or "multiindex" or "add_columns"
                        )
        
        cleaned = X_clean.columns.to_list()

        real_clean = []
        for clean_feat in cleaned:
            if clean_feat.startswith("_"):
                real_clean.append(clean_feat[1:])
            else:
                real_clean.append(clean_feat)

        cleaned = real_clean
        cleaned, mapping = self._clean_feature_names(
            cols=X.copy().columns.to_list(),
            multi_rules=multi,
            single_rules=single,
           # optional extras:
            remove_substrings=[],                 # keep empty; token rules handle most
            remove_regexes=[r'w\d+(?:\.\d+)?'],   # if you still want to drop w25.0/w50.0 anywhere
            replace_map={},
            collapse_underscores=True,
        )
        
        # return X.copy().columns.to_list()
        # Copy X just for labeling / clustering; don’t touch model training data
        #X_clean = X.copy()
        #X_clean.columns = cleaned

        # Ensure SHAP object uses cleaned names so all plots display them
        try:
            # shap.Explanation usually allows this attribute to be set
            if shap_values is not None:
                shap_values.feature_names = cleaned
            else:
                shap_values = explainer(X)

        except Exception as ex:
            print("shap_values.feature_names Failed!", ex)
            try:
                explainer = shap.Explainer(model=model.predict, masker=X, seed=seed)
                
                if type(explainer).__name__ == "PermutationExplainer":
                    max_num = 2 * int(X.shape[1]) + 1 
                    shap_values = explainer(X, max_evals = max_num)
                else:
                    shap_values = explainer(X)
                shap_values.feature_names = cleaned
            except Exception as ex1:

                # Fallback: some older SHAP versions might not support assignment
                # In that case, pass 'feature_names' explicitly where supported
                print(f"Something wrent wrong when trying to polish the feature names! {ex1}")
                pass

        # SHAP Plotting
        if not os.path.exists(output_path + '/Waterfall_' + type(model).__name__ + "_" + self.RunID + '.png'):
            # Visualization with Waterfall plots
            # fig = plt.figure(figsize=(250, 220))
            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            plt.title(title)
            # plt.tight_layout()
            
            try:
                plt.savefig(output_path + '/Waterfall_' + type(model).__name__ + "_" + self.RunID + '.png', bbox_inches='tight')
            except:
                # if figure size is too big
                plt.savefig(output_path + '/Waterfall_' + type(model).__name__ + "_" + self.RunID + '.png')
            
            plt.close()
            plt.clf()

        if type(explainer).__name__ != "PermutationExplainer":
            if not os.path.exists(output_path + '/Force_' + type(model).__name__ + "_" + self.RunID + '.png'):
                # Visualization with force plots
                # fig = plt.figure(figsize=(150, 120))
                fig = plt.figure(figsize=(8, 6))
                shap.plots.force(explainer.expected_value, shap_values[0, :], show=False, matplotlib=True)
                plt.title(title)
                # plt.tight_layout()
                plt.savefig(output_path + '/Force_' + type(model).__name__ + "_" + self.RunID + '.png', bbox_inches='tight')
                plt.close()
                plt.clf()

        if not os.path.exists(output_path + '/Beeswarm_' + type(model).__name__ + "_" + self.RunID + '.png'):
            # summarize the effects of all the features
            # fig = plt.figure(figsize=(150, 120))
            fig = plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, show=False, max_display=10)

            ax = plt.gca()
            ax.set_title(title, fontsize=22)

            ax.set_xlabel("SHAP Value (impact on model output)", fontsize=20)
            ax.set_ylabel("Features", fontsize=20)

            ax.tick_params(axis="y", labelsize=18)
            ax.tick_params(axis="x", labelsize=18)

            for text in ax.texts:   # the "+0.27" annotations
                text.set_fontsize(16)
            
            # --- Fix colorbar axis ---
            fig = plt.gcf()
            if len(fig.axes) > 1:
                cbar_ax = fig.axes[-1]  # SHAP colorbar axis
                cbar_ax.set_ylabel("Feature value", fontsize=20, labelpad=15)
                cbar_ax.tick_params(labelsize=18)
                for t in cbar_ax.get_yticklabels():
                    t.set_fontsize(18)

            # plt.tight_layout()
            plt.savefig(output_path + '/Beeswarm_' + type(model).__name__ + "_" + self.RunID + '.png', bbox_inches='tight')
            plt.close()
            plt.clf()

        if not os.path.exists(output_path + '/Bar_' + type(model).__name__ + "_" + self.RunID + '.png'):
            # mean absolute value of the SHAP values
            # fig = plt.figure(figsize=(150, 120))
            fig = plt.figure(figsize=(8, 6))
            shap.plots.bar(shap_values, max_display=10, show=False)
            # plt.title(title)
            ax = plt.gca()
            ax.set_title(title, fontsize=22)
            ax.set_xlabel("mean(|SHAP value|)", fontsize=20)
            ax.set_ylabel("Features", fontsize=20)
            ax.tick_params(axis="y", labelsize=18)
            ax.tick_params(axis="x", labelsize=18)
            for text in ax.texts:   # the "+0.27" annotations
                text.set_fontsize(16)
            # plt.tight_layout()
            plt.savefig(output_path + '/Bar_' + type(model).__name__ + "_" + self.RunID + '.png', bbox_inches='tight')
            plt.close()
            plt.clf()

        if not os.path.exists(output_path + '/Clustered_Bar_' + type(model).__name__ + "_" + self.RunID + '.png'):
            # by default this trains (X.shape[1] choose 2) 2-feature XGBoost models
            # clustering = shap.utils.hclust(X, y)
            # Clustering for the clustered bar
            clustering = shap.utils.hclust(X_clean, y)

            # fig = plt.figure(figsize=(150, 120))
            fig = plt.figure(figsize=(8, 6))
            shap.plots.bar(shap_values, clustering=clustering, max_display=10, clustering_cutoff=0.9, show=False)
            plt.title(title)
            # plt.tight_layout()
            plt.savefig(output_path + '/Clustered_Bar_' + type(model).__name__ + "_" + self.RunID + '.png',
                        bbox_inches='tight')
            plt.close()
            plt.clf()

        if not os.path.exists(output_path + '/Explaination_matrix_' + type(model).__name__ + "_" + self.RunID + '.png'):
            # hierarchical clustering by the explanation similarity
            # fig = plt.figure(figsize=(150, 120))
            fig = plt.figure(figsize=(8, 6))
            shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1), max_display=10, show=False)
            plt.title(title)
            # plt.tight_layout()
            plt.savefig(output_path + '/Explaination_matrix_' + type(model).__name__ + "_" + self.RunID + '.png',
                        bbox_inches='tight')
            plt.close()
            plt.clf()

    @staticmethod
    def plot_probability_distribution(df, label_column, proba_column, save_path=None, bins=10, colors=None, title="Distribution of Prediction Probabilities", model_name=""):
        """
        Plots the distribution of prediction probabilities for multiple labels or regression predictions.
        
        Parameters:
        - df: pandas DataFrame containing the data
        - label_column: str, name of the column with true labels (or categories for classification)
        - proba_column: str, name of the column with predicted probabilities (or regression values)
        - bins: int, number of bins for the histogram
        - colors: dict, optional, mapping of labels to colors for visualization
        - title: str, title of the plot
        - save_path: str, optional, file path to save the plot (e.g., 'plot.png', 'plot.pdf')
        
        Supports both classification (multiple labels) and regression cases.
        """

        unique_labels = df[label_column].unique()
        
        # Default color mapping if not provided
        if colors is None:
            color_palette = plt.cm.get_cmap("tab10", len(unique_labels))
            colors = {label: color_palette(i) for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(10, 6))

        for label in unique_labels:
            subset = df[df[label_column] == label][proba_column]
            plt.hist(subset, bins=bins, alpha=0.7, color=colors[label], label=f"Label = {label}")

        if model_name != "":
            title = title + " for " + model_name
            if save_path.endswith("/"):
                save_path = save_path + "Distribution_of_Prediction_Probabilities_" + model_name + ".png"
            else:
                if os.path.isdir(save_path):
                    save_path = save_path + "/Distribution_of_Prediction_Probabilities" + model_name + ".png"
        else:
            if save_path.endswith("/"):
                save_path = save_path + "Distribution_of_Prediction_Probabilities.png"
            else:
                if os.path.isdir(save_path):
                    save_path = save_path + "/Distribution_of_Prediction_Probabilities.png"

        plt.title(title)
        plt.xlabel('Prediction Probability' if df[proba_column].max() <= 1 else 'Predicted Value')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            print("Plot not saved. Use 'save_path' argument to save the plot.")
            raise ValueError("Plot not saved. Use 'save_path' argument to save the plot.")
        
        plt.close()


    def plot_overfitting_plot(self, train_auc_cv, val_auc_cv, test_auc_cv, model_name, output_path):
        """
        Plot performance of model per fold in train/test/val set
        """
        
        folds = range(1, len(train_auc_cv) + 1)
        try:
            plt.plot(folds, train_auc_cv, 'o-', color='green', label='train')
            plt.plot(folds, val_auc_cv, 'o-', color='orange', label='val')
            plt.plot(folds, test_auc_cv, 'o-', color='red', label='test')
            plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0)
            plt.grid()
            plt.title("Performance per fold of the " + model_name)
            plt.xlabel('Fold')
            plt.ylabel('AUROC')
            plt.tight_layout()
            plt.savefig(output_path + '/Overfitting_plot_' + model_name + "_" + self.RunID + '.png', bbox_inches='tight')
            plt.close()
            plt.clf()
        except:
            print("Overfitting Plot generation failed.")
            self.error.warning("Overfitting Plot generation failed.")

    def plot_cv_AUC(self, y, y_pred, fold, ax):
        """
        Plot the cross validation AUC of the model and save it into the output_path folder
        :param model: model to make predictions
        :param y: ground truth label
        :param y_pred: predictions which have been made
        :param fold: in as number of fold from cv
        :param ax: plot object
        """

        # PerformancePlotter
        viz = RocCurveDisplay.from_predictions(y_true=y,
                                               y_pred=y_pred,
                                               name="ROC fold {}".format(fold),
                                               alpha=0.3,
                                               lw=4,
                                               ax=ax,
                                               pos_label=1)
        return viz

        # if self.output_path != "":
        # if not os.path.exists(self.output_path + "/AUC"):
        # os.mkdir(self.output_path + "/AUC")

        # plt.savefig(self.output_path + "/AUC/" + str(type(model).__name__) + "_" + self.RunID + "_" + str(
        # round(mean_auc, 2)) + "_cv_AUROC.png")
        # Clear the figure to free memory if plotting multiple figures in a loop or a script
        # else:
        # plt.savefig(self.output_path + str(type(model).__name__) + "_" + self.RunID + "_" + str(
        # round(mean_auc, 2)) + "_cv_AUROC.png")

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

            plot.xlabel("Iteration")
            plot.ylabel("Best test mean AUROC")
            plot.title(title)
            plot.tight_layout()
            plot.savefig(self.out_dir + str(type(self.model).__name__) + "_" + title + '.png', bbox_inches='tight')

        smallest_best_iteration = df_i.drop_duplicates(subset="iter")["iter"].loc[
            df_i.drop_duplicates(subset="iter")["AUC"] == df_i.drop_duplicates(subset="iter")["AUC"].max()].values[
            0]
        best_auc = df_i.drop_duplicates(subset="iter")["AUC"].loc[
            df_i.drop_duplicates(subset="iter")["AUC"] == df_i.drop_duplicates(subset="iter")["AUC"].max()].values[
            0]

        # if self.verbose == 1:
        print("Max AUC with", smallest_best_iteration, "iteration and", round(best_auc, 3), "AUROC")

        # get the best parameters
        best_parameters = df_i.loc[df_i["iter"] == smallest_best_iteration]["params"].values[0]

        # if df_i.shape[1] > 1:
        #    fig = plot.get_figure()
        #    fig.savefig(self.out_file)
        # plt.close()

        return smallest_best_iteration, best_auc, best_parameters
    
    @staticmethod
    def plot_val_auc_distribution(val_auc_dict, save_path=None):
        """
        This function plots the distribution of Val_AUC values for each model
        and returns the name of the model with the best mean Val_AUC. Optionally, it saves the plot to a file.
        
        Parameters:
        val_auc_dict (dict): A dictionary where keys are model names and values are lists of Val_AUC values.
        save_path (str, optional): The file path where the plot should be saved. If None, the plot will not be saved.
        
        Returns:
        str: The model name with the best mean Val_AUC.
        """
        # Convert the dictionary into a list of tuples for plotting
        model_names = list(val_auc_dict.keys())
        auc_values = [val_auc_dict[model] for model in model_names]
        
        # Create a DataFrame for plotting
        auc_df = pd.DataFrame({
            'Model': [model for model in model_names for _ in val_auc_dict[model]],
            'Val_AUC': [auc for auc_list in auc_values for auc in auc_list]
        })

        # Save the plot if a save_path is provided
        if save_path:
             # Create a boxplot for the distribution of Val_AUC for each model
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Model', y='Val_AUC', data=auc_df, palette="Set2")

            # Customize plot labels and title
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Val AUC', fontsize=14)
            plt.title('Distribution of Val AUC for Each Model', fontsize=16)
            
            # Set the y-axis to always scale from 0 to 1 and set ticks every 0.1 step
            plt.ylim(0, 1.01)
            plt.yticks([i * 0.1 for i in range(11)])  # Set ticks from 0 to 1 with step size of 0.1
            
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=90)
            
            # Show the plot
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')  # Save the plot to the specified file path
        
            plt.close()
            plt.clf()

        # Calculate the mean Val_AUC for each model and return the model with the highest mean
        mean_auc = {model: sum(val_auc_dict[model]) / len(val_auc_dict[model]) for model in val_auc_dict}
        
        # Find the model with the best (highest) mean Val_AUC
        best_model = max(mean_auc, key=mean_auc.get)
        # print("Best Model", best_model, mean_auc)
        
        return best_model, mean_auc[best_model]
    
    @staticmethod
    def extract_model_and_fold(model_string):
        """
        This function extracts the model name and the fold number from strings like
        'optimized_RandomForestClassifier_0_on_fold_val'.
        
        Parameters:
        model_string (str): The string representing the model (e.g., 'optimized_RandomForestClassifier_0_on_fold_val')
        
        Returns:
        tuple: (model_name, fold_number)
        """
        # Extract the model name (after 'optimized_' and before '_<fold>_on_fold_val')
        model_name_match = re.search(r'optimized_([A-Za-z]+)', model_string)
        fold_match = re.search(r'_(\d+)_on_fold_val', model_string)
        
        if model_name_match and fold_match:
            model_name = model_name_match.group(1)
            fold_number = int(fold_match.group(1))
            return model_name, fold_number
        else:
            return None, None  # In case the string doesn't match the expected pattern

    @staticmethod
    def extract_ensemble_model_name(ensemble_string):
        """
        This function extracts the base model name from an ensemble model string like
        'SoftEnsemble_RandomForestClassifier_cv_optimized_RandomForestClassifier'.
        
        Parameters:
        ensemble_string (str): The string representing the ensemble model (e.g., 'SoftEnsemble_RandomForestClassifier_cv_optimized_RandomForestClassifier')
        
        Returns:
        str: The base model name (e.g., 'RandomForestClassifier')
        """
        # Extract the base model name after 'SoftEnsemble_' and before '_cv_'
        model_name_match = re.search(r'SoftEnsemble_([A-Za-z]+)', ensemble_string)
        
        if model_name_match:
            model_name = model_name_match.group(1)
            return model_name
        else:
            return None  # If no match found


    @staticmethod
    def get_AUC_values_per_models(df:pd.DataFrame, number_of_folds:int=5):
        """
        This function processes a DataFrame containing AUC values for different models (both fold models and ensemble models),
        and organizes these values into dictionaries for later analysis.

        The function extracts the AUC values for:
        - **Fold models**: For each individual fold model (e.g., `RandomForestClassifier_0_on_fold_val`), it stores the `Val_AUC` and `Test_AUC` values in separate dictionaries.
        - **Ensemble models**: For ensemble models (e.g., `SoftEnsemble_RandomForestClassifier`), it stores only the `Test_AUC` values.

        The function returns three dictionaries:
        - **val_auc_fold_models**: Maps the model names to a list of `Val_AUC` values across all folds.
        - **test_auc_fold_models**: Maps the model names to a list of `Test_AUC` values across all folds.
        - **test_auc_ensemble_models**: Maps the ensemble model names to a list of `Test_AUC` values.

        Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the columns 'Models', 'Val_AUC', and 'Test_AUC'.
        number_of_folds (int): The number of folds each model should be trained
        
        Returns:
        tuple: A tuple containing three dictionaries:
            - `val_auc_fold_models`: Dictionary of fold models with `Val_AUC` values.
            - `test_auc_fold_models`: Dictionary of fold models with `Test_AUC` values.
            - `test_auc_ensemble_models`: Dictionary of ensemble models with `Test_AUC` values.
        """

        def check_numeric_format_(df:pd.DataFrame, col:str):
            """
            Convert data if wrong numeric format is present.
            """
            wrong_format = df[col].astype(str).str.contains(',', na=False).any()
            if wrong_format:
                # try to check if the values are strings with , numeric format
                col_str = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(col_str, errors='coerce')

            return df

        # Dictionaries to store AUC values for fold models and ensemble models
        val_auc_fold_models = {}
        test_auc_fold_models = {}
        test_auc_ensemble_models = {}

        # Check format
        df = check_numeric_format_(df=df , col="Val_AUC")
        df = check_numeric_format_(df=df , col="Test_AUC")

        # Iterate over each model in the 'Models' column
        for model in df["Models"]:
            # Extract model and fold information using the helper functions
            fold_model, fold = PerformancePlotter.extract_model_and_fold(model)
            ens_model = PerformancePlotter.extract_ensemble_model_name(model)

            # If it's a fold model (not an ensemble), process its AUC values
            if (fold_model is not None) and (fold is not None):
                # Add Val_AUC values to the fold model's entry
                if fold_model in val_auc_fold_models:
                    val_auc_fold_models[fold_model] += df.loc[df["Models"] == model, "Val_AUC"].to_list()
                else:
                    val_auc_fold_models[fold_model] = df.loc[df["Models"] == model, "Val_AUC"].to_list()

                # Add Test_AUC values to the fold model's entry
                if fold_model in test_auc_fold_models:
                    test_auc_fold_models[fold_model] += df.loc[df["Models"] == model, "Test_AUC"].to_list()
                else:
                    test_auc_fold_models[fold_model] = df.loc[df["Models"] == model, "Test_AUC"].to_list()

            # If it's an ensemble model, process its AUC value
            elif (ens_model is not None):
                # Store the Test_AUC for the ensemble model
                test_auc_ensemble_models[ens_model] = df.loc[df["Models"] == model, "Test_AUC"].to_list()

            # If the model does not fit either category, skip it
            #else:
            #    print(f"Skipping {model} (not a valid fold or ensemble model)")
            
        # check if all folds are present 
        for cv_model in val_auc_fold_models:
            if len(val_auc_fold_models[cv_model]) != number_of_folds:
                if len(val_auc_fold_models[cv_model]) > number_of_folds:
                    print(f"Too many fold results for {cv_model}! Expected {number_of_folds} but got {len(val_auc_fold_models[cv_model])}! Check prediction_summary.csv and log files!")
                    
                if len(val_auc_fold_models[cv_model]) < number_of_folds:
                    print(f"Missing fold results for {cv_model}! Expected {number_of_folds} but got {len(val_auc_fold_models[cv_model])}! Check prediction_summary.csv and log files!")
                

        # Return the dictionaries containing the AUC values
        return val_auc_fold_models, test_auc_fold_models, test_auc_ensemble_models

    
    def check_numeric_format(self, df:pd.DataFrame, col:str):
        """
        Convert data if wrong numeric format is present.
        """
        wrong_format = df[col].astype(str).str.contains(',', na=False).any()
        if wrong_format:
            # try to check if the values are strings with , numeric format
            col_str = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(col_str, errors='coerce')

        return df

    @staticmethod
    def check_numeric_format_(df:pd.DataFrame, col:str):
        """
        Convert data if wrong numeric format is present.
        """
        wrong_format = df[col].astype(str).str.contains(',', na=False).any()
        if wrong_format:
            # try to check if the values are strings with , numeric format
            col_str = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(col_str, errors='coerce')

        return df

    @staticmethod
    def plot_model_auc_ci_vertical(
                                df, 
                                filename=None,
                                model_col='Models', 
                                mean_col='Bootstrap_AUC', 
                                lower_col='Bootstrap_AUC_lower', 
                                upper_col='Bootstrap_AUC_upper',
                                fig_size=(8, 6)):
        """
        Plots each model's AUC confidence interval (CI) as vertical bars, 
        with a red dot for the mean AUC.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the following columns:
            - model_col: Name of the model
            - mean_col: Mean bootstrap AUC
            - lower_col: Lower 95% CI bound of bootstrap AUC
            - upper_col: Upper 95% CI bound of bootstrap AUC
        
        model_col : str, default='Models'
            Column name for the model identifiers.
        mean_col : str, default='Bootstrap_AUC'
            Column name for the mean bootstrap AUC.
        lower_col : str, default='Bootstrap_AUC_lower'
            Column name for the lower bound of the bootstrap AUC 95% CI.
        upper_col : str, default='Bootstrap_AUC_upper'
            Column name for the upper bound of the bootstrap AUC 95% CI.
        fig_size : tuple, default=(8, 6)
            Size of the resulting figure.
        filename : str, optional
            If provided, saves the figure to the given filename.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes (if no filename is provided).
        """

        def check_numeric_format_(df:pd.DataFrame, col:str):
            """
            Convert data if wrong numeric format is present.
            """
            wrong_format = df[col].astype(str).str.contains(',', na=False).any()
            if wrong_format:
                # try to check if the values are strings with , numeric format
                col_str = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(col_str, errors='coerce')

            return df

        # Sort dataframe by mean AUC for better visualization
        df_plot = df.copy().sort_values(by=mean_col)
        models = df_plot[model_col].values
        x_positions = np.arange(len(models))  # x-axis positions
        
        # Check format
        df_plot = check_numeric_format_(df=df_plot , col=mean_col)
        df_plot = check_numeric_format_(df=df_plot , col=lower_col)
        df_plot = check_numeric_format_(df=df_plot , col=upper_col)

        # Extract values
        mean_aucs = df_plot[mean_col].values
        lower_aucs = df_plot[lower_col].values
        upper_aucs = df_plot[upper_col].values

        fig, ax = plt.subplots(figsize=fig_size)

        # Plot vertical error bars for each model
        ax.errorbar(x_positions, mean_aucs, 
                    yerr=[mean_aucs - lower_aucs, upper_aucs - mean_aucs], 
                    fmt='o', color='red', markersize=8, capsize=5, label="Mean AUC with 95% CI")

        # Set x ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Labels and title
        ax.set_ylabel('Test AUROC')
        ax.set_xlabel('Models')
        ax.set_title('Model Performance AUROC (CI 95%)')

        # Set y-axis to always range from 0 to 1.0
        ax.set_ylim(0, 1.01)
        ax.set_yticks(np.arange(0, 1.1, 0.1))  # Tick values from 0 to 1.0, inclusive

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')  # Saves the figure
            print(f"Plot saved to {filename}")
            plt.close()
            plt.clf()
        else:
            return fig, ax

    @staticmethod
    def plot_model_auc_ci_with_mean(
                                df, 
                                filename=None,
                                model_col='Models', 
                                mean_col='Bootstrap_AUC', 
                                lower_col='Bootstrap_AUC_lower', 
                                upper_col='Bootstrap_AUC_upper',
                                fig_size=(8, 6)):
        """
        Plots each model's AUC confidence interval (CI) with a red dot for the mean,
        and fixes the x-axis range from 0 to 1.0 with steps of 0.1.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the following columns:
            - model_col: Name of the model
            - mean_col: Mean bootstrap AUC
            - lower_col: Lower 95% CI bound of bootstrap AUC
            - upper_col: Upper 95% CI bound of bootstrap AUC
        
        model_col : str, default='Models'
            Column name for the model identifiers.
        mean_col : str, default='Bootstrap_AUC'
            Column name for the mean bootstrap AUC.
        lower_col : str, default='Bootstrap_AUC_lower'
            Column name for the lower bound of the bootstrap AUC 95% CI.
        upper_col : str, default='Bootstrap_AUC_upper'
            Column name for the upper bound of the bootstrap AUC 95% CI.
        fig_size : tuple, default=(8, 6)
            Size of the resulting figure.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """

        # Sort dataframe by mean if you like a sorted order
        df_plot = df.copy().sort_values(by=mean_col)
        models = df_plot[model_col].values
        y_positions = np.arange(len(models))
        
        # Extract values
        mean_aucs = df_plot[mean_col].values
        lower_aucs = df_plot[lower_col].values
        upper_aucs = df_plot[upper_col].values

        fig, ax = plt.subplots(figsize=fig_size)

        # For each model, plot a horizontal line with bars to represent the CI
        for i, (mean_val, low_val, up_val) in enumerate(zip(mean_aucs, lower_aucs, upper_aucs)):
            # Plot the line for CI
            ax.plot([low_val, up_val], [y_positions[i], y_positions[i]], color='blue', lw=2)
            
            # Plot the bars at the ends of the CI
            ax.plot([low_val, low_val], [y_positions[i] - 0.1, y_positions[i] + 0.1], color='blue', lw=2)  # Left bar
            ax.plot([up_val, up_val], [y_positions[i] - 0.1, y_positions[i] + 0.1], color='blue', lw=2)    # Right bar

            # Plot the mean as a red dot
            ax.plot(mean_val, y_positions[i], 'o', color='red', markersize=8)

        # Set y ticks and labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(models)
        ax.set_xlabel('Test AUROC')
        ax.set_ylabel('Models')
        ax.set_title('Model Performance AUROC (CI 95%)')

        # Set the x-axis to always range from 0 to 1.0
        ax.set_xlim(0, 1.0)

        # Set x-axis ticks in steps of 0.1
        ax.set_xticks(np.arange(0, 1.1, 0.1))  # Tick values from 0 to 1.0, inclusive

        plt.tight_layout()
        
        if not filename is None:
            # Save the plot to a file
            plt.savefig(filename, bbox_inches='tight')  # Saves the figure to the specified file
            plt.close()
            plt.clf()
        else:
            return fig, ax
