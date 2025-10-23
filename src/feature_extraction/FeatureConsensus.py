# -*- coding: utf-8 -*-
"""
ConsensusFeatureFormatter
-------------------------
Unifies feature parsing for both MIRP and PyRadiomics and emits a standardized,
IBSI/MIRP-aligned "consensus" name per feature, prefixed with your preferred
class abbreviations (GLCM, GLRLM, GLSZM, NGLDM/GLDM, NGTDM, IS, IH, IVH, Morphological, ...).

Key outputs per feature row:
  - Feature            (token within class, if detectable)
  - Feature_Class      (long form; same labels you used before)
  - Consensus          (e.g., 'GLCM_info_corr2', 'IS_rms')
  - Abbr               (e.g., 'GLCM', 'IS')
  - Family             (MIRP family code: 'cm','rl','sz','dm','nt','stat','ih','ivh','morph',...)
  - MIRP_Token         (e.g., 'cm_info_corr2', 'stat_rms')
  - IBSI_ID, Non_IBSI  (optional metadata; extend MIRP_META if you need IDs)
  - Image_Kernel       (best-effort filter/kernel tag or 'Original_Image')
  - Name               (original column name)

Also includes:
  - `generate_feature_profile(...)` (same stacked bar plot you used)
  - `export_name_map(...)` → {original_name: consensus} (handy for SHAP labels)
"""

from __future__ import annotations
import re
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from typing import Dict, Iterable, Optional, Tuple, Union, List

# ---------- Logging hook (kept) ----------
try:
    from rptk.src.config.Log_generator_config import LogGenerator
except Exception:
    # Minimal fallback if your LogGenerator isn't available in some envs
    import logging
    class LogGenerator:
        def __init__(self, log_file_name, logger_topic):
            self.log_file_name = log_file_name
            self.logger_topic = logger_topic
        def generate_log(self):
            logger = logging.getLogger(self.logger_topic)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                logger.addHandler(h)
            return logger

        
        # Which MIRP family prefix to use for each PyRadiomics class
_PR_CLASS_TO_MIRP_PREFIX = {
    "glcm": "cm",
    "glrlm": "rlm",      # MIRP uses "rl_*" in IBSI ids table
    "glszm": "szm",
    "gldm": "dzm",      # IBSI renamed GLDM → NGLDM
    "ngtdm": "nt",
    "firstorder": "stat",
    "shape": "morph",
}

# Class-aware token mapping for ambiguous names
PR_CLASS_TOKEN_TO_MIRP = {
    "glcm": {
        "Contrast": "cm_contrast",
        "Autocorrelation": "cm_auto_corr",
        "SumEntropy": "cm_sum_entr",
        "SumAverage": "cm_sum_avg",
        "Imc1": "cm_info_corr1",
        "Imc2": "cm_info_corr2",
        "Idm": "cm_inv_diff_mom",
        "Idmn": "cm_inv_diff_mom_norm",
        "Id": "cm_inv_diff",
        "Idn": "cm_inv_diff_norm",
        "InverseVariance": "cm_inv_var",
        "JointEntropy": "cm_joint_entr",
        "JointEnergy": "cm_energy",
        "Correlation": "cm_corr",
        "ClusterShade": "cm_clust_shade",
        "ClusterProminence": "cm_clust_prom",
        "ClusterTendency": "cm_clust_tend",
        "MaximumProbability": "cm_joint_max",
        # "SumSquares": "cm_sum_var",  # only if you confirmed this equivalence
    },
    "ngtdm": {
        "Contrast": "nt_contrast",
        "Coarseness": "nt_coarseness",
        "Busyness": "nt_busyness",
        "Complexity": "nt_complexity",
        "Strength": "nt_strength",
    },
    "glrlm": {
        "ShortRunEmphasis": "rlm_sre",
        "LongRunEmphasis": "rlm_lre",
        "LowGrayLevelRunEmphasis": "rlm_lgre",
        "HighGrayLevelRunEmphasis": "rlm_hgre",
        "ShortRunLowGrayLevelEmphasis": "rlm_srlge",
        "ShortRunHighGrayLevelEmphasis": "rlm_srhge",
        "LongRunLowGrayLevelEmphasis": "rlm_lrlge",
        "LongRunHighGrayLevelEmphasis": "rlm_lrhge",
        "GrayLevelNonUniformity": "rlm_glnu",
        "GrayLevelNonUniformityNormalized": "rlm_glnu_norm",
        "RunLengthNonUniformity": "rlm_rlnu",
        "RunLengthNonUniformityNormalized": "rlm_rlnu_norm",
        "RunVariance": "rlm_rl_var",
        "RunEntropy": "rlm_rl_entr",
        "RunPercentage": "rlm_r_pct",
        "GrayLevelVariance": "rlm_gl_var",
    },
    "glszm": {
        "SmallAreaEmphasis": "szm_sae",
        "LargeAreaEmphasis": "szm_lae",
        "LowGrayLevelZoneEmphasis": "szm_lglze",
        "HighGrayLevelZoneEmphasis": "szm_hglze",
        "SmallAreaLowGrayLevelEmphasis": "szm_szlge",
        "SmallAreaHighGrayLevelEmphasis": "szm_szhge",
        "LargeAreaLowGrayLevelEmphasis": "szm_lzlge",
        "LargeAreaHighGrayLevelEmphasis": "szm_lzhge",
        "GrayLevelNonUniformity": "szm_glnu",
        "GrayLevelNonUniformityNormalized": "szm_glnu_norm",
        "SizeZoneNonUniformity": "szm_sznu",
        "SizeZoneNonUniformityNormalized": "szm_sznun",
        "ZonePercentage": "szm_z_pct",
        "ZoneVariance": "szm_zs_var",
        "ZoneEntropy": "szm_zs_entr",
        "GrayLevelVariance": "szm_gl_var",
    },
    "ngldm": {  # display may say GLDM; IBSI family remains ngl_*
        "DependenceEntropy": "ngl_dc_entr",
        "DependenceVariance": "ngl_dc_var", 
        "HighDependenceEmphasis": "ngl_hde",
        "GrayLevelNonUniformity": "ngl_glnu",
        "DependenceNonUniformity": "ngl_dcnu",
        "DependenceNonUniformityNormalized": "ngl_dcnu_norm",
        "LowGrayLevelEmphasis": "ngl_lgce",
        "HighGrayLevelEmphasis": "ngl_hgce",
        "SmallDependenceLowGrayLevelEmphasis": "ngl_ldlge",
        "SmallDependenceHighGrayLevelEmphasis": "ngl_ldhge",
        "LargeDependenceLowGrayLevelEmphasis": "ngl_hdlge",
        "LargeDependenceHighGrayLevelEmphasis": "ngl_hdhge",
        "HighGrayLevelEmphasis": "ngl_hgce",
        "LowGrayLevelEmphasis": "ngl_lgce",
    },
    "gldm": {  # display may say GLDM; IBSI family remains ngl_*
        "SmallDependenceEmphasis": "dzm_sde",
        "LargeDependenceEmphasis": "dzm_lde",
        "GrayLevelNonUniformity": "dzm_glnu",
        "DependenceNonUniformity": "dzm_dcnu",
        "DependenceNonUniformityNormalized": "dzm_dcnu_norm",
        "GrayLevelVariance":"dzm_gl_var",
        "DependenceVariance":"dzm_zd_var",
        "DependenceEntropy":"dzm_zd_entr",
        "LowGrayLevelEmphasis": "dzm_lgce",
        "HighGrayLevelEmphasis": "dzm_hgce",
        "SmallDependenceLowGrayLevelEmphasis": "dzm_ldlge",
        "SmallDependenceHighGrayLevelEmphasis": "dzm_ldhge",
        "LargeDependenceLowGrayLevelEmphasis": "dzm_hdlge",
        "LargeDependenceHighGrayLevelEmphasis": "dzm_hdhge",
    },
    "firstorder": {
        "Mean": "stat_mean",
        "Median": "stat_median",
        "Minimum": "stat_min",
        "Maximum": "stat_max",
        "Variance": "stat_var",
        "Range": "stat_range",
        "Skewness": "stat_skew",
        "Kurtosis": "stat_kurt",
        "MeanAbsoluteDeviation": "stat_mad",
        "RobustMeanAbsoluteDeviation": "stat_rmad",
        "Energy": "stat_energy",
        "RootMeanSquared": "stat_rms",
        "10Percentile": "stat_p10",
        "90Percentile": "stat_p90",
        # Uniformity you mapped to IH; keep if that’s your convention
        "Uniformity": "ih_uniformity",
    },
    "shape": {
        "Sphericity": "morph_sphericity",
        "Elongation": "morph_pca_elongation",
        "Flatness": "morph_pca_flatness",
        "MajorAxisLength": "morph_pca_maj_axis",
        "MinorAxisLength": "morph_pca_min_axis",
        "Maximum3DDiameter": "morph_diam",
    },
}


def _pr_family_prefix(pr_class: str) -> str:
    return _PR_CLASS_TO_MIRP_PREFIX.get(pr_class, pr_class)

# ---------- Class-abbreviation policy ----------
def _family_to_abbr(use_ngldm: bool = True) -> Dict[str, str]:
    base = {
        "cm":   "GLCM",
        "rlm":  "GLRLM",
        "rl":   "GLRLM",
        "szm":  "GLSZM",
        "sz":   "GLSZM",
        "dzm":  "GLDZM",
        "ngt":  "NGTDM",
        "nt":   "NGTDM",   # <-- add this
        "ngl":  "NGLDM",
        "dm":   "GLDM",
        "gldm": "GLDM",
        "stat": "IS",
        "ih":   "IH",
        "ivh":  "IVH",
        "morph":"Morph",
        "li":   "LI",
        "diag": "Diagnostic",
        "firstorder": "Firstorder",
        "glcm": "GLCM",
        "glrlm": "GLRLM",
        "glszm": "GLSZM",
    }
    return base


# Canonical kernel names (case-insensitive keys)
KERNEL_CANON = {
    "logarithm": "Log",
    "log-sigma": "Log",
    "squareroot": "SquareRoot",
    "-square ": "Square",
    "exponential": "Exponential",
    "gabor": "Gabor",
    "laws_l5s5e5_energy_delta_7": "Laws",
    "lbp-2d": "LBP2D",
    "wavelet-hhh": "WaveletHHH",
    "wavelet-hhl": "WaveletHHL",
    "wavelet-hlh": "WaveletHLH",
    "wavelet-llh": "WaveletLLH",
    "wavelet-hll": "WaveletHLL",
    "wavelet-lhh": "WaveletLHH",
    "wavelet-lll": "WaveletLLL",
    "wavelet-lhl": "WaveletLHL",
    "gauss": "Gauss",
    "gauss_s2.0": "Gauss",
    "mean": "Mean",
    # drop this verbose sequence entirely
    "s_3.0_g_1.0_l_0.9_t_0.0": "",
}

# Regex kernels (picked up even if not in KERNEL_CANON)
RE_KERNELS = [
    (re.compile(r"gauss_s\d+(?:\.\d+)?", re.I), "Gauss"),
    (re.compile(r"mean", re.I), "Mean"),
    (re.compile(r"wavelet-[HL]{3}", re.I),      None),  # handled via KERNEL_CANON too
]

MAX_KERNEL_TAG_LEN = 16

MIRP_IBSI = {
    # --- Morphological (HCUG) ---
    "morph_volume": "RNU0",
    "morph_vol_approx": "YEKZ",
    "morph_area_mesh": "C0JK",
    "morph_av": "2PR5",
    "morph_comp_1": "SKGS",
    "morph_comp_2": "BQWJ",
    "morph_sph_dispr": "KRCK",
    "morph_sphericity": "QCFX",
    "morph_asphericity": "25C7",
    "morph_com": "KLMA",
    "morph_diam": "L0JK",
    "morph_pca_maj_axis": "TDIC",
    "morph_pca_min_axis": "P9VJ",
    "morph_pca_least_axis": "7J51",
    "morph_pca_elongation": "Q3CK",
    "morph_pca_flatness": "N17B",
    "morph_vol_dens_aabb": "PBX1",
    "morph_area_dens_aabb": "R59B",
    "morph_vol_dens_aee": "6BDE",
    "morph_area_dens_aee": "RDD2",
    "morph_vol_dens_conv_hull": "R3ER",
    "morph_area_dens_conv_hull": "7T7F",
    "morph_integ_int": "99N0",
    "morph_moran_i": "N365",
    "morph_geary_c": "NPT7",
    # note: ombb/mvee variants have "reference values absent" in MIRP

    # --- Local intensity (9ST6) ---
    "loc_peak_loc": "VJGA",
    "loc_peak_glob": "0F91",

    # --- Intensity-based statistics (UHIW) ---
    "stat_mean": "Q4LE",
    "stat_var": "ECT3",
    "stat_skew": "KE2A",
    "stat_kurt": "IPH6",
    "stat_median": "Y12H",
    "stat_min": "1GSF",
    "stat_p10": "QG58",
    "stat_p90": "8DWT",
    "stat_max": "84IY",
    "stat_iqr": "SALO",
    "stat_range": "2OJQ",
    "stat_mad": "4FUA",
    "stat_rmad": "1128",
    "stat_medad": "N72L",
    "stat_cov": "7TET",
    "stat_qcod": "9S40",
    "stat_energy": "N8CA",
    "stat_rms": "5ZWQ",

    # --- Intensity histogram (ZVCW) ---
    "ih_mean": "X6K6",
    "ih_var": "CH89",
    "ih_skew": "88K1",
    "ih_kurt": "C3I7",
    "ih_median": "WIFQ",
    "ih_min": "1PR8",
    "ih_p10": "GPMT",
    "ih_p90": "OZ0C",
    "ih_max": "3NCY",
    "ih_mode": "AMMC",
    "ih_iqr": "WR0O",
    "ih_range": "5Z3W",
    "ih_mad": "D2ZX",
    "ih_rmad": "WRZB",
    "ih_medad": "4RNL",
    "ih_cov": "CWYJ",
    "ih_qcod": "SLWD",
    "ih_entropy": "TLU2",
    "ih_uniformity": "BJ5W",
    "ih_max_grad": "12CE",
    "ih_max_grad_g": "8E6O",
    "ih_min_grad": "VQB3",
    "ih_min_grad_g": "RHQZ",

    # --- Intensity-volume histogram (P88C) ---
    "ivh_v10": "BC2M",
    "ivh_v25": "BC2M",
    "ivh_v50": "BC2M",
    "ivh_v75": "BC2M",
    "ivh_v90": "BC2M",
    "ivh_i10": "GBPN",
    "ivh_i25": "GBPN",
    "ivh_i50": "GBPN",
    "ivh_i75": "GBPN",
    "ivh_i90": "GBPN",
    "ivh_diff_v10_v90": "DDTU",
    "ivh_diff_v25_v75": "DDTU",
    "ivh_diff_i10_i90": "CNV2",
    "ivh_diff_i25_i75": "CNV2",
    # ivh_auc: reference values absent per MIRP

    # --- GLCM (LFYI) ---
    "cm_joint_max": "GYBY",
    "cm_joint_avg": "60VM",
    "cm_joint_var": "UR99",
    "cm_joint_entr": "TU9B",
    "cm_diff_avg": "TF7R",
    "cm_diff_var": "D3YU",
    "cm_diff_entr": "NTRS",
    "cm_sum_avg": "ZGXS",
    "cm_sum_var": "OEEB",
    "cm_sum_entr": "P6QZ",
    "cm_energy": "8ZQL",
    "cm_contrast": "ACUI",
    "cm_dissimilarity": "8S9J",
    "cm_inv_diff": "IB1Z",
    "cm_inv_diff_norm": "NDRX",
    "cm_inv_diff_mom": "WF0Z",
    "cm_inv_diff_mom_norm": "1QCO",
    "cm_inv_var": "E8JP",
    "cm_corr": "NI2N",
    "cm_auto_corr": "QWB0",
    "cm_clust_tend": "DG8W",
    "cm_clust_shade": "7NFM",
    "cm_clust_prom": "AE86",
    "cm_info_corr1": "R8DG",
    "cm_info_corr2": "JN9H",
    # cm_mcc: no identifier

    # --- GLRLM (TP0I) ---
    "rlm_sre": "22OV",
    "rlm_lre": "W4KF",
    "rlm_lgre": "V3SW",
    "rlm_hgre": "G3QZ",
    "rlm_srlge": "HTZT",
    "rlm_srhge": "GD3A",
    "rlm_lrlge": "IVPO",
    "rlm_lrhge": "3KUM",
    "rlm_glnu": "R5YN",
    "rlm_glnu_norm": "OVBL",
    "rlm_rlnu": "W92Y",
    "rlm_rlnu_norm": "IC23",
    "rlm_r_pct": "9ZK5",
    "rlm_gl_var": "8CE5",
    "rlm_rl_var": "SXLW",          # run length variance
    "rlm_rl_entr": "HJ9O",

    # --- GLSZM (9SAK) ---
    "szm_sae": "5QRC",
    "szm_lae": "48P8",
    "szm_lglze": "XMSY",
    "szm_hglze": "5GN9",
    "szm_szlge": "5RAI",
    "szm_szhge": "HW1V",
    "szm_lzlge": "YH51",
    "szm_lzhge": "J17V",
    "szm_glnu": "JNSA",
    "szm_glnu_norm": "Y1RO",
    "szm_sznu": "4JP3",
    "szm_sznun": "VB3A",
    "szm_z_pct": "P30P",
    "szm_gl_var": "BYLV",
    "szm_var": "3NSA",
    "szm_entr": "GU8N",

    # --- GLDZM (VMDZ) ---
    "dzm_sde": "0GBI",
    "dzm_lde": "MB4I",
    "dzm_lgze": "S1RA",
    "dzm_hgze": "K26C",
    "dzm_sdlge": "RUVG",
    "dzm_sdhge": "DKNJ",
    "dzm_ldlge": "A7WM",
    "dzm_ldhge": "KLTH",
    "dzm_glnu": "VFT7",
    "dzm_glnu_norm": "7HP3",
    "dzm_zdnu": "V294",
    "dzm_zdnu_norm": "IATH",
    "dzm_z_perc": "VIWW",
    "dzm_gl_var": "QK93",
    "dzm_zd_var": "7WT1",
    "dzm_zd_entr": "GBDU",

    # --- NGTDM (IPET) ---
    "nt_coarseness": "QCDE",
    "nt_contrast": "65HE",
    "nt_busyness": "NQ30",
    "nt_complexity": "HDEZ",
    "nt_strength": "1X9X",

    # --- NGLDM (REK0) ---
    "ngl_lde": "SODN",
    "ngl_hde": "IMOQ",
    "ngl_lgce": "TL9H",
    "ngl_hgce": "OAE7",
    "ngl_ldlge": "EQ3F",
    "ngl_ldhge": "JA6D",
    "ngl_hdlge": "NBZI",
    "ngl_hdhge": "9QMG",
    "ngl_glnu": "FP8K",
    "ngl_glnu_norm": "5SPA",
    "ngl_dcnu": "Z87G",
    "ngl_dcnu_norm": "OKJI",
    "ngl_dc_perc": "6XV8",
    "ngl_gl_var": "1PFV",
    "ngl_dc_var": "DNX2",
    "ngl_dc_entr": "FCBV",
    "ngl_dc_energy": "CAS9",
}

# ---------- PyRadiomics → MIRP tokens (extend as needed) ----------
PR_TOKEN_TO_MIRP_TOKEN = {
    # First-order → stat_*
    "RootMeanSquared": "stat_rms",
    "Mean": "stat_mean",
    "Median": "stat_median",
    "Variance": "stat_var",
    "Range": "stat_range",
    "Skewness": "stat_skew",
    "Kurtosis": "stat_kurt",
    "MeanAbsoluteDeviation": "stat_mad",
    "RobustMeanAbsoluteDeviation": "stat_rmad",
    "Energy": "stat_energy",
    "10Percentile": "stat_p10",
    "90Percentile": "stat_p90",
    "Uniformity": "ih_uniformity",  # closest MIRP token

    # GLCM → cm_*
    "Imc2": "cm_info_corr2",
    "Imc1": "cm_info_corr1",
    "Idm": "cm_inv_diff_mom",
    "Idmn": "cm_inv_diff_mom_norm",
    "Id": "cm_inv_diff",
    "Idn": "cm_inv_diff_norm",
    "InverseVariance": "cm_inv_var",
    "JointEntropy": "cm_joint_entr",
    "JointEnergy": "cm_energy",
    "Correlation": "cm_corr",
    "Contrast": "cm_contrast",
    "DifferenceEntropy": "cm_diff_entr",
    "DifferenceAverage": "cm_diff_avg",
    "DifferenceVariance": "cm_diff_var",
    "ClusterShade": "cm_clust_shade",
    "ClusterProminence": "cm_clust_prom",
    "ClusterTendency": "cm_clust_tend",
    "MaximumProbability": "cm_joint_max",
    "MCC": "cm_mcc",  # MIRP flags as non-IBSI
    
    "Autocorrelation": "cm_auto_corr",
    "SumEntropy":      "cm_sum_entr",
    "SumAverage":      "cm_sum_avg",
    "SumSquares":      "cm_sum_var",

    # GLRLM → rl_*
    "RunEntropy": "rl_entr",
    "ShortRunEmphasis": "rl_sre",
    "LongRunEmphasis": "rl_lre",
    "RunVariance": "rl_var",
    "RunPercentage": "rl_r_pct",
    "GrayLevelNonUniformity": "rl_glnu",
    "GrayLevelNonUniformityNormalized": "rl_glnu_norm",
    "RunLengthNonUniformity": "rl_rlnu",
    "RunLengthNonUniformityNormalized": "rl_rlnu_norm",
    "LowGrayLevelRunEmphasis": "rl_lglre",
    "HighGrayLevelRunEmphasis": "rl_hglre",
    "ShortRunLowGrayLevelEmphasis": "rl_srlgle",
    "ShortRunHighGrayLevelEmphasis": "rl_srhgle",
    "LongRunLowGrayLevelEmphasis": "rl_lrlgle",
    "LongRunHighGrayLevelEmphasis": "rl_lrhgle",

    # GLSZM → szm_*
    "ZonePercentage": "szm_z_pct",
    "ZoneVariance": "szm_var",
    "ZoneEntropy": "szm_entr",
    "SmallAreaEmphasis": "szm_sae",
    "LargeAreaEmphasis": "szm_lae",
    "LowGrayLevelZoneEmphasis": "szm_lglze",
    "HighGrayLevelZoneEmphasis": "szm_hglze",
    "SmallAreaLowGrayLevelEmphasis": "szm_salg",   # if you use these; keep consistent
    "SmallAreaHighGrayLevelEmphasis": "szm_sahg",
    "LargeAreaLowGrayLevelEmphasis": "szm_lalg",
    "LargeAreaHighGrayLevelEmphasis": "szm_lahg",
    "GrayLevelNonUniformity": "szm_glnu",
    "GrayLevelNonUniformityNormalized": "szm_glnu_norm",
    "SizeZoneNonUniformity": "szm_sznu",
    "SizeZoneNonUniformityNormalized": "szm_sznun",

    
    "SmallDependenceEmphasis": "dzm_sde",
    "LargeDependenceEmphasis": "dzm_lde",
    "GrayLevelNonUniformity": "dzm_glnu",
    "DependenceNonUniformity": "dzm_zdnu",
    "DependenceNonUniformityNormalized": "dzm_zdnu_norm",
    "DependeceVariance":"dzm_gl_var",
    "DependeceEntropy":"dzm_zd_entr",
    "LowGrayLevelEmphasis": "dzm_lgze",
    "HighGrayLevelEmphasis": "dzm_hgze",
    "SmallDependenceLowGrayLevelEmphasis": "dzm_sdlge",
    "SmallDependenceHighGrayLevelEmphasis": "dzm_sdhge",
    "LargeDependenceLowGrayLevelEmphasis": "dzm_ldlge",
    "LargeDependenceHighGrayLevelEmphasis": "dzm_ldhge",
    "HighGrayLevelEmphasis": "dzm_hgze",
    "LowGrayLevelEmphasis": "dzm_lgze",
    
    # (N)GLDM (PyRadiomics gldm) → ngl_*  (IBSI renamed GLDM → NGLDM)
    #"DependenceEntropy": "ngl_dc_entr",           # (choose the closest MIRP token)
    #"DependenceVariance": "ngl_dc_var",
    #"SmallDependenceEmphasis": "ngl_lde",         # “low dependence emphasis” in MIRP
    #"LargeDependenceEmphasis": "ngl_hde",         # “high dependence emphasis” in MIRP
    #"GrayLevelNonUniformity": "ngl_glnu",
    #"DependenceNonUniformity": "ngl_dcnu",
    #"DependenceNonUniformityNormalized": "ngl_dcnu_norm",
    #"LowGrayLevelEmphasis": "ngl_lgce",
    #"HighGrayLevelEmphasis": "ngl_hgce",
    #"SmallDependenceLowGrayLevelEmphasis": "ngl_ldlge",
    #"SmallDependenceHighGrayLevelEmphasis": "ngl_ldhge",
    #"LargeDependenceLowGrayLevelEmphasis": "ngl_hdlge",
    #"LargeDependenceHighGrayLevelEmphasis": "ngl_hdhge",

    # NGTDM → nt_*
    "Coarseness": "nt_coarseness",
    "Contrast": "nt_contrast",
    "Busyness": "nt_busyness",
    "Complexity": "nt_complexity",
    "Strength": "nt_strength",

    # Shape → morph_*
    "Sphericity": "morph_sphericity",
    "Elongation": "morph_pca_elongation",
    "Flatness": "morph_pca_flatness",
    "MajorAxisLength": "morph_pca_maj_axis",
    "MinorAxisLength": "morph_pca_min_axis",
    "Maximum3DDiameter": "morph_diam",
}

# Optional MIRP meta (add IBSI IDs you care about)
MIRP_META = {
    "cm_mcc": {"ibsi_id": None, "non_ibsi": True},
}

# ---------- Cleaning helpers ----------
_FILTER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"wavelet-[A-Z]{3}", r"lbp-2D", r"gauss_s[\d.]+", r"laws_[a-z0-9_]+",
    r"exponential", r"log-sigma-[\d.]+-mm(-[1-5])?", r"gradient",
    r"square(root)?", r"logarithm(ic)?", r"mean", r"delta_\d+",
]]
_POST_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"zscore", r"invar", r"peritumoral", r"intratumoral"
]]

import re

# Structured strip rules: (regex, replacement)
# - Most patterns drop completely ("")
# - The boundary-aware d1 pattern preserves the left boundary with \1
_STRIP_RULES = [
    # MIRP discretization/meta tags
    (re.compile(r"_w\d+(?:\.\d+)?", re.IGNORECASE), ""),        # _w25.0 / _w6
    (re.compile(r"_a0(?:\.0)?", re.IGNORECASE), ""),            # _a0.0 / _a0
    (re.compile(r"_2d_s", re.IGNORECASE), ""),                  # _2d_s
    (re.compile(r"_fbs(?:_[a-z0-9.]+)?", re.IGNORECASE), ""),   # _fbs / _fbs_w6.0

    # Remove standalone d1 with boundaries, keep the left boundary
    (re.compile(r"(^|_)d1(?=_|$)", re.IGNORECASE), r"\1"),
]
# --- humanization map (lowercase keys) ---
# Applied case-insensitively to the token part, using safe boundaries where possible.
HUMANIZE_MAP = {
    # multi-token
    "rlnu_norm": "RLNUNorm",
    "rlnu":"RLNU",
    "glnu_norm": "GLNUNorm",
    "glnu":"GLNU",
    "zs_entr": "ZSEntr",
    "zs_var": "ZoneSizeVariance",
    "peak_glob": "GIPeak",
    "zd_var": "ZDVar",
    "diff_entr": "DiffEntr",
    "integ_int": "IntegInt",
    "gauss_s2.0": "Gauss",
    "s_3.0_g_1.0_l_0.9_t_0.0": "",  # drop marker
    "l5s5e5_energy_delta_7":"",
    "l5s5e5_Energy_delta_7":"",
    "inv_diff_mom_norm": "IDMN",
    "info_corr2": "IMCorr2",
    "info_corr1": "IMCorr1",
    "sde":"SDE",
    "lgce": "LGCE",
    "hdlge": "HDLGLE",
    "szlge":"SZLGE",
    "lrlge":"LRLGE",
    "sahg":"SAHG",
    "int_mean_int_init_roi": "MeanInitRoi",
    "diff_i25_i75": "InterquartileRange",
    "clust_prom": "ClusterProminence",
    "zd_entr": "ZDEntr",
    "inv_var": "InVar",
    "vol_dens_aabb": "VolumeDensityAABB",
    "grad_g": "GradientMagnitude",
    "joint_max": "JointMax",
    "diff_mean": "DifferenceMean",
    "dc_energy": "JointEnergy",
    "int_bb_dim_y_init_roi": "BoundingBoxDimY_InitRoi",
    "v_mrg":"",
    "d_15":"",
    "zdnu":"ZDNU",
    "Minimum":"Min",
    "Maximum":"Max",
    "inv_diff_norm":"IDN",
    "salg":"SALGLE",
    "sum_entr":"SUMEnt",
    "contrast":"Contrast",
    "sae":"SAE",
    "hde":"HDE",
    "DependenceVariance":"DVariance",
    "lde":"LDE",
    "mrg":"",
    "Diagnostic_Mask-original":"Diagnostic",
    "Diagnostic_Image-original":"Diagnostic",
    "mcc":"MCC",
    "e5e5e5_Energy_delta_7_InVar":"",
    "-sigma-1-mm-3D":"",
    "s2.0":"",
    "2D_mean":"",
    
    
    # single-token
    "avg": "", "av": "mean", "mean": "mean",
    "median": "Median",
    "max": "Max", "min": "Min", "range": "Range",
    "kurt": "Kurtosis", "mode": "Mode", "skew": "Skewness",
    "qcod": "QCDisp",
    "entropy": "Entropy",
    "var": "Variance",
    "invar": "InVar", 
    "inversevariance": "InVar",
    "sphericity": "Sphericity",
    "energy": "Energy",
    "exponential": "Exponential",
    "complexity": "Complexity",
    "lre": "LRE",
    "2d": "2D", "3d": "3D",
}


IGNORE_EXACT = {"Prediction_Label", "ID"}

# Case-insensitive: ID.<digits> (e.g., ID.1, ID.23, id.007)
IGNORE_ID_DOT_NUM = re.compile(r"^ID\.\d+$", re.IGNORECASE)

# compile regex for replacements (ordered to avoid short-key stomping long phrases)
# sort keys by length desc so "diff_i25_i75" hits before "i25" etc.
_HUM_KEYS = sorted(HUMANIZE_MAP.keys(), key=len, reverse=True)


def _ibsi_id_from_mirp_token(raw_token: str) -> Optional[str]:
    """
    Look up IBSI ID by progressively trimming trailing '_parts' after removing
    known noise (w25.0, a0.0, fbs, d1, kernel fragments).
    """
    # fast path
    if raw_token in MIRP_IBSI:
        return MIRP_IBSI[raw_token]

    # sanitize tails (reuse your strip rules)
    t = raw_token
    for cre, repl in _STRIP_RULES:
        t = cre.sub(repl, t)
    # drop kernel fragments (raw/common)
    t = re.sub(r"(?i)(^|_)(gauss_s\d+(?:\.\d+)?|mean_d_\d+|wavelet-[HL]{3}|exponential|logarithm|squareroot|square|gabor|laws)(?=_|$)", r"\1", t)
    t = re.sub(r"__+", "_", t).strip("_")

    if t in MIRP_IBSI:
        return MIRP_IBSI[t]

    # progressively trim from right
    probe = t
    while "_" in probe:
        probe = probe.rsplit("_", 1)[0]
        if probe in MIRP_IBSI:
            return MIRP_IBSI[probe]
    return None

def _has_kernel_token(text: str, key: str) -> bool:
    """
    True iff `key` appears as a standalone kernel token in `text`.
    Boundaries allowed: start, end, underscore '_', or hyphen '-'.
    (Prevents matching 'Square' in 'RootMeanSquared'.)
    """
    if not key:
        return False
    # case-insensitive, boundary-aware
    pat = re.compile(rf'(?i)(^|[_\-]){re.escape(key)}(?=$|[_\-])')
    return pat.search(text) is not None

def _find_token_match(text: str, keys_to_vals: dict) -> Optional[str]:
    """
    Scan `keys_to_vals` with boundary-aware token matching; return mapped value (or "" if mapping says to drop).
    """
    tl = text.lower()
    for key, val in keys_to_vals.items():
        if key and _has_kernel_token(tl, key.lower()):
            return val or ""
    return None

def _drop_kernel_from_token(token: str, kernel_short: str) -> str:
    """
    Remove any occurrence of the canonical kernel token from the token string so we
    only add it once at the end. Works case-insensitively and respects underscores.
    """
    if not kernel_short:
        return token
    # (^|_)KERNEL(_|$)  -> collapse boundaries to avoid stray underscores
    pat = re.compile(rf"(?i)(^|_){re.escape(kernel_short)}(?=_|$)")
    cleaned = pat.sub(lambda m: "" if m.group(1) == "" else "_", token)
    cleaned = re.sub(r"__+", "_", cleaned).strip("_")
    return cleaned

def _replace_kernels_in_token(token: str) -> str:
    out = token
    # boundary-aware replacements
    for k, v in KERNEL_CANON.items():
        if not k:
            continue
        out = re.sub(rf'(?i)(^|[_\-]){re.escape(k)}([_\-]sigma-[\d.]+-mm(-?[23]D)?)?(?=$|[_\-])',
                     lambda m: m.group(1) + (v if v else ""),
                     out)
    # regex specifics
    out = re.sub(r'(?i)(^|[_\-])log-sigma-\d+(?:\.\d+)?-mm(?:-?[23]D)?(?=$|[_\-])', r'\1Log', out)
    out = re.sub(r'(?i)(^|[_\-])gauss_s\d+(?:\.\d+)?(?=$|[_\-])', r'\1Gauss', out)
    out = re.sub(r'(?i)(^|[_\-])mean_d_\d+(?=$|[_\-])', r'\1Mean', out)
    # remove stray wavelet triplets in-token
    out = re.sub(r'(?i)(^|[_\-])wavelet-[HL]{3}(?=$|[_\-])', r'\1', out)
    out = re.sub(r'-[HL]{3}', '', out, flags=re.IGNORECASE)
    out = re.sub(r'__+', '_', out).strip('_')
    return re.sub(r'__+', '_', out).strip('_')

def _is_ignored_column(name: str) -> bool:
    n = (name or "").strip()
    if n in IGNORE_EXACT:
        return True
    if IGNORE_ID_DOT_NUM.match(n):
        return True
    return False

def _strip_noise_from_token(token: str) -> str:
    """
    Remove MIRP discretization/meta tokens and wavelet triplets from the *token* part
    (not from the class prefix). Safe with/without capture groups.
    """
    cleaned = token
    for cre, repl in _STRIP_RULES:
        cleaned = cre.sub(repl, cleaned)

    # remove triplets like -HHL, -LLL that sometimes leak into the token
    cleaned = re.sub(r"-[HL]{3}", "", cleaned, flags=re.IGNORECASE)

    # collapse underscores and trim
    cleaned = re.sub(r"__+", "_", cleaned).strip("_")
    return cleaned

def _humanize_token(token: str) -> str:
    """
    Make token more readable:
      - apply HUMANIZE_MAP case-insensitively
      - normalize kernel substrings via KERNEL_CANON
      - preserve separators
    """
    out = token

    # normalize kernel substrings (SquareRoot, Log, Gauss, etc.)
    out = _replace_kernels_in_token(out)
    for k in _HUM_KEYS:
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])",
                     HUMANIZE_MAP[k], out, flags=re.IGNORECASE)
    return out

def _detect_margin(raw_name: str) -> bool:
    """True if the feature name indicates peritumoral (margin) region."""
    return "peritumoral" in raw_name.lower()

def _canonicalize_kernel(raw_name: str, kernels_in_files) -> str:
    """
    Return one short, canonical kernel tag (or "" if none).
    Priority:
      1) regex specifics (mean_d_<n>, gauss_s<d>)
      2) KERNEL_CANON (boundary-aware)
      3) kernels_in_files (boundary-aware, mapped through KERNEL_CANON if possible)
    """
    # 1) regex specifics first
    if re.search(r'(?i)(^|[_\-])log-sigma-\d+(?:\.\d+)?-mm(?:-?[23]D)?(?=$|[_\-])', raw_name):
        return "Log"
    if re.search(r'(?i)(^|[_\-])mean_d_\d+(?=$|[_\-])', raw_name):
        return "Mean"
    if re.search(r'(?i)(^|[_\-])gauss_s\d+(?:\.\d+)?(?=$|[_\-])', raw_name):
        return "Gauss"
    # wavelet handled via canonical map or list below

    # 2) canonical keys, boundary-aware
    val = _find_token_match(raw_name, KERNEL_CANON)
    if val is not None:
        return val

    # 3) provided kernels, boundary-aware, then map via KERNEL_CANON
    for k in kernels_in_files:
        if _has_kernel_token(raw_name, k):
            mapped = KERNEL_CANON.get(k.lower())
            if mapped is None:
                # no mapping; keep if short enough
                return k if len(k) <= MAX_KERNEL_TAG_LEN else ""
            return mapped or ""  # empty string in map means drop
    return ""

def _strip_context(name: str) -> str:
    """Strip modality/filter/ROI/postproc tags; keep class+token."""
    if name.lower() in {"id", "prediction_label", "config"}:
        return name
    s = re.sub(r"^(original|derived|log|square(root)?|wavelet).*?_", "", name)
    for pat in _FILTER_PATTERNS:
        s = re.sub(rf"_{pat.pattern}", "", s, flags=re.IGNORECASE)
    for pat in _POST_PATTERNS:
        s = re.sub(rf"_{pat.pattern}", "", s, flags=re.IGNORECASE)
    return re.sub(r"__+", "_", s).strip("_")

def _split_pr_class_token(clean: str) -> Tuple[str, str]:
    """'glcm_Imc2' → ('glcm','Imc2'); handle firstorder_*/shape_* fallbacks."""
    parts = clean.split("_", 2)
    if len(parts) >= 2 and parts[0].islower():
        return parts[0], parts[1]
    if clean.startswith("firstorder_"):
        return "firstorder", clean.split("_", 1)[1]
    if clean.startswith("shape_"):
        return "shape", clean.split("_", 1)[1]
    return "", clean

def _consensus_from_mirp_token(mirp_token: str, *, use_ngldm: bool = True) -> str:
    """'cm_info_corr2' → 'GLCM_info_corr2'."""
    fam_abbr = _family_to_abbr(use_ngldm)
    if "_" not in mirp_token:
        return mirp_token
    fam, rest = mirp_token.split("_", 1)
    return f"{fam_abbr.get(fam, fam.upper())}_{rest}"


# ======================================================================
#                           THE CLASS
# ======================================================================
class ConsensusFeatureFormatter:
    """
    Replacement for FeatureFormatter.

    Usage:
        fmt = ConsensusFeatureFormatter(features=<list of column names>, extractor="PyRadiomics" | "MIRP", ...)
        df  = fmt.run(title="LiverCRC")
        # df contains Consensus / Family / Abbr / MIRP_Token / IBSI_ID / Non_IBSI / Image_Kernel etc.
    """
    def __init__(self,
                 features: Optional[Iterable[str]] = None,
                 extractor: str = "MIRP",
                 output_path: str = ".",
                 logger=None,
                 error=None,
                 additional_ROIs: Iterable[str] = (),
                 generate_feature_profile_plot: bool = True,
                 run_id: str = "RPTK",
                 non_RPTK_format: bool = False,
                 use_ngldm: bool = True,
                 kernels_in_files: Optional[Iterable[str]] = None,
                 df: Optional[pd.DataFrame] = None,
                 feature_cols: Optional[Iterable[str]] = None):
        """
        Args:
            features: list of column names (previous behavior).
            df: an input DataFrame whose columns will be parsed (optional).
            feature_cols: which columns in df are radiomics features (default: all df columns).
        """
        self.df: Optional[pd.DataFrame] = df.copy() if df is not None else None
        self.feature_cols: Optional[List[str]] = list(feature_cols) if feature_cols is not None else None

        # if df is provided, derive features from it (unless explicit features were passed)
        if self.df is not None and (features is None or len(list(features)) == 0):
            cols = list(self.df.columns)
            if self.feature_cols is not None:
                cols = [c for c in cols if c in set(self.feature_cols)]
            self.features = cols
        else:
            self.features = list(features or [])
        self.extractor = extractor
        self.output_path = output_path
        self.additional_ROIs = list(additional_ROIs or [])
        self.generate_feature_profile_plot = generate_feature_profile_plot
        self.run_id = run_id
        self.non_RPTK_format = non_RPTK_format
        self.use_ngldm = use_ngldm
        
        try:
            os.makedirs(self.output_path, exist_ok=True)
        except Exception:
            # if we can’t create it, still proceed; we’ll fall back to console logging
            pass

        if kernels_in_files is None:
            self.kernels_in_files = [
                # wavelet variants (upper/lower/nonseparable)
                "Wavelet-HHH","Wavelet-LHH","Wavelet-HLH","Wavelet-HHL","Wavelet-HLL","Wavelet-LHL","Wavelet-LLH","Wavelet-LLL",
                "wavelet-HHH","wavelet-LHH","wavelet-HLH","wavelet-HHL","wavelet-HLL","wavelet-LHL","wavelet-LLH","wavelet-LLL",
                "WaveletHHH","WaveletLHH","WaveletHLH","WaveletHHL","WaveletHLL","WaveletLHL","WaveletLLH","WaveletLLL",
                # other filters
                "LoG","SquareRoot","LBP2D","LBP3D","Square","Logarithm","Gradient","Exponential",
                "log-","squareroot","square","logarithm","exponential","gradient","lbp-2D","lbp-3D",
                "laws","gabor","gauss","wavelet","separable","mean","log"
            ]
        else:
            self.kernels_in_files = list(kernels_in_files)

        # ---------- logging (with safe fallback) ----------
        if logger is None:
            try:
                self.logger = LogGenerator(
                    log_file_name=f"{self.output_path}/RPTK_feature_profiling_{self.run_id}.log",
                    logger_topic="RPTK Feature Profiling"
                ).generate_log()
            except Exception:
                # fallback to a simple console logger
                import logging
                self.logger = logging.getLogger("RPTK Feature Profiling")
                if not self.logger.handlers:
                    h = logging.StreamHandler()
                    h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                    self.logger.addHandler(h)
                self.logger.setLevel(logging.INFO)
                self.logger.warning("Could not create file logger; falling back to console logging.")
        else:
            self.logger = logger

        if error is None:
            try:
                self.error = LogGenerator(
                    log_file_name=f"{self.output_path}/RPTK_feature_profiling_{self.run_id}.err",
                    logger_topic="RPTK Feature Profiling error"
                ).generate_log()
            except Exception:
                import logging
                self.error = logging.getLogger("RPTK Feature Profiling error")
                if not self.error.handlers:
                    h = logging.StreamHandler()
                    h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                    self.error.addHandler(h)
                self.error.setLevel(logging.WARNING)
                self.error.warning("Could not create error file logger; using console logging.")
        else:
            self.error = error

        if not self.features:
            self.error.error("No Features provided!")
            raise ValueError("No Features provided!")

        # class IDs (long form) kept for profile plotting / checks
        self.mirp_feature_class_IDs = [
            "morphological","local_intensity","intensity-based_statistics","intensity-volume_histogram","intensity_histogram",
            "grey_level_co-occurrence_matrix","grey_level_run_length_matrix","grey_level_size_zone_matrix",
            "grey_level_distance_zone_matrix","neighbourhood_grey_tone_difference_matrix","neighbouring_grey_level_dependence_matrix",
            "diagnostic"
        ]
        self.pyradiomics_feature_class_IDs = [
            "diagnostics","morphological","firstorder","grey_level_co-occurrence_matrix","grey_level_distance_zone_matrix",
            "grey_level_run_length_matrix","grey_level_size_zone_matrix","neighbourhood_grey_tone_difference_matrix",
        ]

    # ------------------ Core normalization (both extractors) ------------------
    def _detect_kernel(self, raw_name: str) -> str:
        """Best-effort kernel tag from raw name."""
        for k in self.kernels_in_files:
            if k in raw_name:
                tail = raw_name.split(k, 1)[1]
                val = k + tail
                return val[:-len("_zscore")] if val.endswith("_zscore") else val
        # MIRP style "filter:token"
        if ":" in raw_name:
            return raw_name.split(":", 1)[0]
        return "Original_Image"

    def transform(self,
                  df: Optional[pd.DataFrame] = None,
                  mapping: Optional[Dict[str, str]] = None,
                  mode: str = "rename",
                  inplace: bool = False) -> Optional[pd.DataFrame]:
        """
        Apply the consensus name mapping to a DataFrame.

        Args:
            df: DataFrame to transform; if None and self.df is set, use self.df.
            mapping: {original_name: consensus}; if None, will call export_name_map on last run.
            mode:
                - "rename": rename columns using mapping (default).
                - "multiindex": keep data, set columns = MultiIndex (Original, Consensus or Original if unmapped).
                - "add_columns": keep original columns, also add new columns with consensus names (duplicated data).
            inplace: if True and mode="rename", rename df in place. Otherwise return a new DataFrame.

        Returns:
            DataFrame if not inplace, else None.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No DataFrame provided and no df stored in the formatter.")
            df = self.df

        # if no mapping provided, try using last report (stored on self after run)
        if mapping is None:
            if not hasattr(self, "_last_report") or self._last_report is None:
                raise ValueError("No mapping provided and no previous run() report found.")
            mapping = self.export_name_map(self._last_report)

        mode = mode.lower()
        if mode == "rename":
            if inplace:
                df.rename(columns=mapping, inplace=True)
                return None
            return df.rename(columns=mapping)

        elif mode == "multiindex":
            # MultiIndex (Original, Consensus or Original) for auditability
            pairs = []
            for c in df.columns:
                cons = mapping.get(c, c)
                pairs.append((c, cons))
            new_df = df.copy()
            new_df.columns = pd.MultiIndex.from_tuples(pairs, names=["Original", "Consensus"])
            return new_df

        elif mode == "add_columns":
            new_df = df.copy()
            for orig, cons in mapping.items():
                if cons not in new_df.columns and orig in new_df.columns:
                    new_df[cons] = new_df[orig]
            return new_df

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'rename', 'multiindex', or 'add_columns'.")

    def _normalize_pyrad_row(self, raw_name: str) -> dict:
        """
        Normalize a single PyRadiomics column name into a consensus record.

        Handles:
          - diagnostics_* (included, Non_IBSI=True)
          - class-aware PR→MIRP mapping
          - GLDM display override (Consensus uses GLDM_, MIRP token is ngl_*)
          - margin suffix and one canonical kernel suffix at the very end
          - IBSI lookup on MIRP token (with trimming/cleaning)
        """
        # 1) Quick ignore
        if _is_ignored_column(raw_name):
            return {
                "Name": raw_name, "Feature": None, "Feature_Class": None,
                "MIRP_Token": None, "Consensus": None, "Family": None, "Abbr": None,
                "Region": None, "IBSI_ID": None, "Non_IBSI": None, "Image_Kernel": None
            }

        # 2) Strip broad context (filters, zscore, etc.) to get "class_token..."
        cleaned = _strip_context(raw_name)
        pr_class, pr_token = _split_pr_class_token(cleaned)
        pr_class_l = (pr_class or "").lower()

        # 3) SPECIAL CASE: diagnostics_* → include and mark as Non-IBSI
        if pr_class_l == "diagnostics":
            diag_token = cleaned.split("_", 1)[1] if "_" in cleaned else pr_token
            # drop trivial tails like _z / _zscore
            diag_token = re.sub(r"(_zscore|_z)$", "", diag_token, flags=re.IGNORECASE)

            is_margin = _detect_margin(raw_name)
            # diagnostics have no kernel semantics
            consensus = f"Diagnostic_{diag_token}"
            if is_margin:
                consensus = f"{consensus}_margin"

            return {
                "Name": raw_name,
                "Feature": diag_token,
                "Feature_Class": "diagnostics",
                "MIRP_Token": f"diag_{diag_token}",
                "Consensus": consensus,
                "Family": "diag",
                "Abbr": "Diagnostic",
                "Region": "margin" if is_margin else "tumor",
                "IBSI_ID": None,
                "Non_IBSI": True,
                "Image_Kernel": "Original_Image",
            }

        # 4) Class-aware PR→MIRP token resolution
        fam_map = PR_CLASS_TOKEN_TO_MIRP.get(pr_class_l, {})
        mirp_token = fam_map.get(pr_token)
        if not mirp_token:
            # fallback to class-agnostic map if present
            mirp_token = PR_TOKEN_TO_MIRP_TOKEN.get(pr_token)
        if not mirp_token:
            # last resort: compose from a PR→MIRP family prefix
            fam_prefix = _PR_CLASS_TO_MIRP_PREFIX.get(pr_class_l, pr_class_l)
            mirp_token = f"{fam_prefix}_{pr_token}"

        # 5) MIRP family key for IBSI (cm / rlm / szm / ngl / ...)
        fam = mirp_token.split("_", 1)[0] if "_" in mirp_token else mirp_token

        # 6) DISPLAY family + long feature class (override GLDM display)
        display_abbr = _family_to_abbr(self.use_ngldm).get(fam, fam.upper())
        feature_class_long = self._pr_class_to_long(pr_class_l)
        if pr_class_l == "gldm":
            display_abbr = "GLDM"  # show GLDM to users
            feature_class_long = "grey_level_dependence_matrix"  # correct long name

        # 7) Token part (the bit after the family) → clean + humanize
        rest = mirp_token.split("_", 1)[1] if "_" in mirp_token else mirp_token

        is_margin = _detect_margin(raw_name)
        kernel_short = _canonicalize_kernel(raw_name, self.kernels_in_files)  # e.g., Mean, Gauss, WaveletHHL

        # remove literal 'peritumoral' from token (we’ll add '_margin' explicitly)
        tok = re.sub(r"peritumoral", "", rest, flags=re.IGNORECASE)
        tok = re.sub(r"__+", "_", tok).strip("_")

        # strip MIRP noise (w25.0, a0.0, 2d_s, fbs, d1, etc.)
        tok = _strip_noise_from_token(tok)
        # humanize short codes (SRE→SRE, qcod→QCDisp, inv_var→InVar, etc.)
        tok = _humanize_token(tok)

        # IMPORTANT: don’t let any in-token kernel duplicate the final suffix
        if kernel_short:
            tok = _drop_kernel_from_token(tok, kernel_short)

        # add region suffix if needed
        if is_margin:
            tok = f"{tok}_margin"

        # 8) Build consensus with the DISPLAY family (GLDM for PR gldm)
        consensus = f"{display_abbr}_{tok}"
        if kernel_short:
            consensus = f"{consensus}_{kernel_short}"

        # 9) IBSI lookup uses MIRP token (unchanged), with robust trimming
        meta = MIRP_META.get(mirp_token, {"ibsi_id": None, "non_ibsi": False})
        ibsi_id = _ibsi_id_from_mirp_token(mirp_token) or meta.get("ibsi_id")

        return {
            "Name": raw_name,
            "Feature": pr_token,
            "Feature_Class": feature_class_long,
            "MIRP_Token": mirp_token,
            "Consensus": consensus,
            "Family": fam,                     # MIRP family key for traceability
            "Abbr": display_abbr,              # DISPLAY prefix (e.g., GLDM for PR gldm)
            "Region": "margin" if is_margin else "tumor",
            "IBSI_ID": ibsi_id,
            "Non_IBSI": meta.get("non_ibsi", False),
            "Image_Kernel": kernel_short if kernel_short else "Original_Image",
        }



    def _normalize_mirp_row(self, raw_name: str) -> dict:
        token = raw_name.split(":", 1)[-1]
        fam = token.split("_", 1)[0] if "_" in token else None

        # base consensus (FIX: use token)
        consensus = _consensus_from_mirp_token(token, use_ngldm=self.use_ngldm)

        # region + kernel (define once)
        is_margin = _detect_margin(raw_name)
        kernel_short = _canonicalize_kernel(raw_name, self.kernels_in_files)

        if "_" in consensus:
            cls, tok = consensus.split("_", 1)

            # remove literal 'peritumoral' inside token (we'll add 'margin' explicitly)
            tok = re.sub(r"peritumoral", "", tok, flags=re.IGNORECASE)
            tok = re.sub(r"__+", "_", tok).strip("_")

            # strip MIRP noise and humanize
            tok = _strip_noise_from_token(tok)
            tok = _humanize_token(tok)

            # drop in-token kernel mention
            if kernel_short:
                tok = _drop_kernel_from_token(tok, kernel_short)

            # add region suffix
            if is_margin:
                tok = f"{tok}_margin"

            consensus = f"{cls}_{tok}"
        else:
            if is_margin:
                consensus = f"{consensus}_margin"

        # append one canonical kernel suffix
        if kernel_short:
            consensus = f"{consensus}_{kernel_short}"

        abbr = _family_to_abbr(self.use_ngldm).get(fam)
        meta = MIRP_META.get(token, {"ibsi_id": None, "non_ibsi": False})
        ibsi_id = _ibsi_id_from_mirp_token(token) or meta.get("ibsi_id")

        return {
            "Name": raw_name,
            "Feature": token.split("_", 1)[-1] if "_" in token else token,
            "Feature_Class": self._mirp_family_to_long(fam),
            "MIRP_Token": token,
            "Consensus": consensus,
            "Family": fam,
            "Abbr": abbr,
            "Region": "margin" if is_margin else "tumor",
            "IBSI_ID": ibsi_id,
            "Non_IBSI": meta.get("non_ibsi", False),
            "Image_Kernel": kernel_short if kernel_short else "Original_Image",
        }

    
    @staticmethod
    def _pr_class_to_long(pr_class: str) -> Optional[str]:
        return {
            "shape": "morphological",
            "firstorder": "firstorder",
            "glcm": "grey_level_co-occurrence_matrix",
            "glrlm": "grey_level_run_length_matrix",
            "glszm": "grey_level_size_zone_matrix",
            # FIX: GLDM is "grey_level_dependence_matrix" (not distance zone)
            "gldm": "grey_level_dependence_matrix",
            "ngtdm": "neighbourhood_grey_tone_difference_matrix",
            "diagnostics": "diagnostics",
            "": None,
        }.get(pr_class, pr_class)

    @staticmethod
    def _mirp_family_to_long(fam: Optional[str]) -> Optional[str]:
        return {
            "morph": "morphological",
            "li": "local_intensity",
            "stat": "intensity-based_statistics",
            "ivh": "intensity-volume_histogram",
            "ih": "intensity_histogram",
            "cm": "grey_level_co-occurrence_matrix",
            "rl": "grey_level_run_length_matrix",
            "sz": "grey_level_size_zone_matrix",
            "dm": "grey_level_distance_zone_matrix",
            "nt": "neighbourhood_grey_tone_difference_matrix",
            "ngl": "neighbouring_grey_level_dependence_matrix",  # if present in your data
            "diag": "diagnostic",
        }.get(fam, fam)

    # ------------------ Public API ------------------
    def run(self,
            title: str,
            return_dataframe: bool = False,
            rename_mode: str = "rename",
            df_override: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Main entry point. Returns the per-feature report DataFrame. Optionally also returns
        a DataFrame with consensus-named columns.

        Args:
            title: plot title / bookkeeping
            return_dataframe: if True and a DataFrame is available, also return a renamed/augmented DataFrame
            rename_mode: how to apply mapping if return_dataframe=True
                         ("rename", "multiindex", "add_columns")
            df_override: if provided, use this DataFrame for renaming instead of self.df

        Returns:
            report_df if return_dataframe is False
            (report_df, df_out) if return_dataframe is True and a DataFrame is available
        """
        # Prepare input list
        features = [f for f in self.features if not _is_ignored_column(f)]
        
        title = title.replace("_", " ")
       
        
        features = list(self.features)

        # remove generic suffixes and extra ROI tags (your behavior)
        features = [s.replace("_zscore", "") for s in features]
        for roi in self.additional_ROIs:
            if roi:
                features = [s.replace("_" + roi, "") for s in features]
                
        features = [f for f in features if not _is_ignored_column(f)]

        rows = []
        if self.extractor == "PyRadiomics":
            for f in features:
                # ignore obvious meta columns
                if f == "Image" or f == "Mask":
                    continue
    
                rows.append(self._normalize_pyrad_row(f))
        elif self.extractor == "MIRP":
            for f in features:
                if f == "Image" or f == "Mask":
                    continue
                rows.append(self._normalize_mirp_row(f))
        else:
            self.error.warning("Extractor not supported; returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if df.empty:
            self.error.warning("No features parsed.")
            return df

        # sanity check: warn if expected classes are missing
        expected = self.mirp_feature_class_IDs if self.extractor == "MIRP" else self.pyradiomics_feature_class_IDs
        if "Feature_Class" in df.columns:
            present = set([c for c in df["Feature_Class"] if c])
            for c in expected:
                if c not in present:
                    self.error.warning(f"Missing Feature Class: {c}")

        # optional profile plot
        if self.generate_feature_profile_plot:
            self.generate_feature_profile(df, title=f"{self.extractor} {title}", path=self.output_path)

        # reorder columns pleasantly
        preferred = ["Name","Feature","Feature_Class","Consensus","Abbr","Family",
                     "MIRP_Token","IBSI_ID","Non_IBSI","Image_Kernel"]
        others = [c for c in df.columns if c not in preferred]
        report = df.loc[:, [c for c in preferred if c in df.columns] + others]

        # keep last report for transform()
        self._last_report = report

        if not return_dataframe:
            return report

        # We need an input DataFrame to rename/augment
        base_df = df_override if df_override is not None else self.df
        if base_df is None:
            # no DataFrame available; just return the report
            return report

        name_map = self.export_name_map(report)  # {original_name: consensus}
        df_out = self.transform(base_df, mapping=name_map, mode=rename_mode, inplace=False)
        return report, df_out
    
    
        
    def generate_feature_profile(self, df: pd.DataFrame, title: str, path: str):
        """
        Same stacked bar as before: count Feature_Class per Image_Kernel.
        """
        data = pd.DataFrame(index=sorted(set(df["Feature_Class"])))
        for kernel in tqdm.tqdm(df["Image_Kernel"], desc="Generating Feature Profile"):
            counts = df.loc[df["Image_Kernel"] == kernel, "Feature_Class"].value_counts(dropna=False)
            data[kernel] = counts
        ax = data.fillna(0).plot.bar(stacked=True, colormap="rainbow_r")
        plt.legend(title="Kernels", loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_xlabel("Feature Classes")
        ax.set_ylabel("Number of Features")
        ax.set_title(title, color='black')
        plt.savefig(f"{path}/{title}.png", bbox_inches='tight', dpi=200)
        plt.close()

    def export_name_map(self, df: pd.DataFrame, index_col: str = "Name", value_col: str = "Consensus") -> Dict[str, str]:
        """
        Convenience for SHAP labels / harmonization:
            returns {original_name: consensus}
        """
        if index_col not in df or value_col not in df:
            return {}
        return pd.Series(df[value_col].values, index=df[index_col].values).to_dict()
