# RPTK Configuration File Documentation

This document explains the parameters used in `rptk_config.json` for the RPTK pipeline, along with their default values.

---

## Preprocessing_config

### General Parameters
- **isotropic_scale**: Target voxel size for isotropic resampling (mm).  
  **Default**: `1.0`
- **resampling**: Enable or disable spatial resampling of the input image.  
  **Default**: `false`
- **resample_slice_thickness_threshold**: Slice thickness threshold for applying resampling.  
  **Default**: `6`
- **min_roi_threshold / roi_threshold**: Minimum voxel count for a valid ROI.  
  **Default**: `3`
- **max_num_rois**: Max number of ROIs to process per sample.  
  **Default**: `1`
- **normalization**: Apply image normalization.  
  **Default**: `false`
- **normalization_method**: Method(s) for normalization (e.g., `z_score`).  
  **Default**: `["z_score"]`

### Transformation Kernels
- **transformation_kernels**: List of transformations applied before feature extraction.  
  **Default**: `["Wavelet", "Square", "LoG", "SquareRoot", "Logarithm", "Exponential", "Gradient", "LBP2D", "laws", "gabor", "gaussian", "mean"]`
- **transformation_kernel_config**: Configuration per transformation kernel.  
  **Default**: see `transformation_kernel_config` section in the config file

### Perturbation
- **segmentation_perturbation**: Enable segmentation perturbation.  
  **Default**: `true`
- **perturbation_method**: Perturbation methods to use.  
  **Default**: `["supervoxel", "connected_component", "random_walker"]`
- **roi_pert_repetition**: Number of perturbations to apply.  
  **Default**: `3`
- **perturbation_roi_adapt_type**: Strategy for adapting ROI during perturbation.  
  **Default**: `"fraction"`
- **perturbation_roi_adapt_size**: ROI adaptation sizes for perturbation.  
  **Default**: `[0.01, -0.01]`
- **dice_threshold**: Dice similarity coefficient threshold for accepting a perturbed segmentation.  
  **Default**: `0.80`
- **peritumoral_seg**: Whether to compute features for peritumoral areas.  
  **Default**: `true`
- **peri_dist**: Distance defining peritumoral region.  
  **Default**: `[3]`
- **expand_seg_dist**: Segmentation expansion margin.  
  **Default**: `[1]`
- **perturbation_factor**: Factor for perturbation strength.  
  **Default**: `0.05`
- **seg_closing_radius**: Radius for morphological closing.  
  **Default**: `10`
- **consider_multiple_labels**: Whether to process masks with multiple labels.  
  **Default**: `false`

---

## Feature_extraction_config

- **cropping**: Whether to crop to ROI bounding box.  
  **Default**: `false`
- **resampledPixelSpacing**: Resampling voxel spacing.  
  **Default**: `[1, 1, 1]`
- **bin_width**: Bin width for intensity discretization.  
  **Default**: `25.0`
- **min_num_bin**: Minimum number of bins allowed.  
  **Default**: `10.0`
- **max_num_bin**: Maximum number of bins allowed.  
  **Default**: `1000.0`

---

## Feature_filtering_config

- **variance_threshold**: Threshold for removing low-variance features.  
  **Default**: `0.1`
- **correlation_threshold**: Threshold for removing highly correlated features.  
  **Default**: `0.9`
- **ICC_threshold**: Threshold for Intraclass Correlation Coefficient filtering.  
  **Default**: `0.9`

---

## Feature_selection_config

- **models**: Models used for selection. `"all"` applies all available, or you can select a simple model or a combination of models [`"RandomForestClassifier"`,`"GradientBoostingClassifier"`,`"XGBClassifier"`,`"SVC"`, `"KNeighborsClassifier"`].  
  **Default**: `"RandomForestClassifier"`
- **sfs_lib**: Library used for sequential feature selection.  (`"mlxtend"` or `"sklearn"`)
  **Default**: `"mlxtend"`
- **save_model**: Save the resulting model.  
  **Default**: `true`
- **n_features**: Select the best n features or the total best selection of features (`"best"` or `"auto"`) (`"any number"` or `"best"` if `"sfs_lib"` is `"mlxtend"` or `"auto"` if `"sfs_lib"` is `"sklearn"`)
  **Default**: `10`
- **critical_feature_size**: Number of important features to retain (if `"best"` or `"auto"` are selected).  
  **Default**: `200`
- **min_feature / max_feature**: Minimum and maximum features to select (if `"best"` or `"auto"` are selected).  
  **Default**: `5` / `20`

---

## RPTK_prediction_config

- **use_cross_validation**: Whether to use k-fold cross-validation.  
  **Default**: `true`
- **cross_val_splits**: Number of folds for cross-validation.  
  **Default**: `5`
- **ensemble**: Enable ensemble modeling.  
  **Default**: `true`
- **model**: Models to apply. `"all"` applies all supported ones or select individual models [`"RandomForestClassifier"`,`"GradientBoostingClassifier"`,`"LGBMClassifier"`,`"TabNetClassifier"`,`"XGBClassifier"`,`"SVC"`]
  **Default**: `"all"`
- **shap_analysis**: Perform SHAP interpretability analysis.  
  **Default**: `true`
- **stable_pretraining**: Pretraining model based on performance fluctuation, based on model size to get the most stable model size with less performance fluctuations.
  **Default**: `true`
- **optimize**: Enable model hyperparameter optimization.  
  **Default**: `true`
- **optimization_iter**: Number of optimization iterations.  
  **Default**: `200`

---

This documentation is intended to help users understand and safely customize their RPTK pipeline configurations.
