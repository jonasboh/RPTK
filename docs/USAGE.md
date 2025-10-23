# Using RPTK: Advanced Usage and Special Cases

## Table of Contents
- [Basic Workflow](#basic-workflow)
- [Basic Workflow Rerun](#basic-workflow-rerun)
- [Multiple ROIs Per Sample](#multiple-rois-per-sample)
  - [Case 1: Averaging Features Over Multiple ROIs Per Sample](#case-1-averaging-features-over-multiple-rois-per-sample)
  - [Case 2: Processing Each ROI Individually](#case-2-processing-each-roi-individually)
- [Tips for Advanced Usage](#tips-for-advanced-usage)
- [When to Use Each Strategy](#when-to-use-each-strategy)
- [Full Configuration Reference](#full-configuration-reference)
- [RPTK Output](#rptk-output)


The **Radiomics Processing Toolkit (RPTK)** provides a robust and configurable pipeline for medical image analysis. This guide focuses on **advanced usage scenarios**, especially those involving **multiple Regions of Interest (ROIs) per sample**.

Before diving into special cases, ensure you understand the [basic input CSV structure](../README.md#input-csv-configuration). 
Checkout the `example/` folder for example usage and detailed application of RPTK as well as tutorials of how a usecase could look like.

## Basic Workflow

### Use Case
- One Image per Patient
- One Mask per Patient
- Always label 1 in the Mask for the 

The simplest configuration of the input CSV file is to have for each patient one Image, one Segmentation, with one ROI always having the label 1.
Each row in your input file typically corresponds to a unique combination of:

- Image
- Mask
- Modality
- Prediction label

#### Example Input:

| ID        | Image                          | Mask                         | Modality | Prediction_Label | ROI_Label |
|-----------|--------------------------------|------------------------------|----------|------------------|-----------|
| Sample001 | /path/Sample001-image.nii.gz   | /path/Sample001-mask.nii.gz | CT       | 1                | 1         |
| Sample002 | /path/Sample002-image.nii.gz   | /path/Sample002-mask.nii.gz | CT       | 1                | 1         |


Optional metadata columns (e.g., `ROI_Label`, `Timepoint`, `Rater`) help define specific behaviors in the pipeline.

### Docker execution
Alternatively to installing the tool on your machine, we are offering a Docker image to run independently on your machine. For Docker execution, please see `docs/Docker/README.md` for detailed instructions.

#### Example Code for Basic Workflow:

```python
from os import path
import sys
import os

from rptk.rptk import RPTK
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

rptk = RPTK(path2confCSV= "/path/to/input.csv", 
             n_cpu = 30,  # number of cpu to use
             input_reformat=True,  # use reformat if the ID is not at the beginning of the files
             out_path="/path/to/output/folder/"
           )

rptk.run()

```

## Basic Workflow Rerun

Your analysis may fail or crash based on any hardware-related issues.
**RPTK can reuse the results from a previous run** to start from the latest processing point and make additional sanity checks for data completeness.

**Set the parameter `use_previous_output` to `True` and point to the output folder from the previous run, which ends with the RunID**

```python
from os import path
import sys
import os

from rptk.rptk import RPTK
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

rptk = RPTK(path2confCSV= "/path/to/input.csv", 
             n_cpu = 30,  # number of cpu to use
             input_reformat=True,  # use reformat if the ID is not at the beginning of the files
             use_previous_output = True,  # use previous output for feature extraction
             out_path="/path/to/output/folder/of/latest/result/RunID/"
           )

rptk.run()

```

---

## Multiple ROIs Per Sample

### Use Case 1: Averaging Features Over Multiple ROIs Per Sample

- One Image per Patient
- More than one Mask per Patient
- Masks are related to the same object (e.g. multiple tumors)

**Use case**: Multiple ROIs represent *the same anatomical or pathological structure*, such as multiple tumors in a single organ.

**Goal**: Combine the feature vectors into a single summary (mean, median, etc.) per sample.

### How to Configure:
- Provide separate rows in the input CSV for each ROI, using the same `ID` and image path.
- Use the `ROI_Label` column to distinguish between ROI instances.
- In your RPTK config, set `aggregate_rois=True`.

#### Example Input:

| ID        | Image                          | Mask                            | Modality | Prediction_Label | ROI_Label | Mask_Transformation |
|-----------|--------------------------------|----------------------------------|----------|------------------|-----------|----------------------|
| Sample001 | /path/Sample001-image.nii.gz   | /path/Sample001-tumor01.nii.gz  | CT       | 1                | 1         |                      |
| Sample001 | /path/Sample001-image.nii.gz   | /path/Sample001-tumor02.nii.gz  | CT       | 1                | 2         |                      |


#### Example Code for Averaging Features Over Multiple ROIs Per Sample:

```python
from os import path
import sys
import os

from rptk.rptk import RPTK
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

rptk = RPTK(path2confCSV= "/path/to/input.csv", 
             n_cpu = 30,  # number of cpu to use
             input_reformat=True,  # use reformat if the ID is not at the beginning of the files
             out_path="/path/to/output/folder/"
           )

rptk.run()

```

#### Notes:
- The final feature set per sample will be a statistical summary (e.g., mean) across ROIs.
- This is suitable when all ROIs contribute to the same predictive target.

---

### Use Case 2: Processing Each ROI Individually

**Use case**: ROIs have distinct meanings — e.g., *tumor vs. organ*, *lesion vs. surrounding tissue*.

**Goal**: Treat each ROI as a separate data point for feature extraction and modeling.

### How to Configure:
- Each ROI gets a distinct `ID` or `SubID` in the input CSV.
- This approach increases the number of rows/samples processed.

#### Example Input:

| ID        | Image                          | Mask                          | Modality | Prediction_Label | ROI_Label | Mask_Transformation |
|-----------|--------------------------------|-------------------------------|----------|------------------|-----------|----------------------|
| Sample001 | /path/Sample001-image.nii.gz   | /path/Sample001-tumor.nii.gz | CT       | 1                | 1         |                      |
| Sample001 | /path/Sample001-image.nii.gz   | /path/Sample001-organ.nii.gz | CT       | 1                | 2         | organ               |


#### Example Code for Processing Each ROI Individually:

```python
from os import path
import sys
import os

from rptk.rptk import RPTK
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

rptk = RPTK(path2confCSV= "/path/to/input.csv", 
             n_cpu = 30,  # number of cpu to use
             input_reformat=True,  # use reformat if the ID is not at the beginning of the files
             additional_rois_to_features=True,
             out_path="/path/to/output/folder/"
           )

rptk.run()

```


#### Notes:
- Use different prediction labels for each ROI if they are assessed independently.
- This method is ideal when ROIs represent functionally or pathologically distinct areas.

---

## Tips for Advanced Usage

- **Combine with `Timepoint` or `Rater` columns** to evaluate longitudinal changes or inter-rater variability.
- You can still use consensus feature selection or stability filtering across ROIs.
- Set `merge_on_id=False` if you want RPTK to treat each row as an independent unit, not merge ROIs by sample.

---

## When to Use Each Strategy

| Strategy                     | Use If...                                             |
|-----------------------------|--------------------------------------------------------|
| **Mean across ROIs**        | ROIs are functionally similar (e.g., multiple tumors) |
| **Separate ROIs as samples**| ROIs differ in purpose (e.g., organ vs. tumor)        |

---

For a full list of configuration options, see the configuration `CONFIG.md` file to get a complete explanation of the RPTK configuration.
The configuration file for rptk can be found at `src/config/rptk_config.json`.

## RPTK Output

The output folder is structured as follows:

```
RunID (timestamp of submission)
│
├── input_reformatted
│   ├── img/ (Reformatted images)
│   ├── msk/ (Reformatted segmentations)
│   ├── reformatted_input.csv (Summary of reformatted samples)
│
├── preprocessing_data
│   ├── Data_stats/ (Data fingerprint)
│   ├── accepted_perturbation/ (Accepted perturbations based on Dice threshold)
│   ├── multilabel_seg/ (Segmentations without artifacts)
│   ├── peritumoral_seg/ (Surrounding segmentations around target ROI)
│   ├── transformed_images/ (Transformed images)
│   ├── perturbed_seg/ (Perturbed segmentations)
│   ├── plots/ (Data overview plots)
│   ├── RunID_preprocessing_out.csv (Run configuration summary)
│   ├── clinical_data.csv (Clinical data extracted from input CSV)
│
├── Extracor (MIRP or PyRadiomics) (Feature extraction and filtering results)
│   ├── extracted_features/
│   │   ├── Extractor_extraction_RunID.csv (Extracted features)
│   │   ├── Memory_usage_profile_RunID.csv (Memory usage during extraction)
│   │   ├── RPTK_feature_extraction_Extractor_RunID.err (Errors/Warnings)
│   │   ├── RPTK_feature_extraction_Extractor_RunID.log (Extraction logs)
│   │   ├── tmp/subjects/ (CSV extractions for each sample)
│   ├── filtered_features/
│   │   ├── filtered_features.csv (Filtered features summary)
│   │   ├── Extractor_General_feature_Profile.csv/png (General feature profile)
│   │   ├── Extractor_Unstable_Feature_Profile.csv/png (Unstable feature profile)
│   │   ├── feature_alteration_profile.csv/png (Feature space size during filtering)
│   ├── selected_features/
│   │   ├── Feature_selection_RunID.csv (Selected features summary)
│   │   ├── Extractor_selected_feature_profile.csv/png (Kernel & feature class distribution)
│   │   ├── Performed_Feature_selection_status_RunID.csv (Feature selection status)
│   │   ├── model_out/ (Trained models and outputs)
│   │   ├── plots/ (Sequential Feature Selection visualization)
│   │   │   ├── RunID_Sequential_direction_model_selectionlib_Selection.png (Feature selection process)
│   │   │   ├── Label_distribution_RunID.png (Label distribution plots)
│   │   ├── Test_data_RunID_seed.csv (Test data split)
│   │   ├── Train_data_RunID_seed.csv (Train data split)
│   │   ├── Test_idx_RunID_seed.csv (Test data index)
│   │   ├── Train_idx_RunID_seed.csv (Train data index)
│   │   ├── Val_idx_RunID_seed.csv (Validation data index)
│   │   ├── Selected_Features_Correlation_Matrix.png (Correlation clusters in selected features)
│   ├── prediction/
│   │   ├── Prediction_summary.csv (Training/prediction results)
│   │   ├── model_parameters.csv (Trained model hyperparameters)
│   │   ├── Pretrained_model_overview_RunID.csv (Performance & stable parameter of pretraining)
│   │   ├── models/ (Trained models)
│   │   ├── optimization_log/ (Optimization status logs)
│   │   ├── plots/
│   │   │   ├── AUC/ (AUROC curves for training/testing)
│   │   │   ├── confusion_matrix/ (False/correct predictions visualization)
│   │   │   ├── Decision_region_plots/ (Decision-making visualization)
│   │   │   ├── optimization/ (Hyperparameter importance plots)
│   │   │   ├── pretraining/ (Performance curves for parameter estimation)
│   │   │   ├── SHAP/ (SHAP analysis plots)
```
