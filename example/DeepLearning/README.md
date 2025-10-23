# Deep Learning baseline for RPTK (binary 3D image classification)

This module trains several 3D CNN baselines (DenseNet-121/169/201/264, ResNet-18/200) to **benchmark against RPTK radiomics** on a **binary** classification task. It includes a reproducible preprocessing pipeline (optional z-score normalization, mask-guided cropping, 1 mm isotropic resampling, resize) and a cross-validated training/evaluation routine driven by precomputed splits. Results (per-fold, ensemble) are written as CSVs, and optional experiment tracking via Weights & Biases is supported.

> Core script: `run_DeepLearning.py`. It builds data loaders, configures MONAI models, trains per fold with cosine-annealed LR, tracks **AUC** (primary) and accuracy on validation, saves the best model per fold, and produces test-time **ensemble** predictions by averaging per-fold probabilities.

---

## Main functionalities

1. **Reads inputs**
   - CSV with paths and labels; optional mask column for cropping and (optionally) a second training channel.
   - JSON with explicit `train`/`test` ID splits (k-fold training on `train`, hold-out evaluation on `test`).

2. **Preprocesses volumes**
   - Optional **z-score** normalization.
   - If `--cropping`: crop ROIs using the mask’s bounding box; then **resample to spacing** (default `1,1,1`) and **resize** (default `96³`, or `32³` if cropping).

3. **Trains multiple 3D CNNs**
   - DenseNet (121/169/201/264), ResNet-18, ResNet-200; plus HighResNet when training single-channel images.
   - Loss: Cross-Entropy; Optimizer: Adam; Scheduler: CosineAnnealingWarmRestarts; optional **early stopping**.

4. **Evaluates & logs**
   - Metrics on validation each epoch: **AUC** (MONAI ROCAUCMetric) and accuracy; saves the **best-AUC** model per fold.
   - Writes per-fold **validation** predictions and **test** predictions; additionally writes an **ensemble** CSV (mean of per-fold probabilities).
   - Optional Weights & Biases run logging (`--wandb_key`, `--project_name`, `--run_name`).

---

## Inputs

- **CSV (`--csv`)** with at least:
  - `ID` (unique sample identifier)
  - `Image` (path to NIfTI; `.nii.gz`)
  - `Prediction_Label` (0/1)
  - Optional: `Mask` (path to NIfTI mask) — required for cropping and/or adding mask as 2nd channel.  
  Notes:
  - If a column `Mask_Trasformation` exists, rows with non-null values are excluded.

- **Split JSON (`--split_json`)** with structure (see AutoRadiomics for details):
  ```json
  {
    "test": ["ID_1", "ID_2", "..."],
    "train": {
      "fold_0": [["train_id_..."], ["val_id_..."]],
      "fold_1": [["..."], ["..."]],
      "...": ["..."]
    }
  }
  ```

The script trains on each `fold_*`’s train/val split and evaluates on the shared test set. 
  
---

## Key options

### Data & preprocessing
| Option | Description | Default |
|--------|--------------|----------|
| `--z_norm` | Enables z-score normalization of image intensities. | *Disabled* |
| `--cropping` | Crops images to mask bounding box; automatically sets default resize to `32,32,32`. | *Disabled* |
| `--spacing <x,y,z>` | Resampling spacing in mm (isotropic voxel size). | `1,1,1` |
| `--resize_dim <x,y,z>` | Target image size after resampling/cropping. | `96,96,96` |

---

### Channels / masks
| Option | Description |
|--------|--------------|
| `--add_mask2train` | Adds mask as a second input channel (requires `Mask` column in CSV). |

---

### Optimization & training
| Option | Description |
|--------|--------------|
| `--batch_size` | Mini-batch size used for training. |
| `--max_epoch` | Maximum number of training epochs. |
| `--init_lr` | Initial learning rate. |
| `--min_lr` | Minimum learning rate for cosine annealing. |
| `--early_stopping` | Enables early stopping based on monitored metric (AUC). |
| `--patience` | Number of epochs without improvement before stopping. |
| `--min_delta` | Minimum AUC change to qualify as improvement. |
| `--no_data_aug` | Disables default data augmentations (flips, rotations, intensity shifts). |
| `--no_batch_norm` | Disables batch normalization layers in DenseNet configurations. |

---

### Device / logging
| Option | Description |
|--------|--------------|
| `--device <cuda|cpu>` | Specifies computation device (defaults to CUDA if available). |
| `--wandb_key` | API key for Weights & Biases experiment tracking. |
| `--project_name` | WANDB project name. |
| `--run_name` | WANDB run name (used for grouping experiments). |
| `--output_dir` | Directory where models, predictions, and preprocessed data are stored. |

---

## How to run

> **Hardware requirement:** An **NVIDIA GPU** with a CUDA-compatible PyTorch build is recommended for efficient training.  
> CPU-only mode is supported but not recommended due to excessive runtime.

### 1. Create environment & install dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
```
# Install PyTorch matching your CUDA version (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then install the remaining dependencies:
pip install -r DeepLearning-requirements.txt


## Example baseline run with commands

```bash
python run_DeepLearning.py \
  --csv /path/to/input.csv \
  --split_json /path/to/splits.json \
  --output_dir /path/to/output_dir \
  --project_name Project_name_for_wandb \
  --run_name Run_name_for_wandb \
  --max_epoch 200 --batch_size 15 --num_classes 2 --n_cpu 4 \
  --cropping --spacing 1,1,1 --resize_dim 96,96,96 \
  --early_stopping --patience 15 --min_delta 0.002
```

# Example run with settings to compare against RPTK

```bash
python run_DeepLearning.py \
  --proxy http://www.proxy.de \
  --csv  /path/to/input.csv \
  --split_json /path/to/autoradiomics/splits.json \
  --output_dir /path/to/output_dir \
 --project_name Project_name_for_wandb \
  --run_name Run_name_for_wandb \
  --max_epoch 200 --batch_size 15 --num_classes 2 --n_cpu 4 \
  --cropping --spacing 1,1,1 --resize_dim 96,96,96 \
  --monitor_metric auc --early_stopping --patience 15 --min_delta 0.002 \
  --use_ema --ema_decay 0.999 \
  --eval_checkpoints both \
  >&Run_logging.log
```
  
  # Output Structure
```bash  
  <output_dir>/<ModelName>/
  best_metric_model_classification3d_<fold>.pth
  validation_predictions_fold_<fold>.csv
  test_predictions_fold_<fold>.csv
  ensemble_test_predictions.csv
<output_dir>/preprocessed/
  <ID>_<orig>_preprocessed*.nii.gz
```










  
