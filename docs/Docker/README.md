# Radiomics Processing Toolkit (RPTK) – Docker

This repository provides a Docker image for running the **Radiomics Processing Toolkit (RPTK)** fully offline.  
The container bundles RPTK together with all required dependencies in a prebuilt Conda environment.
For more extensive experiments and custimization of framework optins we recommend to use the conda installation (see '')

---

## Summary

- [Quick Start](#quick-start)
  - [1. Load the Image](#1-load-the-image-offline)
  - [2. Prepare Input](#2-prepare-input)
  - [3. Run the Container](#3-run-the-container)
- [Troubleshooting](#troubleshooting)
- [Options](#options)
- [Example Commands](#example-commands)

---
 In order to run RPTK in the Docker you need to pull the Docker first

**Pull the image:**

```bash
docker pull jonasboh/rptk:v1.0
```

## Quick Start

### 1. Load the Image (offline)

Run this once to load the image into Docker:

```bash
docker load -i rptk_010_docker.tar
```

You should then see something like:
```bash
Loaded image: rptk:offline
```

---

### 2. Prepare Input

Your input data structure:
```bash
project_folder/
├─ input.csv
├─ images/...
└─ masks/...
```

You need a CSV file with at least these columns:

- **Image** – absolute path to `.nii.gz` (or other imaging file format)  
- **Mask** – absolute path to corresponding mask file  

Example:
```csv
Image,Mask,ID, Modality, Prediction_Label
data/images/P001_image.nii.gz,/data/masks/P001_mask.nii.gz,P001, CT, 0
data/images/P002_image.nii.gz,/data/masks/P002_mask.nii.gz,P002, CT, 1
```

---

### 3. Run the Container

Run the tool by mounting your input and output folders:

```bash
docker run --rm -it rptk:offline --help
```

Run the container (for detailed instructions see section Options below).

```bash
docker run --rm \
  -u $(id -u):$(id -g) \
  --mount type=tmpfs,destination=/workspace/input,tmpfs-mode=1777 \
  --mount type=tmpfs,destination=/workspace/tmp,tmpfs-mode=1777 \
  -v /ABS/PATH/project_folder/:/data \
  rptk:offline \
  --input_csv /data/CRLM_docker_test.csv \
  --output_folder /data/out \
  --num_cpus 8
```
All paths **must** be absolute paths on your machine.

- Replace `/ABS/PATH/...` with your actual paths.  
- Outputs appear under `/data/out/<RunID>`.

---

## Verify the installation

To check that everything works correctly:

```bash
    docker run --rm --entrypoint "" rptk:offline \
      /opt/conda/envs/rptk/bin/python -c \
      "import pandas, psutil, radiomics, rptk; print('OK:', pandas.__version__, radiomics.__version__, hasattr(rptk,'RPTK'))"
```

Expected output:
```bash
    OK: 2.2.1 v3.0.1 True
```
## Help

List all available arguments:
```bash
    docker run --rm rptk:offline --help
```

## Troubleshooting

- **FileNotFoundError**  
  Check that the absolute paths in your CSV are correctly mounted inside the container.

- **PermissionError in `/workspace/input` or `/workspace/tmp`**  
  Ensure you are using the `tmpfs` mounts as shown in the run command.

- **PermissionError in `/workspace/output`**  
  Make sure your output directory is writable by you, or run with:
  ```bash
  --user $(id -u):$(id -g)
  ```
---

### Required arguments

- `--input-csv PATH`  
  Path to the input CSV file (must have columns `Image` and `Mask` with absolute paths).

- `--output-folder PATH`  
  Directory where results will be written. The pipeline will create a timestamped subfolder inside.

---







