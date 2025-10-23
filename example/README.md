# Examples

This folder contains example data, scripts, and tutorials demonstrating how to use **RPTK** and related workflows.

---

## Example Input and Scripts
- **`CRLM_example.csv`**  
  Example input CSV file derived from the [WORC database](https://www.medrxiv.org/content/10.1101/2021.08.19.21262238v1.full).  
  It shows how a configured CSV file for RPTK application can look.

- **`CRLM_example.py`**  
  Example script that processes the provided `CRLM_example.csv`.

- **`Download_CRLM_WORC.ipynb`**  
  Jupyter notebook to download samples from the WORC database.  
  Samples can also be downloaded directly via the [WORC project webpage](https://xnat.health-ri.nl/data/projects/worc).

## Tutorials
- **`RPTK_Tutorial.ipynb`**  
  A small tutorial notebook introducing RPTK functionality.

## RPTK Execution
- **`run_rptk.py`**  
  Command-line script to execute an RPTK workflow.  
  Uses `argparse` to configure parameters (e.g., specifying input data and settings). Run `python run_rptk.py --help` to see detailed description.

## Deep Learning Comparison
- **`DeepLearning/`**  
  Contains scripts to benchmark deep learning models against radiomics.

  - **`run_DeepLearning.py`**  
    Runs MONAI implementations of DenseNet and ResNet models for comparison.  
    Available models:
    - `ResNet18`  
    - `ResNet200`  
    - `DenseNet121`  
    - `DenseNet169`  
    - `DenseNet201`  
    - `DenseNet264`  

---

## References

- Starmans, M. P. A., Timbergen, M. J. M., Vos, M., et al. (2021).  
  The WORC Database: MRI and CT scans, segmentations, and clinical labels for classification, regression, and radiomics benchmarking.  
  *medRxiv*. https://doi.org/10.1101/2021.08.19.21262238  

- Cardoso, M. J., Li, W., Brown, R., *et al.* (2022).  
  MONAI: An open-source framework for deep learning in healthcare.  
  *arXiv preprint* arXiv:2211.02701. https://arxiv.org/abs/2211.02701 (**MONAI**)

---

## Usage Notes

- Start by exploring **`CRLM_example.csv`** and **`CRLM_example.py`** to understand the data structure and preprocessing.  
- Use **`RPTK_Tutorial.ipynb`** to get a quick introduction to RPTK.  
- For full workflows, configure and run **`run_rptk.py`** from the command line.  
- To explore deep learning comparisons, run **`DeepLearning/run_DeepLearning.py`** with your preferred model.

---

