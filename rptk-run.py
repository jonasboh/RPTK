#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Assume rptk is installed (pip/poetry). e.g. `pip install -e .` in the RPTK repo.
from rptk.rptk import RPTK


def best_or_int(value: str) -> Union[int, str]:
    v = value.strip().lower()
    if v == "best":
        return "best"
    try:
        i = int(v)
        if i <= 0:
            raise argparse.ArgumentTypeError("--n-features must be a positive integer or 'best'")
        return i
    except ValueError as e:
        raise argparse.ArgumentTypeError("--n-features must be an integer or 'best'") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RPTK radiomics pipeline on an input CSV."
    )
    p.add_argument("--input-csv", required=True, type=Path, help="Path to the input CSV.")
    p.add_argument("--output-folder", required=True, type=Path, help="Folder for outputs.")
    p.add_argument("--num-cpus", type=int, default=1, help="Number of CPUs to use (default: 1).")

    p.add_argument(
        "--perturbation",
        choices=["supervoxel","connected_component", "random_walker"],
        default=None,
        help="Segmentation perturbation method."
    )
    p.add_argument(
        "--n-features",
        type=best_or_int,
        default=10,
        help="Number of features to keep (int) or 'best'."
    )
    p.add_argument(
        "--select-model",
        action="append",
        default=["RandomForestClassifier"],
        help="Model(s) for feature selection; can be repeated."
    )

    p.add_argument("--config", default=None,  help="give path to rptk config file for processing changes.")

    # Booleans as real flags
    p.add_argument("--resample", dest="resample", action="store_true", help="Resample image/mask to 1x1x1.")
    p.add_argument("--no-resample", dest="resample", action="store_false", help="Disable resampling.")
    p.set_defaults(resample=True)

    p.add_argument("--instability-filter", dest="instability_filter", action="store_true", help="Enable instability filter.")
    p.add_argument("--no-instability-filter", dest="instability_filter", action="store_false", help="Disable instability filter.")
    p.set_defaults(instability_filter=True)
    
    p.add_argument("--rerun", action="store_true", help="Use previous output folder as cache.")
    p.add_argument("--normalize", action="store_true", help="Enable image normalization (z-score by default).")

    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return p


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)


def main():
    args = build_parser().parse_args()
    configure_logging(args.verbose)

    # Input checks
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    print("Starting RPTK with configuration:")
    for k, v in vars(args).items():
        if k != "verbose":
            print("  %s: %s", k, v)

    # Initialize RPTK
    rptk = RPTK(
        path2confCSV=str(args.input_csv),
        n_cpu=args.num_cpus,
        self_optimize=False,
        input_reformat=True,
        use_previous_output=args.rerun,
        normalization=args.normalize,
        resampling=args.resample,
        rptk_config_json=args.config,
        out_path=str(args.output_folder) + "/",
    )

    # Run pipeline
    rptk.get_rptk_config()
    rptk.create_folders()
    rptk.check_input_csv_format()

    # If ID is a number, cast to string
    if "ID" in rptk.data.columns and is_numeric_dtype(rptk.data["ID"]):
        rptk.data["ID"] = rptk.data["ID"].astype(str)

    rptk.get_data_fingerprint()
    rptk.preprocessing()
    rptk.extract_features()

    if args.perturbation is not None:
        rptk.filter_features(perturbation_method=args.perturbation)
    else:
        rptk.filter_features()

    rptk.select_features()
    rptk.predict()

    print("Processing complete!")


if __name__ == "__main__":
    main()
