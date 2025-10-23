
#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from typing import Tuple

WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
RPTK_DIR = Path(os.environ.get("RPTK_DIR", "/opt/rptk"))
MIRP_DIR = Path(os.environ.get("MIRP_DIR", "/opt/mirp"))

def _copy_and_rewrite_csv(input_csv: Path, staging_dir: Path) -> Path:
    """Copy image/mask files referenced in CSV into staging_dir and rewrite CSV paths.
    CSV must have columns 'Image' and 'Mask'. Other columns are preserved.
    Returns path to the rewritten CSV in WORKDIR/tmp/rewritten.csv
    """
    df = pd.read_csv(input_csv)
    required_cols = {"Image", "Mask"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    staging_dir.mkdir(parents=True, exist_ok=True)

    new_image_paths = []
    new_mask_paths = []

    for idx, row in df.iterrows():
        # Resolve and copy image
        for col, collector in (("Image", new_image_paths), ("Mask", new_mask_paths)):
            src = Path(str(row[col])).expanduser()
            if not src.is_absolute():
                # If relative, interpret relative to the CSV location
                src = (input_csv.parent / src).resolve()
            if not src.exists():
                raise FileNotFoundError(f"Referenced file does not exist: {src}")
            dst = staging_dir / f"{idx}_{col}{src.suffix}"
            # Allow .nii.gz (double suffix)
            if str(src).endswith(".nii.gz"):
                dst = staging_dir / f"{idx}_{col}.nii.gz"
            shutil.copy2(src, dst)
            collector.append(dst.as_posix())

    # Overwrite the dataframe columns with container-internal paths
    df.loc[:, "Image"] = new_image_paths
    df.loc[:, "Mask"] = new_mask_paths

    rewritten_csv = WORKDIR / "tmp" / "rewritten.csv"
    rewritten_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(rewritten_csv, index=False)
    return rewritten_csv

def _ensure_pythonpath():
    # Ensure parent folder of 'rptk' and 'mirp' is on PYTHONPATH
    parent = Path("/opt")
    cur = os.environ.get("PYTHONPATH", "")
    if str(parent) not in cur.split(":"):
        os.environ["PYTHONPATH"] = f"{parent}:{cur}" if cur else str(parent)

def run(args) -> int:
    _ensure_pythonpath()

    # Validate dirs
    if not (RPTK_DIR.exists() and (RPTK_DIR / "__init__.py").exists()):
        print(f"[ERROR] RPTK package not found at {RPTK_DIR}. Expected __init__.py there.", file=sys.stderr)
        return 2
    if not (MIRP_DIR.exists() and MIRP_DIR.is_dir()):
        print("[WARN] MIRP directory not found. If RPTK requires MIRP adjacent to 'rptk', "
              "bind-mount your local mirp to /opt/mirp or provide MIRP_REPO build args.", file=sys.stderr)

    input_csv = Path(args.input_csv).resolve()
    output_folder = Path(args.output_folder).resolve()
    staging_dir = WORKDIR / "input"

    rewritten_csv = _copy_and_rewrite_csv(input_csv, staging_dir)

    # Build CLI for the user's run_rptk.py inside the image
    run_script = RPTK_DIR / "run_rptk.py"
    if not run_script.exists():
        alt = RPTK_DIR / "rptk-run.py"
        if alt.exists():
            run_script = alt
        elif args.run_script:
            run_script = Path(args.run_script)
        else:
            raise FileNotFoundError(
                f"Neither {RPTK_DIR/'run_rptk.py'} nor {RPTK_DIR/'rptk-run.py'} found, "
                f"and no --run_script provided."
            )

    # Compose command
    cmd = [
        sys.executable, str(run_script),
        "--input-csv", str(rewritten_csv),
        "--num-cpus", str(args.num_cpus),
        "--output-folder", str(output_folder),
    ]
    
    if args.perturbation:
        cmd += ["--perturbation", args.perturbation]
    cmd += ["--n-features", str(args.n_features)]
    cmd += ["--select-model", args.select_models]

    # Boolean flags
    if args.resample:
        cmd += ["--resample"]
    else:
        cmd += ["--no-resample"]

    if args.instability_filter:
        cmd += ["--instability-filter"]
    else:
        cmd += ["--no-instability-filter"]

    if args.rerun:
        cmd += ["--rerun"]

    if args.normalization:
        cmd += ["--normalize"]
        
    if args.config:
        cmd += ["--config"]
        
    print(">> Executing:", " ".join(cmd), flush=True)
    # Exec
    return os.execvp(cmd[0], cmd)  # replace current process

def main():
    p = argparse.ArgumentParser(description="RPTK Docker entrypoint: stage input CSV/files, rewrite paths, then run RPTK.")
    p.add_argument("--input_csv", required=True, help="Path to CSV on host. Must contain columns: Image, Mask.")
    p.add_argument("--output_folder", required=True, help="Container path to write outputs (use a bind mount).")
    p.add_argument("--num_cpus", type=int, default=1)
    p.add_argument("--perturbation", type=str, default=None)
    p.add_argument("--n_features", default=10)
    p.add_argument("--select_models", default="RandomForestClassifier")
    p.add_argument("--config", default=None, help="Config file for RPTK config json. If not set we execute with default.")
    p.add_argument("--resample", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    p.add_argument("--instability_filter", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    p.add_argument("--rerun", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    p.add_argument("--normalization", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    p.add_argument("--run_script", default=None, help="Optional path to run_rptk.py if not in cloned repo.")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()

