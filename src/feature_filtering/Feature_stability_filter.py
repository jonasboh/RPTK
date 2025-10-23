import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Dict, List

# Optional: your logger factory
# from rptk.src.config.Log_generator_config import LogGenerator
from rptk.src.feature_filtering.Feature_formater import FeatureFormatter


class FeatureStabilityFilter:
    """
    Compute ICC(1,1) per feature (one-way random effects, single measurement),
    plot CIs, and select unstable features based on a threshold.

    Key design decisions (statistically sound & Pingouin-compatible behavior):
    - Do NOT impute missing ratings with 0 (destroys variance structure).
    - Average duplicate (target, rater) pairs before pivoting.
    - Drop subjects without complete rater coverage for the feature.
    - Use classical ANOVA decomposition for ICC(1,1).
    - F-test and 95% CI via McGraw & Wong transform of the F-interval.
    """

    def __init__(
        self,
        extractor: str = "MIRP",
        logger: logging.Logger or None = None,
        error: logging.Logger or None = None,
        df_data: pd.DataFrame = pd.DataFrame(),
        ICC_threshold: float = 0.90,
        RunID: str = "",
        out_path: str = "",
    ):
        self.extractor = extractor
        self.logger = logger or logging.getLogger("RPTK Feature Stability Filter")
        self.error = error or logging.getLogger("RPTK Feature Stability Filter error")
        self.df_data = df_data
        self.ICC_threshold = float(ICC_threshold)  # DO NOT override later
        self.RunID = RunID
        self.out_path = out_path

        self.icc_error = ""

        if self.df_data.empty:
            self.error.warning("No data included for feature stability filtering.")

    # ---------- Core statistics (license-clean) ----------

    @staticmethod
    def icc_oneway_icc1(
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        nan_policy: str = "omit",
    ) -> pd.DataFrame:
        """
        ICC(1,1): one-way random effects, single measurement.

        Parameters
        ----------
        data : long/tidy DataFrame with columns [targets, raters, ratings]
        targets : subject identifier column
        raters : rater identifier column (here: Mask_Transformation)
        ratings : numeric rating column
        nan_policy : {'omit','raise'}

        Returns
        -------
        DataFrame with columns:
          ['Type','ICC','F','df1','df2','pval','CI95%','n_targets','n_raters']
        CI95% is a np.ndarray [low, high].
        """
        if nan_policy not in {"omit", "raise"}:
            raise ValueError("nan_policy must be 'omit' or 'raise'.")

        df = data[[targets, raters, ratings]].copy()
        if nan_policy == "omit":
            df = df.dropna(subset=[targets, raters, ratings])
        elif df[[targets, raters, ratings]].isna().any().any():
            raise ValueError("NaNs present but nan_policy='raise'.")

        # Average duplicates per (target, rater) pair
        df_agg = df.groupby([targets, raters], as_index=False, sort=False)[ratings].mean()

        # Wide: subjects × raters
        wide = df_agg.pivot(index=targets, columns=raters, values=ratings)
        # Enforce complete subjects (Pingouin-like)
        wide = wide.dropna(axis=0, how="any")
        # Safety: drop all-NaN columns
        wide = wide.dropna(axis=1, how="all")

        X = wide.to_numpy(dtype=float)
        n, k = X.shape
        if n < 2 or k < 2:
            raise ValueError("Need at least 2 targets and 2 raters after NaN handling.")

        # One-way ANOVA
        row_means = X.mean(axis=1, keepdims=True)
        grand = X.mean()

        ss_between = k * np.sum((row_means - grand) ** 2)
        df_between = n - 1
        ms_between = ss_between / df_between

        ss_within = np.sum((X - row_means) ** 2)
        df_within = n * (k - 1)
        ms_within = ss_within / df_within

        # ICC(1,1)
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)

        # F-test for H0: ICC = 0 (i.e., MS_between == MS_within)
        F = ms_between / ms_within
        pval = stats.f.sf(F, df_between, df_within)

        # 95% CI via McGraw & Wong transform from variance ratio
        alpha = 0.05
        F_lower = F / stats.f.ppf(1 - alpha / 2, df_between, df_within)
        F_upper = F * stats.f.ppf(1 - alpha / 2, df_within, df_between)
        ci_low = (F_lower - 1) / (F_lower + (k - 1))
        ci_high = (F_upper - 1) / (F_upper + (k - 1))

        if np.isnan(float(icc)):
            icc = 1.0
        return pd.DataFrame(
            {
                "Type": ["ICC1"],
                "ICC": [float(icc)],
                "F": [float(F)],
                "df1": [int(df_between)],
                "df2": [int(df_within)],
                "pval": [float(pval)],
                "CI95%": [np.array([float(ci_low), float(ci_high)], dtype=float)],
                "n_targets": [int(n)],
                "n_raters": [int(k)],
            }
        )

    # ---------- Batch computation ----------

    def calculate_ICC(self, df: pd.DataFrame = pd.DataFrame(), feature_class: str = "") -> pd.DataFrame:
        """
        Compute ICC(1,1) per feature on long-format data with columns:
        'ID', 'Mask_Transformation', <feature columns...> (plus optional 'config').

        Returns a DataFrame with columns:
        ['Feature','ICC1','pval','CI0','CI1','n_targets','n_raters']
        """
        if df.empty:
            df = self.df_data.copy()

        # Do NOT fill NaN globally (critical)
        features = [c for c in df.columns if c not in ("ID", "Mask_Transformation", "config")]
        results = []

        for feat in features:
            try:
                sub = df[["ID", "Mask_Transformation", feat]].copy()
                sub = sub.rename(columns={feat: "rating"})
                # Drop NaN ratings only for this feature
                sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")
                sub = sub.dropna(subset=["rating"])
                if sub.empty:
                    continue

                icc_df = self.icc_oneway_icc1(
                    data=sub, targets="ID", raters="Mask_Transformation", ratings="rating", nan_policy="omit"
                )

                results.append(
                    {
                        "Feature": feat,
                        "ICC1": float(icc_df.loc[0, "ICC"]),
                        "pval": float(icc_df.loc[0, "pval"]),
                        "CI0": float(icc_df.loc[0, "CI95%"][0]),
                        "CI1": float(icc_df.loc[0, "CI95%"][1]),
                        "n_targets": int(icc_df.loc[0, "n_targets"]),
                        "n_raters": int(icc_df.loc[0, "n_raters"]),
                    }
                )
            except Exception as e:
                self.error.warning(f"ICC calculation failed for {feat}: {e}")
                self.icc_error = str(e)
                continue

        ICC = pd.DataFrame(results)
        # Ensure numeric dtype for filtering
        if not ICC.empty:
            ICC["ICC1"] = ICC["ICC1"].astype(float)
            ICC["CI0"] = ICC["CI0"].astype(float)
            ICC["CI1"] = ICC["CI1"].astype(float)
        return ICC

    # ---------- Visualization ----------

    @staticmethod
    def _plot_ci_row(ax, y, mean, lo, hi, line_width=0.25, color="#2187bb"):
        """Draw a horizontal CI whisker at y with a point at mean."""
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.0)
        ax.plot([lo, lo], [y - line_width / 2, y + line_width / 2], color=color, linewidth=1.0)
        ax.plot([hi, hi], [y - line_width / 2, y + line_width / 2], color=color, linewidth=1.0)
        ax.plot(mean, y, "o", color="#f44336", markersize=4)

    def generate_plot(
        self,
        feature_class: str,
        df: pd.DataFrame,
        kernel: str = "",
        threshold: Optional[float] = None,
        inches_per_row: float = 0.30,     # visual density per feature
        min_height_in: float = 2.0,       # minimum figure height (inches)
        dpi: int = 120,                   # rendering DPI for PNG
        max_rows_per_fig: Optional[int] = None,  # optional manual cap
        prefer_vector_if_large: bool = False  # optionally switch to SVG for huge plots
    ) -> None:
        """
        Plot ICC1 with 95% CI for each feature (one panel per feature class),
        splitting into multiple figures when there are many features to avoid
        exceeding raster size limits.
        """

        if df.empty:
            return

        # Output path
        base_dir = os.path.join(self.out_path, "plots", "ICC")
        output_dir = os.path.join(base_dir, kernel) if kernel else base_dir
        os.makedirs(output_dir, exist_ok=True)

        # Compute a safe upper bound of rows per figure based on pixel limits.
        # Keep a margin below 65536 to be safe (e.g., 64000 px).
        max_px = 64000
        safe_rows_from_pixels = max(
            1,
            int(max_px / max(dpi * inches_per_row, 1e-6))
        )

        # Final rows per figure: user cap (if provided) ∧ pixel-safe cap
        rows_per_fig = safe_rows_from_pixels if max_rows_per_fig is None else min(max_rows_per_fig, safe_rows_from_pixels)

        # Chunk the dataframe
        n = len(df)
        n_parts = (n + rows_per_fig - 1) // rows_per_fig

        for part_idx in range(n_parts):
            start = part_idx * rows_per_fig
            stop = min((part_idx + 1) * rows_per_fig, n)
            sub = df.iloc[start:stop].reset_index(drop=True)

            fig_height = max(min_height_in, inches_per_row * len(sub))
            fig, ax = plt.subplots(figsize=(10, fig_height))  # inches

            title = f"ICC Feature Stability for {feature_class}"
            if n_parts > 1:
                title += f" (Part {part_idx + 1} / {n_parts})"
            ax.set_title(title)

            ax.set_xlim(0, 1)  # cosmetic clipping
            ax.set_xlabel("ICC(1,1)")
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(sub["Feature"].tolist())

            for i, row in sub.iterrows():
                mean, lo, hi = float(row["ICC1"]), float(row["CI0"]), float(row["CI1"])
                # clip CI to [0,1] for display neatness
                plo = max(0.0, min(1.0, lo))
                phi = max(0.0, min(1.0, hi))
                pmean = max(0.0, min(1.0, mean))
                self._plot_ci_row(ax, i, pmean, plo, phi)

            if threshold is not None:
                ax.axvline(x=float(threshold), color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.2f}")
                ax.legend(loc="upper left")

            fig.tight_layout()

            # File naming: single file if one chunk, else add part suffix
            if n_parts == 1:
                stem = f"{feature_class}"
            else:
                stem = f"{feature_class}__part{part_idx + 1:02d}"

            
            out_file = os.path.join(output_dir, f"{stem}.pdf")
            fig.savefig(out_file, dpi=dpi)

            plt.close(fig)

    # ---------- Filtering ----------

    def filter_for_unstable_feature_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only features with ICC1 < threshold.
        """
        if df.empty:
            return df
        return df.loc[df["ICC1"] < float(self.ICC_threshold)]

    def plot_icc_by_profile(self, ICC: pd.DataFrame, profile_path: Optional[str] = None) -> None:
        """
        Plot ICC per feature class using the extractor-specific General Feature Profile.
        - Expects profile CSV with columns: ['Feature_Class', 'Name']
        - Expects ICC DataFrame with columns: ['Feature', 'ICC1', 'CI0', 'CI1']
        - For each Feature_Class in the profile, selects ICC rows whose Feature is in Name for that class,
        sorts by CI0 ascending, and calls self.generate_plot(feature_class=..., df=...).
        - If the profile CSV does not exist, falls back to a single 'ALL' plot.
        """

        # Basic sanity check on ICC columns (fail fast with clear message)
        req_icc_cols = {"Feature", "ICC1", "CI0", "CI1"}
        missing = req_icc_cols - set(ICC.columns)
        if missing:
            msg = f"ICC DataFrame missing required columns: {missing}"
            print(msg)
            if hasattr(self, "error"):
                self.error.warning(msg)
            return

        # Resolve profile CSV path
        if profile_path is None:
            profile_path = os.path.join(self.out_path, f"{self.extractor}_General_Feature_Profile.csv")

        if not os.path.exists(profile_path):
            # Fallback: single ALL plot (your current behavior)
            try:
                ICC_sorted = ICC.sort_values(by="CI0", ascending=True)
                self.generate_plot(feature_class="ALL", df=ICC_sorted, kernel="", threshold=self.ICC_threshold)
            except Exception as ex:
                msg = f"Could not plot ICC distribution plot: {ex}"
                print(msg)
                if hasattr(self, "error"):
                    self.error.warning(msg)
            return

        # Load the profile with exact expected columns
        try:
            profile_df = pd.read_csv(profile_path, usecols=["Feature_Class", "Name"])
        except Exception as ex:
            # If loading fails, fall back once
            msg = f"Could not read profile CSV '{profile_path}': {ex}"
            print(msg)
            if hasattr(self, "error"):
                self.error.warning(msg)
            try:
                ICC_sorted = ICC.sort_values(by="CI0", ascending=True)
                self.generate_plot(feature_class="ALL", df=ICC_sorted, kernel="", threshold=self.ICC_threshold)
            except Exception as ex2:
                msg2 = f"Could not plot ICC distribution plot: {ex2}"
                print(msg2)
                if hasattr(self, "error"):
                    self.error.warning(msg2)
            return

        # Iterate over feature classes exactly as listed in the profile
        feature_classes = profile_df["Feature_Class"].dropna().unique().tolist()

        plotted_any = False
        for fclass in feature_classes:
            # All feature names for this class
            names = (
                profile_df.loc[profile_df["Feature_Class"] == fclass, "Name"]
                .dropna()
                .unique()
                .tolist()
            )

            if not names:
                continue

            # Select matching features from ICC
            sub = ICC[ICC["Feature"].isin(names)].copy()
            if sub.empty:
                continue

            # Your ordering
            sub = sub.sort_values(by="CI0", ascending=True)

            # One plot per feature class
            try:
                self.generate_plot(
                    feature_class=str(fclass),
                    df=sub,
                    kernel="",
                    threshold=self.ICC_threshold,
                )
                plotted_any = True
            except Exception as ex:
                msg = f"Could not plot ICC for class '{fclass}': {ex}"
                print(msg)
                if hasattr(self, "error"):
                    self.error.warning(msg)

        # If nothing was plotted (no overlap), fall back once to ALL
        if not plotted_any:
            try:
                ICC_sorted = ICC.sort_values(by="CI0", ascending=True)
                self.generate_plot(feature_class="ALL", df=ICC_sorted, kernel="", threshold=self.ICC_threshold)
            except Exception as ex:
                msg = f"Could not plot ICC distribution plot (fallback ALL): {ex}"
                print(msg)
                if hasattr(self, "error"):
                    self.error.warning(msg)


    # ---------- Orchestration ----------

    def exe(self) -> list[str]:
        """
        Example orchestrator:
        - (Optionally) format features
        - Compute ICC per feature class or globally
        - Plot and collect unstable features
        """
        iccs = {}
        total_ICC = pd.DataFrame()
        perform_per_featureclass_only = False

        self.logger.info(
            "Feature Stability Filter config:\n"
            f"  ICC threshold: {self.ICC_threshold}\n"
            f"  Extractor: {self.extractor}\n"
        )

        if self.df_data.empty:
            self.error.error("No data for ICC computation.")
            return []

        # IMPORTANT: Do NOT fill NaN globally here.
        # Expect a tidy/long frame: columns 'ID', 'Mask_Transformation', and feature columns.

        # Check if Feature Format if present and generate otherwise
        profile_path = os.path.join(self.out_path, f"{self.extractor}_General_Feature_Profile.csv")
        if not os.path.exists(profile_path):
            feature_formatter = FeatureFormatter(features=self.df_data.columns.to_list(),
                                             extractor=self.extractor,
                                             logger=self.logger,
                                             error=self.error,
                                             output_path=self.out_path)

            formatted_features = feature_formatter.exe(title="General_Feature_Profile")
            formatted_features.to_csv(self.out_path + "/" + self.extractor + "_General_Feature_Profile.csv", index=False)
        else:
            formatted_features = pd.read_csv(profile_path)

        if formatted_features.empty:
            self.error.error(f"Feature Formatting Failed for stability filtering! Check Profile: {profile_path}")
            print(f"ERROR: Feature Formatting Failed for stability filtering! Check Profile: {profile_path}")

        if len(self.df_data) > 0:
            self.df_data.fillna(0, inplace=True)
        else:
            self.error.error(f"ICC Calculation Failed! No unstable Feature detection ...")
            print(f"ICC Calculation Failed! No unstable Feature detection ...")
            return pd.DataFrame()

        if "Image_Kernel" in formatted_features.columns:
            for kernel in formatted_features["Image_Kernel"].unique():
                kernel_features = formatted_features.loc[formatted_features["Image_Kernel"] == kernel]
                pbar = tqdm(formatted_features["Feature_Class"].unique(), desc=str("Calculate ICC for " + kernel + " Features"))

                for feature_class in pbar:
                    pbar.set_description(str("Calculate ICC for " + kernel + " " + feature_class + " Features"))
                    if feature_class in kernel_features["Feature_Class"].values:
                        selected_features = list(kernel_features.loc[kernel_features["Feature_Class"] == feature_class]["Name"].values)
                        selected_features.append("Mask_Transformation")
                        selected_features.append("ID")

                        # clean feature names
                        self.df_data.columns = [s.replace("_zscore", "") for s in self.df_data.copy().columns.to_list()]
                        
                        included_selected_features = []
                        for feature in selected_features:
                            if feature in self.df_data.columns.to_list():
                                included_selected_features.append(feature)
                        if len(included_selected_features) >0:
                            ICC = self.calculate_ICC(df=self.df_data[included_selected_features],
                                                    feature_class=feature_class)
                            ICC.to_csv(f"{kernel}_{feature_class}_ICC.csv")
                        else:
                            ICC = pd.DataFrame()
                            self.icc_error = f"Feature Class {feature_class} is not present in feature space!"
                        if ICC.empty:
                            print("ICC calculation failed for feature class " + feature_class + ": " +  self.icc_error + " !")
                            self.error.warning("ICC calculation failed for feature class " + feature_class + ": " +  self.icc_error + " !")
                            continue

                        ICC.sort_values(by=['CI0'], inplace=True)

                        # plotting ICC
                        self.generate_plot(feature_class=feature_class + "_" + kernel, df=ICC, kernel=kernel, threshold=self.ICC_threshold)

                        iccs[feature_class + "_" + kernel] = ICC
                        total_ICC = pd.concat([total_ICC,ICC])

                    else:
                        self.logger.warning(
                            "Feature class " + feature_class + " is not present in data for kernel " + kernel)

        # if not Kernels are included summirze ICC per feature class
        if len(total_ICC) == 0:
            perform_per_featureclass_only = True

        # In addition make plot to summarize the feature classes in one plot
        pbar = tqdm(formatted_features["Feature_Class"].unique(), desc=str("Calculate ICC for Features"))
        for feature_class in pbar:
            pbar.set_description(str("Calculate ICC for all " + feature_class + " Features"))
            selected_features = list(formatted_features.loc[formatted_features["Feature_Class"] == feature_class]["Name"].values)
            selected_features.append("Mask_Transformation")
            selected_features.append("ID")

            # print(selected_features)
            # clean feature names
            self.df_data.columns = [s.replace("_zscore", "") for s in self.df_data.copy().columns.to_list()]

            included_selected_features = []
            for feature in selected_features:
                if feature in self.df_data.columns.to_list():
                    included_selected_features.append(feature)

            if len(included_selected_features) >0:
                ICC = self.calculate_ICC(df=self.df_data[included_selected_features],
                                        feature_class=feature_class)
            # ICC.to_csv("ICC.csv")
            if ICC.empty:
                print("ICC calculation failed for feature class " + feature_class + " !")
                self.logger.warning("ICC calculation failed for feature class " + feature_class + " !")
                continue

            ICC.sort_values(by=['CI0'], inplace=True)

            # plotting ICC
            self.generate_plot(feature_class=feature_class, df=ICC, kernel="", threshold=self.ICC_threshold)
            
            if perform_per_featureclass_only:
                iccs[feature_class] = ICC

                total_ICC = pd.concat([total_ICC,ICC])

        # For demonstration we treat the whole set as a single class:
        #iccs = {}
        #ICC = self.calculate_ICC(df=self.df_data, feature_class="ALL")
        #if ICC.empty:
        #    self.error.warning(f"ICC calculation produced no rows. Last error: {self.icc_error}")
        #    print(f"ICC calculation produced no rows. Last error: {self.icc_error}")
        #    return []

        # Sort for plotting (optional: by CI0 for conservative ranking)
        #ICC = ICC.sort_values(by="CI0", ascending=True)
        #try:
        #    self.plot_icc_by_profile(ICC.copy()) # self.plot_icc_per_feature_class(ICC.copy())
        #except Exception as ex:
        #    print(f"Could not plot ICC distribution plot: {ex}")
        #    self.error.warning(f"Could not plot ICC distribution plot: {ex}")


        #try:
        #    self.generate_plot(feature_class="ALL", df=ICC, kernel="", threshold=self.ICC_threshold)
        #except Exception as ex:
        #    print(f"Could not plot ICC distribution plot: {ex}")
        #    self.error.warning(f"Could not plot ICC distribution plot: {ex}")

        # Save a flat CSV with all ICCs
        os.makedirs(self.out_path, exist_ok=True)
        csv_path = os.path.join(self.out_path, "ICC.csv")
        total_ICC.to_csv(csv_path, index=False)

        # Filter unstable
        unstable_df = self.filter_for_unstable_feature_class(ICC)
        unstable_features = unstable_df["Feature"].tolist()

        self.logger.info(f"Detected unstable features: {len(unstable_features)}")
        pd.DataFrame({"unstable_features": unstable_features}).to_csv(
            os.path.join(self.out_path, f"unstable_features_threshold_{self.ICC_threshold}_{self.RunID}.csv"), index=False
        )
        return unstable_features
