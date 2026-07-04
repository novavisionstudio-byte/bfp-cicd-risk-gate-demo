"""Preprocessing: feature engineering, label encoding, and median imputation.

CRITICAL: `fit` is called on TRAIN ONLY. The learned label maps and medians are
saved and reused unchanged at validation/test/inference time. Unseen categories
map to -1; missing numerics fill with the saved train median.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from . import config as C

MISSING_CAT = "__MISSING__"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two derived features. Pure function of raw numerics (no fitting)."""
    out = df.copy()
    out["churn_ratio"] = df["git_diff_test_churn"] / (df["git_diff_src_churn"] + 1.0)
    out["test_coverage_proxy"] = df["gh_test_lines_per_kloc"] * df["gh_sloc"] / 1000.0
    return out


class Preprocessor:
    """Stateful, fit-on-train preprocessor. Serializable with joblib."""

    def __init__(self):
        self.cat_maps_: dict[str, dict] = {}
        self.medians_: dict[str, float] = {}
        self.feature_order_ = list(C.FEATURE_ORDER)
        self.fitted_ = False

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        eng = engineer(df_train)
        # label maps from TRAIN categories (missing -> its own category)
        for col in C.FEATURES_CATEGORICAL:
            vals = eng[col].astype("object").where(eng[col].notna(), MISSING_CAT)
            classes = sorted(pd.unique(vals).tolist(), key=lambda x: str(x))
            self.cat_maps_[col] = {c: i for i, c in enumerate(classes)}
        # medians from TRAIN (numeric raw + engineered), computed AFTER engineering
        for col in C.NUMERIC_COLS + C.FEATURES_ENGINEERED:
            med = float(np.nanmedian(eng[col].to_numpy(dtype="float64")))
            self.medians_[col] = med if np.isfinite(med) else 0.0
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "Preprocessor must be fit before transform"
        eng = engineer(df)
        out = pd.DataFrame(index=eng.index)
        # numeric + engineered: impute with saved train medians
        for col in C.NUMERIC_COLS + C.FEATURES_ENGINEERED:
            s = pd.to_numeric(eng[col], errors="coerce")
            # tame infinities from ratios before imputation
            s = s.replace([np.inf, -np.inf], np.nan)
            out[col] = s.fillna(self.medians_[col]).astype("float32")
        # categorical: label-encode with saved maps (unseen/missing -> -1)
        for col in C.FEATURES_CATEGORICAL:
            vals = eng[col].astype("object").where(eng[col].notna(), MISSING_CAT)
            m = self.cat_maps_[col]
            out[col] = vals.map(lambda v: m.get(v, -1)).astype("int32")
        return out[self.feature_order_]

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        return joblib.load(path)
