"""Online inference path.

Feature extraction -> identical preprocessing using the SAVED encoders/medians ->
Random Forest -> Platt calibration -> three-state decision -> TreeSHAP attribution
-> report payload. The SAME preprocessor / model / calibrator / thresholds produced
offline are loaded here (verification test #4).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import joblib
from . import config as C
from .preprocess import Preprocessor


class InferencePipeline:
    def __init__(self, models_dir: str = C.MODELS_DIR, with_shap: bool = True):
        self.pre = Preprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.model = joblib.load(os.path.join(models_dir, "rf_model.joblib"))
        self.calibrator = joblib.load(os.path.join(models_dir, "calibrator.joblib"))
        with open(os.path.join(models_dir, "thresholds.json")) as f:
            th = json.load(f)
        self.tau1, self.tau2 = th["tau1"], th["tau2"]
        self.feature_order = self.pre.feature_order_
        self.explainer = None
        if with_shap:
            self._init_shap(models_dir)

    def _init_shap(self, models_dir):
        try:
            import shap
            bg = np.load(os.path.join(models_dir, "shap_background.npy"))
            bg_df = pd.DataFrame(bg, columns=self.feature_order)
            self.explainer = shap.TreeExplainer(
                self.model, data=bg_df, feature_perturbation="interventional",
                model_output="raw")
        except Exception as e:                     # SHAP is optional at inference time
            self.explainer = None
            self._shap_error = str(e)

    def _to_frame(self, raw) -> pd.DataFrame:
        if isinstance(raw, dict):
            raw = {k: [v] for k, v in raw.items()}
            return pd.DataFrame(raw)
        return raw.copy()

    def predict(self, raw, top_k: int = 6) -> list[dict]:
        """`raw` = dict (one build) or DataFrame of RAW pre-build feature columns.
        Returns one payload per row."""
        df = self._to_frame(raw)
        X = self.pre.transform(df)                 # saved encoders + medians
        rf_p = self.model.predict_proba(X)[:, 1]
        cal_p = self.calibrator.transform(rf_p)
        decisions = np.where(cal_p < self.tau1, "PASS",
                             np.where(cal_p < self.tau2, "WARN", "ROLLBACK"))
        shap_vals = self._shap(X)
        out = []
        for i in range(len(X)):
            top = self._top_features(X.iloc[i], shap_vals[i] if shap_vals is not None else None, top_k)
            out.append({
                "failure_probability": float(cal_p[i]),
                "rf_probability": float(rf_p[i]),
                "decision": str(decisions[i]),
                "thresholds": {"tau1": self.tau1, "tau2": self.tau2},
                "top_features": top,
            })
        return out

    def _shap(self, X):
        if self.explainer is None:
            return None
        sv = self.explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    def _top_features(self, x_row, sv_row, top_k):
        if sv_row is None:
            return [{"feature": f, "value": float(x_row[f])} for f in self.feature_order[:top_k]]
        order = np.argsort(np.abs(sv_row))[::-1][:top_k]
        return [{"feature": self.feature_order[j], "value": float(x_row.iloc[j]),
                 "shap": float(sv_row[j])} for j in order]
