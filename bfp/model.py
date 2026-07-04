"""Model training, probability calibration, threshold selection, explainability.

- RandomForest tuned by GridSearchCV (StratifiedGroupKFold, F-beta) on a stratified
  subsample; final model refit on the full training-fit set.
- Platt scaling (logistic regression, no class weight) fit on a held-out calibration
  subset carved from TRAIN.
- Thresholds selected by a 1-D sweep on VALIDATION only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve
from . import config as C


# --------------------------------------------------------------------------- tuning
def stratified_subsample(X, y, groups, n, seed=C.SEED):
    if n is None or n >= len(y):
        return X, y, groups
    rng = np.random.RandomState(seed)
    idx = []
    y = np.asarray(y)
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(round(n * len(cls_idx) / len(y))))
        idx.extend(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
    idx = np.sort(np.array(idx))
    return X.iloc[idx], y[idx], np.asarray(groups)[idx]


def tune_rf(X, y, groups, param_grid=None, subsample=C.GRID_SUBSAMPLE, verbose=1):
    """Grid search on a stratified subsample. Returns (best_params, summary)."""
    param_grid = param_grid or C.PARAM_GRID
    Xs, ys, gs = stratified_subsample(X, y, groups, subsample)
    cv = StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    scorer = make_scorer(fbeta_score, beta=C.BETA, pos_label=1, zero_division=0)
    base = RandomForestClassifier(**C.RF_FIXED)
    gscv = GridSearchCV(base, param_grid, scoring=scorer, cv=cv, n_jobs=-1,
                        refit=True, verbose=verbose)
    gscv.fit(Xs, ys, groups=gs)
    r = gscv.cv_results_
    candidates = sorted(
        [{"params": p, "mean_test_score": float(m), "std_test_score": float(s),
          "rank": int(rk)}
         for p, m, s, rk in zip(r["params"], r["mean_test_score"],
                                r["std_test_score"], r["rank_test_score"])],
        key=lambda d: d["rank"])
    summary = {
        "search_subsample_n": int(len(ys)),
        "scoring": f"fbeta(beta={C.BETA})",
        "cv": f"StratifiedGroupKFold({C.CV_FOLDS})",
        "best_params": gscv.best_params_,
        "best_cv_score": float(gscv.best_score_),
        "param_grid": param_grid,
        "n_candidates": len(candidates),
        "cv_results": candidates,           # full per-candidate table (thesis defensibility)
    }
    return gscv.best_params_, summary


def fit_rf(X, y, params):
    model = RandomForestClassifier(**C.RF_FIXED, **params)
    model.fit(X, y)
    return model


# --------------------------------------------------------------------------- Platt calibration
class PlattCalibrator:
    """Logistic recalibration of the RF positive-class probability (BCE loss, no
    class weighting). Near-unregularized (large C) to approximate pure Platt."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, class_weight=None, max_iter=1000)

    def fit(self, rf_proba, y):
        f = np.asarray(rf_proba).reshape(-1, 1)
        self.lr.fit(f, y)
        return self

    def transform(self, rf_proba):
        f = np.asarray(rf_proba).reshape(-1, 1)
        return self.lr.predict_proba(f)[:, 1]


def rf_pos_proba(model, X):
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- thresholds
def select_thresholds(y_val, probs_val, r_star=C.R_STAR, p_star=C.P_STAR):
    """tau1 = largest tau with Recall>=r_star; tau2 = smallest tau>tau1 with
    Precision>=p_star. Enforce tau1<tau2. Fallbacks recorded if a target is unmet."""
    y_val = np.asarray(y_val)
    probs_val = np.asarray(probs_val)
    precision, recall, thr = precision_recall_curve(y_val, probs_val)
    # align: precision/recall have len(thr)+1; drop the trailing (recall=0) point
    prec, rec = precision[:-1], recall[:-1]

    fallback = {"tau1": False, "tau2": False}

    # tau1: largest threshold whose recall >= r_star
    ok1 = np.where(rec >= r_star)[0]
    if len(ok1):
        tau1 = float(thr[ok1].max())
    else:
        tau1 = float(np.median(probs_val)); fallback["tau1"] = True

    # tau2: smallest threshold > tau1 whose precision >= p_star
    mask = (thr > tau1) & (prec >= p_star)
    ok2 = np.where(mask)[0]
    if len(ok2):
        tau2 = float(thr[ok2].min())
    else:
        tau2 = float(np.quantile(probs_val, 0.99)); fallback["tau2"] = True

    # enforce strict ordering
    if not (tau1 < tau2):
        tau2 = float(min(1.0, max(tau1 + 1e-6, np.quantile(probs_val, 0.99))))
        fallback["tau2"] = True

    info = {
        "tau1": tau1, "tau2": tau2, "r_star": r_star, "p_star": p_star,
        "fallback": fallback,
        "recall_at_tau1": float(_recall_at(y_val, probs_val, tau1)),
        "precision_at_tau2": float(_precision_at(y_val, probs_val, tau2)),
    }
    return tau1, tau2, info


def _recall_at(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp / (tp + fn) if (tp + fn) else 0.0


def _precision_at(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


def decide(probs, tau1, tau2):
    probs = np.asarray(probs)
    out = np.where(probs < tau1, "PASS", np.where(probs < tau2, "WARN", "ROLLBACK"))
    return out


# --------------------------------------------------------------------------- SHAP (TreeSHAP, interventional)
def treeshap_summary(model, X_background, X_explain, feature_names):
    import shap
    explainer = shap.TreeExplainer(
        model, data=X_background, feature_perturbation="interventional",
        model_output="raw",
    )
    sv = explainer.shap_values(X_explain, check_additivity=False)
    if isinstance(sv, list):                       # [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                               # (n, features, classes)
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    summary = [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
               for i in order]
    return summary, sv, explainer
