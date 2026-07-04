"""Build-Failure Prediction pipeline (thesis).

Two phases: offline training (`run_offline.py`) and online inference
(`bfp.inference`). Highest priority across the whole package: ZERO data leakage.
"""
__all__ = ["config", "data", "preprocess", "splits", "model", "metrics", "inference"]
