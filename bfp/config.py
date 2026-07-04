"""Central configuration: paths, column lists, leakage drop list, and all
named constants. Everything that affects scientific validity lives here so it is
auditable and reproducible. Edit here, not scattered through the code.
"""
from __future__ import annotations
import os

# ----------------------------------------------------------------------------- paths
# MODELS_DIR / ARTIFACTS_DIR are overridable via env vars so ablation re-runs (Chapter 4,
# section 4-5) can write to a separate location instead of clobbering the main run's
# artifacts/models. Defaults are unchanged from before -> normal runs are unaffected.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "final-2017.csv")
MODELS_DIR = os.environ.get("BFP_MODELS_DIR", os.path.join(ROOT, "models"))
ARTIFACTS_DIR = os.environ.get("BFP_ARTIFACTS_DIR", os.path.join(ROOT, "artifacts"))

# ----------------------------------------------------------------------------- reproducibility
SEED = 42

# ----------------------------------------------------------------------------- target
TARGET_RAW = "tr_status"
POSITIVE_IS_FAILURE = True          # y=1 means FAILURE
PASS_LABEL = "passed"               # the only label mapped to y=0
DROP_STATUS = {"started"}           # in-progress builds: outcome unknown -> drop

# ----------------------------------------------------------------------------- keys (NOT features)
KEY_BUILD = "tr_build_id"           # dedup key (one row per build)
KEY_GROUP = "gh_project_name"       # grouping key for the project-grouped split

# ----------------------------------------------------------------------------- features (X) — all pre-build / known at trigger time
FEATURES_NUMERIC = [
    "gh_team_size", "git_num_all_built_commits", "gh_num_commit_comments",
    "git_diff_src_churn", "git_diff_test_churn", "gh_diff_files_added",
    "gh_diff_files_deleted", "gh_diff_files_modified", "gh_diff_tests_added",
    "gh_diff_tests_deleted", "gh_diff_src_files", "gh_diff_doc_files",
    "gh_diff_other_files", "gh_num_commits_on_files_touched", "gh_sloc",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "gh_repo_age", "gh_repo_num_commits",
]
FEATURES_CATEGORICAL = [
    "gh_lang", "gh_is_pr", "gh_by_core_team_member", "git_prev_commit_resolution_status",
]
# engineered (built in preprocessing from raw numerics)
FEATURES_ENGINEERED = ["churn_ratio", "test_coverage_proxy"]

# historical features: per-project statistics over a build's OWN PRIOR builds only
# (computed with a shift so the current outcome is never included). These are known
# at trigger time (the project's earlier build outcomes) -> legitimate, NOT leakage.
# This is the transferable signal ("recently-failing projects keep failing") that
# generalizes across projects, unlike raw project identity.
# Overridable via env var for the ablation re-run (Chapter 4, section 4-5: diff-only
# vs diff+history). Default True is unchanged from before.
USE_HISTORY = os.environ.get("BFP_USE_HISTORY", "1") != "0"
FEATURES_HISTORICAL = [
    "hist_prev_status",     # outcome of the immediately previous build (0/1)
    "hist_fail_rate_5",     # mean failure over previous 5 builds
    "hist_fail_rate_20",    # mean failure over previous 20 builds
    "hist_fail_rate_all",   # expanding mean failure over all previous builds
    "hist_consec_fail",     # length of the trailing run of consecutive prior failures
    "hist_build_seq",       # number of prior builds of this project (experience)
]
# columns used ONLY to order builds chronologically within a project (never features)
ORDER_COLS = ["tr_build_number"]

# all numeric columns that get median-imputed (raw + historical)
NUMERIC_COLS = FEATURES_NUMERIC + (FEATURES_HISTORICAL if USE_HISTORY else [])

# final model feature order (numeric raw + historical + engineered + categorical-encoded)
FEATURE_ORDER = NUMERIC_COLS + FEATURES_ENGINEERED + FEATURES_CATEGORICAL

# columns to actually read from the CSV (features + keys + target + order-only cols;
# skip the giant hash-list columns entirely for speed/memory)
USECOLS = ([KEY_BUILD, KEY_GROUP, TARGET_RAW] + FEATURES_NUMERIC + FEATURES_CATEGORICAL
           + (ORDER_COLS if USE_HISTORY else []))

# ----------------------------------------------------------------------------- LEAKAGE DROP LIST (floor + extensions; the verification tests assert none reach X)
# (a) post-outcome: known only during/after the build
LEAKAGE_POST_OUTCOME = [
    "tr_status", "tr_duration", "tr_log_status", "tr_log_setup_time",
    "tr_log_buildduration", "tr_log_testduration", "tr_log_num_tests_ok",
    "tr_log_num_tests_failed", "tr_log_num_tests_run", "tr_log_num_tests_skipped",
    "tr_log_num_test_suites_run", "tr_log_num_test_suites_ok",
    "tr_log_num_test_suites_failed", "tr_log_tests_failed", "tr_log_bool_tests_ran",
    "tr_log_bool_tests_failed", "tr_log_lan", "tr_log_analyzer", "tr_log_frameworks",
    "tr_jobs", "tr_prev_build", "tr_virtual_merged_into",
]
# (b) identifiers / hashes / timestamps
LEAKAGE_IDS_TIME = [
    "tr_build_id", "gh_project_name", "tr_job_id", "tr_build_number",
    "git_trigger_commit", "git_merged_with", "git_prev_built_commit",
    "tr_original_commit", "git_all_built_commits", "gh_commits_in_push",
    "git_branch", "gh_pull_req_num", "gh_pr_created_at", "gh_first_commit_created_at",
    "gh_pushed_at", "gh_build_started_at",
]
# (c) dropped for high missingness / no univariate signal
DROPPED_MISSINGNESS = [
    "gh_num_commits_in_push", "gh_num_issue_comments", "gh_num_pr_comments",
    "gh_description_complexity",
]
LEAKAGE_DROP = LEAKAGE_POST_OUTCOME + LEAKAGE_IDS_TIME + DROPPED_MISSINGNESS

# ----------------------------------------------------------------------------- split (grouped by project)
SPLIT_TEST_FRAC = 0.15
SPLIT_VAL_FRAC = 0.15
CALIB_FRAC_OF_TRAIN = 0.15           # carved (grouped) out of TRAIN, for Platt scaling only

# ----------------------------------------------------------------------------- model + tuning
RF_FIXED = dict(criterion="gini", class_weight="balanced", random_state=SEED, n_jobs=-1)
BETA = 2                              # F-beta cost ratio (missed failure vs false alarm)
GRID_SUBSAMPLE = 80_000              # stratified subsample size for GridSearch
CV_FOLDS = 5
# Full search: 2 x 2 x 3 x 2 = 24 candidates x 5 folds = 120 fits. Grid search runs on
# GRID_SUBSAMPLE; the final model is refit on the FULL training set.
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 16],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt", 0.4],
}

# ----------------------------------------------------------------------------- threshold policy (selected on VALIDATION only)
R_STAR = 0.80                        # tau1: largest tau with Recall >= R_STAR (PASS/WARN)
P_STAR = 0.70                        # tau2: smallest tau>tau1 with Precision >= P_STAR (WARN/ROLLBACK)
THRESH_GRID_STEPS = 1001             # 0..1 sweep resolution

# ----------------------------------------------------------------------------- explainability
SHAP_BACKGROUND = 100                # background sampled from passing builds
SHAP_EXPLAIN_N = 500                 # test rows to attribute

# ----------------------------------------------------------------------------- leakage alarm thresholds (verification test #2)
ALARM_TEST_ROC_AUC = 0.99            # fail if test ROC-AUC >= this
ALARM_FEATURE_IMPORTANCE = 0.50      # fail if any single feature importance >= this

DECISION_LABELS = ["PASS", "WARN", "ROLLBACK"]


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
