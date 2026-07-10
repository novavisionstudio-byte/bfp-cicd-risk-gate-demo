"""Live pre-build feature extraction from a real git repository + the GitHub Actions
REST API, mapped onto the exact raw-feature contract the saved model expects
(models/preprocessor.joblib -> feature_order_, cat_maps_).

This is the "how a TravisTorrent-trained model gets fed live data" piece of Chapter 4,
section 4-6-3. Every field below is commented with which raw feature it fills and how.
Nothing here is invented -- if a signal genuinely isn't available cheaply for a small
demo repo (e.g. per-line PR-comment counts), it's documented as an honest approximation,
not silently guessed.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

SRC_EXT = {".py"}
DOC_EXT = {".md", ".rst", ".txt"}


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def _is_test_path(path: str) -> bool:
    return "test" in path.replace("\\", "/").split("/")[-1] or path.startswith("tests/")


def _category(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in DOC_EXT:
        return "doc"
    if ext in SRC_EXT:
        return "src"
    return "other"


PRODUCT_PATHS = ["app", "tests"]  # scope diff stats to the product code only, excluding
                                   # vendored ci_gate/bfp/models/.github infra so those
                                   # one-time additions don't inflate "application change" risk


def git_diff_stats(prev_sha: str, curr_sha: str) -> dict:
    """gh_diff_* / git_diff_* : numstat + name-status between two commits, scoped to
    PRODUCT_PATHS only (the vendored model/ci_gate infra is deliberately excluded)."""
    numstat = _run(["git", "diff", "--numstat", prev_sha, curr_sha, "--", *PRODUCT_PATHS])
    namestatus = _run(["git", "diff", "--name-status", prev_sha, curr_sha, "--", *PRODUCT_PATHS])

    src_churn = test_churn = 0
    files_added = files_deleted = files_modified = 0
    tests_added = tests_deleted = 0
    src_files = doc_files = other_files = 0

    status_by_path = {}
    for line in namestatus.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status, path = parts[0], parts[-1]
        status_by_path[path] = status[0]  # A / M / D (ignore rename detail)

    for line in numstat.splitlines():
        if not line.strip():
            continue
        added, deleted, path = line.split("\t")
        added = 0 if added == "-" else int(added)
        deleted = 0 if deleted == "-" else int(deleted)
        is_test = _is_test_path(path)
        if is_test:
            test_churn += added + deleted
        else:
            src_churn += added + deleted

        status = status_by_path.get(path, "M")
        if status == "A":
            files_added += 1
            if is_test:
                tests_added += 1
        elif status == "D":
            files_deleted += 1
            if is_test:
                tests_deleted += 1
        else:
            files_modified += 1

        cat = _category(path)
        if cat == "src":
            src_files += 1
        elif cat == "doc":
            doc_files += 1
        else:
            other_files += 1

    return {
        "git_diff_src_churn": src_churn,
        "git_diff_test_churn": test_churn,
        "gh_diff_files_added": files_added,
        "gh_diff_files_deleted": files_deleted,
        "gh_diff_files_modified": files_modified,
        "gh_diff_tests_added": tests_added,
        "gh_diff_tests_deleted": tests_deleted,
        "gh_diff_src_files": src_files,
        "gh_diff_doc_files": doc_files,
        "gh_diff_other_files": other_files,
    }


def repo_level_stats(changed_files: list[str]) -> dict:
    """gh_team_size, gh_sloc, test density, repo age/size -- computed at HEAD."""
    authors = _run(["git", "log", "--format=%ae"]).splitlines()
    team_size = len(set(authors))

    first_ts = int(_run(["git", "log", "--reverse", "--format=%ct"]).splitlines()[0])
    repo_age_days = (datetime.now(timezone.utc).timestamp() - first_ts) / 86400.0

    n_commits = int(_run(["git", "rev-list", "--count", "HEAD"]))

    sloc = 0
    for root, _, files in os.walk("app"):
        for fn in files:
            if fn.endswith(".py"):
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    sloc += sum(1 for line in f if line.strip())
    sloc = max(sloc, 1)

    test_lines = test_cases = test_asserts = 0
    for root, _, files in os.walk("tests"):
        for fn in files:
            if fn.endswith(".py"):
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        test_lines += 1
                        if s.startswith("def test_"):
                            test_cases += 1
                        if "assert" in s:
                            test_asserts += 1

    commits_on_touched = 0
    if changed_files:
        out = _run(["git", "log", "--oneline", "--", *changed_files])
        commits_on_touched = len(out.splitlines()) if out else 0

    return {
        "gh_team_size": team_size,
        "gh_repo_age": repo_age_days,
        "gh_repo_num_commits": n_commits,
        "gh_sloc": sloc,
        "gh_test_lines_per_kloc": test_lines / sloc * 1000.0,
        "gh_test_cases_per_kloc": test_cases / sloc * 1000.0,
        "gh_asserts_cases_per_kloc": test_asserts / sloc * 1000.0,
        "gh_num_commits_on_files_touched": commits_on_touched,
    }


def _gh_api(path: str) -> dict | list:
    """Unauthenticated call to the public GitHub REST API (works with no token for
    public repos, subject to the standard unauthenticated rate limit)."""
    repo = os.environ["GITHUB_REPOSITORY"]  # "<owner>/<name>", set by Actions automatically
    url = f"https://api.github.com/repos/{repo}{path}"
    token = os.environ.get("GITHUB_TOKEN", "")
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"}
                                  | ({"Authorization": f"Bearer {token}"} if token else {}))
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def history_features(workflow_file: str) -> dict:
    """hist_* : derived from this repo's OWN past GitHub Actions run conclusions,
    strictly prior runs only (the current, in-progress run is excluded by construction
    -- it has no conclusion yet when this script executes). Mirrors bfp/data.py's
    add_history() semantics EXACTLY, computed against live API data instead of a
    static CSV:
      - zero prior runs -> hist_prev_status / hist_fail_rate_* are None (JSON null ->
        NaN -> the saved preprocessor median-imputes, identical to offline shift(1)
        NaN on a project's first build);
      - hist_consec_fail / hist_build_seq are 0.0 on the first build (offline uses
        fillna(0) / cumcount, NOT NaN, for these two);
      - >=1 prior run: rolling windows use min_periods=1 offline, i.e. the mean over
        however many priors exist (up to the window size) -- same as computed here.
    The run-history fetch is PAGINATED so hist_fail_rate_all reflects the full
    history, not just the most recent API page."""
    items = []
    try:
        page = 1
        while True:
            runs = _gh_api(f"/actions/workflows/{workflow_file}/runs"
                           f"?per_page=100&status=completed&page={page}")
            batch = runs.get("workflow_runs", [])
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        items.sort(key=lambda r: r["run_started_at"])
    except Exception as e:
        print(f"WARNING: could not fetch run history ({e}); treating as first build", file=sys.stderr)
        items = []

    outcomes = [0 if r["conclusion"] == "success" else 1 for r in items]

    def fail_rate(window):
        # offline: shift(1).rolling(window, min_periods=1).mean() / expanding().mean()
        # -> None (NaN -> median-impute) when there are no priors at all
        if not outcomes:
            return None
        seq = outcomes if window is None else outcomes[-window:]
        return sum(seq) / len(seq)

    consec = 0
    for o in reversed(outcomes):
        if o == 1:
            consec += 1
        else:
            break

    return {
        "hist_prev_status": float(outcomes[-1]) if outcomes else None,
        "hist_fail_rate_5": fail_rate(5),
        "hist_fail_rate_20": fail_rate(20),
        "hist_fail_rate_all": fail_rate(None),
        "hist_consec_fail": float(consec),   # offline fillna(0): 0.0 on first build
        "hist_build_seq": float(len(outcomes)),  # offline cumcount: 0.0 on first build
        "_n_prior_runs": len(outcomes),  # for logging only, not a model feature
    }


def categorical_features(is_pr: bool, has_prior_runs: bool) -> dict:
    """gh_lang / gh_is_pr / gh_by_core_team_member / git_prev_commit_resolution_status
    -- values must match the STRING categories the saved encoder was fit on
    (see models/preprocessor.joblib -> cat_maps_): gh_lang in {go,java,python,ruby},
    gh_is_pr/gh_by_core_team_member in {"TRUE","FALSE"}, git_prev_commit_resolution_status
    in {"build_found","merge_found","no_previous_build"}. This demo app is Python, has
    a single maintainer (always "core"), and approximates TravisTorrent's
    "prev commit resolution" concept as "no_previous_build" only for the very first
    tracked run -- an honest approximation, documented in Chapter 4 section 4-6-3."""
    return {
        "gh_lang": "python",
        "gh_is_pr": "TRUE" if is_pr else "FALSE",
        "gh_by_core_team_member": "TRUE",
        "git_prev_commit_resolution_status": "build_found" if has_prior_runs else "no_previous_build",
    }


def extract(prev_sha: str, curr_sha: str, workflow_file: str, is_pr: bool) -> dict:
    diff = git_diff_stats(prev_sha, curr_sha)
    changed_files = [l.split("\t")[-1] for l in
                     _run(["git", "diff", "--name-only", prev_sha, curr_sha,
                           "--", *PRODUCT_PATHS]).splitlines() if l]
    repo_stats = repo_level_stats(changed_files)
    hist = history_features(workflow_file)
    cats = categorical_features(is_pr, has_prior_runs=hist["_n_prior_runs"] > 0)

    raw = {}
    raw.update(diff)
    raw.update(repo_stats)
    raw.update({k: v for k, v in hist.items() if not k.startswith("_")})
    raw.update(cats)
    # feature the model expects but that isn't cheaply observable for a tiny demo repo;
    # documented approximation (see Chapter 4, section 4-6-3 mapping table)
    raw["gh_num_commit_comments"] = 0
    raw["git_num_all_built_commits"] = int(_run(["git", "rev-list", f"{prev_sha}..{curr_sha}", "--count"]) or 1)

    return {"raw_features": raw, "history_debug": hist}


if __name__ == "__main__":
    prev_sha, curr_sha, workflow_file = sys.argv[1], sys.argv[2], sys.argv[3]
    is_pr = os.environ.get("GITHUB_EVENT_NAME", "") == "pull_request"
    result = extract(prev_sha, curr_sha, workflow_file, is_pr)
    print(json.dumps(result, indent=2))
    with open("risk_gate_features.json", "w") as f:
        json.dump(result, f, indent=2)
