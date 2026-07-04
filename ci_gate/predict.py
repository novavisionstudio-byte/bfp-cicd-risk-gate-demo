"""Chapter 4, section 4-6-1/4-6-4: the processing + explainability layers, run live.

Loads the SAVED model artifacts (rf_model.joblib, calibrator.joblib,
preprocessor.joblib, thresholds.json -- vendored from the main thesis repo's
models/, byte-identical modulo joblib compression) via bfp.inference.InferencePipeline
(same class used offline), scores the live-extracted raw features, and emits:
  - risk_gate_result.json (the report_payload contract from LLM_PROMPT.md)
  - a GitHub Actions step summary (human-readable, shows up in the run UI)
  - a GITHUB_OUTPUT `decision` value for the workflow to branch on
  - process exit code: 1 if decision == ROLLBACK (fails this job by design), else 0
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bfp.inference import InferencePipeline  # noqa: E402


def main():
    features_path = sys.argv[1] if len(sys.argv) > 1 else "risk_gate_features.json"
    with open(features_path) as f:
        payload_in = json.load(f)
    raw = payload_in["raw_features"]

    pipe = InferencePipeline(models_dir=os.path.join(ROOT, "models"), with_shap=True)
    result = pipe.predict(raw, top_k=6)[0]

    with open("risk_gate_result.json", "w") as f:
        json.dump(result, f, indent=2)

    summary_lines = [
        "## CI/CD Risk Gate result",
        "",
        f"**Decision: `{result['decision']}`**",
        "",
        f"- Calibrated failure probability: `{result['failure_probability']:.4f}`",
        f"- Thresholds: tau1 (PASS/WARN)=`{result['thresholds']['tau1']:.4f}`, "
        f"tau2 (WARN/ROLLBACK)=`{result['thresholds']['tau2']:.4f}`",
        "",
        "| Feature | Value | SHAP |",
        "|---|---|---|",
    ]
    for feat in result["top_features"]:
        shap_v = feat.get("shap")
        shap_s = f"{shap_v:+.4f}" if shap_v is not None else "n/a"
        summary_lines.append(f"| {feat['feature']} | {feat['value']:.4g} | {shap_s} |")

    summary = "\n".join(summary_lines)
    print(summary)

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write(summary + "\n")

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"decision={result['decision']}\n")
            f.write(f"probability={result['failure_probability']:.6f}\n")

    if result["decision"] == "ROLLBACK":
        print("::error::Risk gate decision is ROLLBACK -- failing this job by design "
              "(deployment stopped before the test suite runs).")
        sys.exit(1)
    if result["decision"] == "WARN":
        print("::warning::Risk gate decision is WARN -- proceeding, but flagged for review.")
    sys.exit(0)


if __name__ == "__main__":
    main()
