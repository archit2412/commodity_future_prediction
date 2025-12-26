"""Research agent: analyze model outputs and generate research-friendly summaries.

This module implements a small, auditable ResearchAgent that can:
- Load evaluation metrics and prediction outputs
- Compute aggregated metrics (MAE, RMSE, R2, etc.)
- Detect simple failure modes (high-error groups, low-sample slices, heteroskedasticity)
- Produce a JSON summary and a short Markdown report with bullet points

The implementation is intentionally small and dependency-light so it can be extended later.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ResearchReport:
    summary_bullets: List[str]
    key_metrics: Dict[str, Any]
    failure_modes: List[str]
    suggested_next_steps: List[str]
    artifacts: List[str]


class ResearchAgent:
    """A small utility to synthesize research notes from model artifacts.

    Basic flow:
    - load metrics json (if present) or compute metrics from predictions CSV
    - compute group-level metrics for categorical columns and find slices with high errors or low sample sizes
    - create short bullet points describing insights and next steps
    """

    def __init__(self, model_path: Optional[str] = None, metrics_path: Optional[str] = None,
                 predictions_path: Optional[str] = None, out_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else None
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.predictions_path = Path(predictions_path) if predictions_path else None
        self.out_path = Path(out_path) if out_path else Path("research_report.json")
        self.artifacts: List[str] = []

    def run(self) -> ResearchReport:
        metrics = self._load_metrics()
        preds = self._load_predictions()

        if preds is not None and ("y_true" in preds.columns and "y_pred" in preds.columns):
            computed = self._compute_metrics_from_preds(preds)
            # merge computed into metrics (computed wins when available)
            metrics.update(computed)
        elif not metrics:
            raise ValueError("No metrics or predictions with `y_true`/`y_pred` provided.")

        failure_modes = self._detect_failure_modes(preds) if preds is not None else []
        suggested = self._suggest_next_steps(metrics, failure_modes, preds)
        bullets = self._generate_bullets(metrics, failure_modes, suggested, preds)

        report = ResearchReport(
            summary_bullets=bullets,
            key_metrics=self._summarize_metrics(metrics),
            failure_modes=failure_modes,
            suggested_next_steps=suggested,
            artifacts=self.artifacts,
        )

        self._write_output(report)
        return report

    def _load_metrics(self) -> Dict[str, Any]:
        if self.metrics_path and self.metrics_path.exists():
            try:
                return json.loads(self.metrics_path.read_text())
            except Exception:
                return {}
        return {}

    def _load_predictions(self) -> Optional[pd.DataFrame]:
        if not self.predictions_path:
            return None
        if not self.predictions_path.exists():
            return None
        try:
            df = pd.read_csv(self.predictions_path)
            return df
        except Exception:
            return None

    def _compute_metrics_from_preds(self, df: pd.DataFrame) -> Dict[str, float]:
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        residuals = y_true - y_pred
        # simple heteroskedasticity signal: compare std of residuals in top vs bottom quartiles
        q = np.percentile(y_pred, [25, 75])
        low_std = float(np.std(residuals[y_pred <= q[0]])) if np.any(y_pred <= q[0]) else float(np.std(residuals))
        high_std = float(np.std(residuals[y_pred >= q[1]])) if np.any(y_pred >= q[1]) else float(np.std(residuals))
        hetero_ratio = high_std / (low_std + 1e-12)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "hetero_ratio": hetero_ratio,
        }

    def _detect_failure_modes(self, df: Optional[pd.DataFrame]) -> List[str]:
        if df is None:
            return []
        failures: List[str] = []
        # if there are categorical columns, compute MAE per category and flag worse-than-mean groups
        cat_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name == "category"]
        if not cat_cols:
            # also consider low-cardinality numeric columns
            for c in df.columns:
                if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 20:
                    cat_cols.append(c)
        for col in cat_cols:
            grouped = df.groupby(col).agg(count=("y_true", "size"), mae=("y_true", lambda x: 0.0))
            # compute MAE per group
            maes = {}
            for name, g in df.groupby(col):
                maes[name] = mean_absolute_error(g["y_true"], g["y_pred"]) if len(g) > 0 else float("nan")
            # find groups with high MAE and/or low counts
            values = np.array([v for v in maes.values() if not np.isnan(v)])
            if len(values) == 0:
                continue
            mean_mae = float(np.mean(values))
            std_mae = float(np.std(values))
            for k, v in maes.items():
                if np.isnan(v):
                    continue
                # high error: > mean + 1 std
                if v > mean_mae + std_mae:
                    failures.append(f"High MAE in {col}={k}: MAE={v:.3f} (group size={int((df[col]==k).sum())})")
            # low sample size groups
            small = [k for k, g in df.groupby(col) if len(g) < max(5, int(0.05 * len(df)))]
            for k in small:
                failures.append(f"Low sample size for {col}={k} (n={len(df[df[col]==k])})")
        # heteroskedasticity check if present
        if "y_pred" in df.columns and "y_true" in df.columns:
            residuals = df["y_true"] - df["y_pred"]
            # compare variance in top and bottom deciles
            q_low = df["y_pred"].quantile(0.1)
            q_high = df["y_pred"].quantile(0.9)
            low_var = float(np.var(residuals[df["y_pred"] <= q_low])) if (df["y_pred"] <= q_low).any() else 0.0
            high_var = float(np.var(residuals[df["y_pred"] >= q_high])) if (df["y_pred"] >= q_high).any() else 0.0
            if low_var > 0 and high_var / (low_var + 1e-12) > 4:
                failures.append("Strong heteroskedasticity: residual variance much larger for high predictions")
        # deduplicate and return
        return list(dict.fromkeys(failures))

    def _summarize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # pick and format common metrics
        res = {}
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                res[k] = float(v)
            else:
                res[k] = v
        return res

    def _suggest_next_steps(self, metrics: Dict[str, Any], failure_modes: List[str], df: Optional[pd.DataFrame]) -> List[str]:
        steps: List[str] = []
        # suggest more data for low-sample slices
        for f in failure_modes:
            if f.startswith("Low sample size"):
                steps.append("Collect more data for the low-sample slices listed above (or use reweighting/oversampling).")
        # if heteroskedasticity, suggest transforms
        if any("heteroskedastic" in f.lower() for f in failure_modes):
            steps.append("Consider transforming the target (log, sqrt) or using heteroskedastic-aware losses.")
        # if MAE large, suggest calibration
        if "MAE" in metrics and metrics["MAE"] > 0.5:  # threshold heuristic
            steps.append("MAE is high; consider stronger regularization, feature engineering, or ensembling.")
        # generic
        if not steps:
            steps.append("Run targeted slice analysis, check feature importances, and validate on additional holdout folds.")
        return steps

    def _generate_bullets(self, metrics: Dict[str, Any], failure_modes: List[str], suggested: List[str], df: Optional[pd.DataFrame]) -> List[str]:
        bullets: List[str] = []
        # top-line metric
        if "MAE" in metrics:
            bullets.append(f"Model achieves MAE={metrics['MAE']:.3f} on provided data.")
        if "RMSE" in metrics:
            bullets.append(f"RMSE={metrics['RMSE']:.3f}, R2={metrics.get('R2', float('nan')):.3f}.")

        # hetero ratio
        if "hetero_ratio" in metrics and metrics["hetero_ratio"] > 1.5:
            bullets.append("Residual spread increases with predicted magnitude (heteroskedasticity).")

        # mention top failure modes
        for f in failure_modes[:5]:
            bullets.append(f)

        # suggestions
        if suggested:
            bullets.append("Suggested next steps: " + "; ".join(suggested[:3]))

        return bullets

    def _write_output(self, report: ResearchReport) -> None:
        data = asdict(report)
        self.out_path.write_text(json.dumps(data, indent=2))
        # also write a small markdown summary
        md_path = self.out_path.with_suffix(".md")
        md_lines = ["# Research Summary\n"]
        md_lines.append("## Key bullets\n")
        for b in report.summary_bullets:
            md_lines.append(f"- {b}\n")
        md_lines.append("\n## Suggested steps\n")
        for s in report.suggested_next_steps:
            md_lines.append(f"- {s}\n")
        md_text = "".join(md_lines)
        md_path.write_text(md_text)
        self.artifacts.extend([str(self.out_path), str(md_path)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", help="Path to metrics JSON file", default=None)
    parser.add_argument("--predictions", help="Path to predictions CSV with columns y_true,y_pred", default=None)
    parser.add_argument("--out", help="Output JSON path", default="research_report.json")
    args = parser.parse_args()

    agent = ResearchAgent(metrics_path=args.metrics, predictions_path=args.predictions, out_path=args.out)
    report = agent.run()
    print(json.dumps(asdict(report), indent=2))