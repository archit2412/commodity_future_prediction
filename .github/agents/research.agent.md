---
description: "Research agent: synthesize model outputs into research-friendly bullet points and structured artifacts"
tools:
  - name: python
    description: "Runs model introspection and metrics extraction scripts"
  - name: filesystem
    description: "Reads model artifacts, logs, and sample data files"
---

# Research Agent

## Purpose
This agent ingests ML model artifacts and evaluation outputs and produces concise, structured research artifacts suitable for inclusion in reports, PRs, or lab notebooks. It extracts key metrics, failure modes, dataset characteristics, and concise bullet-point summaries (insights + recommended next steps).

## When to use
- After training or evaluating a model to automatically create a brief research summary
- When generating release notes or experiment comparisons
- When curating evidence for model cards or reproducibility checklists

## Inputs
- Path to model artifacts (pickled models, saved checkpoints, weights)
- Evaluation outputs (CSV/JSON metrics, predictions, confusion matrices)
- Optional: sample input/target data for qualitative inspection
- Optional: config or hyperparameter YAML/JSON

## Outputs
- A JSON object containing:
  - `summary_bullets`: array of short, actionable bullet strings (insights + impact)
  - `key_metrics`: dict of main metrics (MAE, SHARPE, accuracy, etc.) with means and stds
  - `failure_modes`: list of detected patterns or slices where performance degrades
  - `suggested_next_steps`: list of prioritized actions (experiments, data collection)
  - `artifacts`: list of paths/filenames of generated artifacts (plots, CSVs)

- A human-readable Markdown summary suitable for PR descriptions or lab notes

## Constraints and Guardrails
- The agent **does not** make claims beyond extracted metrics and observed patterns; it must avoid overclaiming causality.
- The agent flags any missing or incomplete inputs and refuses to produce a summary if core evaluation metrics are absent.
- Confidential or PII-containing files will be ignored unless explicitly allowed by the user.

## Example usage
- CLI: `python scripts/run_research_agent.py --model artifacts/model.pkl --metrics metrics.json --out report.json`
- Programmatic: `from agents.research_agent import ResearchAgent; agent = ResearchAgent(); agent.run(<config>)`

## Progress and Reporting
- The agent logs progress at these steps: input validation, metrics extraction, slice analysis, summary generation, artifact write.
- When used within Copilot, the agent can prompt for missing inputs or clarification before continuing.

## Ideal Tools
- Small analysis utilities for loading common model formats and metrics files
- Plotting utilities for confusion matrices, residuals, and feature importance
- Optionally, lightweight LLM calls for rewrite/phrasing of bullets (must be permissive and auditable)

## Example output (JSON)
{
  "summary_bullets": [
    "Model X achieves MAE=0.23 on holdout, 12% better than baseline.",
    "Performance drops on commodity types A and B (MAE +0.4), likely due to low sample size.",
    "Residuals show heteroskedasticity at high volumes; consider log-transforming targets."
  ],
  "key_metrics": {"MAE": {"mean": 0.23, "std": 0.02}},
  "failure_modes": ["low-sample slices: commodity types A,B"],
  "suggested_next_steps": ["Increase data for A,B via targeted scraping", "Experiment with target transform"],
  "artifacts": ["reports/residuals.png", "reports/feature_importance.csv"]
}

---

Please implement the agent code under `agents/research_agent.py` and add a small CLI `scripts/run_research_agent.py` with example inputs and tests in `tests/test_research_agent.py`.