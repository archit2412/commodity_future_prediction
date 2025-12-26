# Mitsui & Co. Commodity Prediction Challenge – Practice Repository

This repository contains my **practice notebooks and experiments** for the [Mitsui & Co. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge) hosted on Kaggle.  
The competition involves building robust models to **predict commodity returns and return spreads** across multiple global markets.

## 📂 Contents
- **EDA (Exploratory Data Analysis)**  
  Initial data exploration, visualization of price trends, missing value handling, and correlation analysis.
  
- **Feature Engineering**  
  Creation of lagged features, rolling statistics, volatility indicators, and return-based transformations.

- **Modeling Experiments**  
  - Baseline models: Linear Regression, LightGBM  
  - Time-series validation approaches (rolling window, expanding window)  
  - Experiments with ensemble and deep learning models

- **Utilities**  
  Helper functions for preprocessing, validation, and submission file generation.

## 🛠 Tech Stack
- **Languages:** Python (Pandas, NumPy)
- **ML Libraries:** LightGBM, XGBoost, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Notebook Environment:** Kaggle & Jupyter

## 🚀 Goals
- Understand and analyze the competition dataset
- Build reliable baseline models
- Experiment with advanced time-series and machine learning techniques
- Generate valid submissions for the Kaggle leaderboard

## 📌 References
- [Official Kaggle Competition Page](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)
- Kaggle public notebooks for inspiration and benchmarking

---

*This repository is purely for learning and practice purposes and is not intended as a polished solution.*

## 🧪 Agents

This project includes a lightweight **Research Agent** that extracts concise research summaries from model artifacts and evaluation outputs.

Quick example:

```bash
python scripts/run_research_agent.py --predictions path/to/preds.csv --out reports/research_report.json
```

The agent expects a predictions CSV with `y_true` and `y_pred` columns and will write a JSON and Markdown summary to the `--out` path (and `<out>.md`).

