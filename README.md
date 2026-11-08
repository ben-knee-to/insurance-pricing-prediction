# Insurance Pricing Prediction & Analytics

End-to-end Python data science portfolio project predicting medical insurance charges from demographic and lifestyle features.

## Dataset
`data/raw/insurance.csv` contains 1,338 rows and 7 columns. Target: `charges` (USD). These are medical charges, not verified insurance premiums.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/profile_data.py
```

## Step 1 outputs
- `reports/profile_summary.json`
- `reports/descriptive_statistics.csv`
- `reports/column_profile.csv`

## Roadmap
1. Profiling and GitHub setup
2. Cleaning and validation
3. EDA
4. Feature engineering
5. Baseline models
6. Model tuning
7. Explainability
8. Streamlit application
9. Deployment and documentation

## Notes
Dataset is a common public demonstration dataset. Verify original source/license before redistributing. This model is educational and not appropriate for real insurance underwriting or pricing decisions.

## Step 2 — Cleaning and validation
```powershell
python src/clean_data.py
python -m pytest -q
```
Produces `data/processed/insurance_clean.csv`, `reports/cleaning_report.json`, and
`reports/invalid_rows.csv`. Cleaning normalizes categories, removes exact duplicates,
and audits invalid rows. High-but-valid BMI and charges are retained. Never fit scalers,
imputers, or encoders on the complete dataset before the train/test split.

## Step 3 — Exploratory data analysis
```powershell
python src/eda.py
python -m pytest -q
```
Outputs: `reports/eda/eda_summary.json`, grouped charge summaries, Pearson and Spearman correlations, and nine PNG charts in `reports/eda/figures/`. This is descriptive observational analysis, not causal inference. Smoker comparisons are unadjusted. The dataset is not a real-world actuarial pricing benchmark.

## Expanded Step 3 — Advanced EDA

Run `python src/advanced_eda.py` after `python src/clean_data.py` and `python src/eda.py`.
Outputs in `reports/eda/advanced/` include Lorenz/Pareto curves, Gini coefficient,
smoking × BMI and smoking × age segment comparisons, subgroup correlations,
high-charge customer profiles, six new figures and JSON/CSV reports.
These are descriptive associations, not causal effects or predictive explanations.
SHAP will be calculated after fitting models using a leakage-safe train/test split.
Run tests with `python -m pytest -q`.
