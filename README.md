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

## Step 4 — Leakage-safe feature preparation

Run `python src/prepare_features.py` after Steps 1–3, then `python -m pytest -q`.
This creates `data/processed/ml/train.csv`, `test.csv`, `split_manifest.json`, and
`models/preprocessor_base.joblib` and `models/preprocessor_interactions.joblib`.
The split is fixed (random seed 42), 80/20, stratified on smoking status.
Preprocessing is fitted **only on training rows**; the holdout is not used to
choose models, features, or hyperparameters. `source_row_id` is an audit key,
not a model input. The fitted preprocessors are exploration artifacts; Step 5
should fit a complete model pipeline on training data (and within each CV fold)
to avoid validation-fold leakage. Never load untrusted joblib artifacts.

## Step 5 — Baseline model development

Run `python src/prepare_features.py` first, then `python src/train_models.py`.
This compares six regression algorithms with and without engineered smoking interactions using
five-fold training-only cross-validation. It saves results under `reports/ml/` and the selected
full pipeline to `models/best_baseline_model.joblib`. **The holdout `test.csv` is never opened
by the training script.** Selection is based on the lowest mean cross-validation MAE;
Step 6 will tune candidates and perform a single final holdout evaluation.

Metrics: MAE and RMSE in USD, R² unitless. Segment errors use out-of-fold predictions.
Medical charges are not insurance premiums. Do not use this demonstration model for underwriting.

## Step 6 — Model tuning and final holdout evaluation

Run `python src/tune_models.py` from the repository root. It performs five-fold training-only grid searches over Random Forest and Gradient Boosting with and without smoking interaction features, chooses the configuration with lowest cross-validated MAE, then evaluates the selected fitted pipeline once on the holdout set. Outputs are saved in `reports/ml/step6/` and the full fitted pipeline in `models/final_insurance_model.joblib`.

Selected: Random Forest, base features, 150 trees, max_features=0.7, min_samples_leaf=4. CV MAE: $2,728.77. Holdout MAE: $2,116.69, RMSE: $3,516.07, R²: 0.9142. This is a public synthetic/educational-style dataset, not a production pricing study. Earlier exploratory analyses inspected the entire dataset, so the holdout is not fully blinded in a strict research sense. Do not tune again against this holdout. In Step 7, explain the frozen model without modifying it.
