# Insurance Charge Prediction & Analytics

End-to-end **Python / Scikit-learn / Streamlit** portfolio project using a 1,338-row US medical charges dataset (1,337 after removing one exact duplicate). Includes cleaning, EDA, customer segments, charge concentration, cross-validated regression, hyperparameter tuning, model explainability and a six-tab Streamlit dashboard.

## Results

| Metric | Frozen holdout result (268 observations) |
|---|---:|
| MAE | $2,116.69 |
| RMSE | $3,516.07 |
| R² | 0.9142 |

Selected model: Random Forest with 150 trees, `max_features=0.7`, `min_samples_leaf=4`, chosen on training-only five-fold CV (MAE $2,728.77). The final holdout was evaluated once. Earlier exploratory analysis examined the full dataset, so this is not a strictly blinded research benchmark.

**Data insights:** top 10% of observations account for approximately 31.99% of total charges; top 20% account for 51.69%. Smoking status, age and BMI were prominent predictive factors. These are associations, not causal findings.

## Quick start (Windows / PyCharm)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The downloadable archive includes a locally trained model under `models/final_insurance_model.joblib`. If the model is absent, regenerate the frozen pipeline with `python src/tune_models.py`. **Only load model files you generated or trust:** joblib uses pickle deserialization. Keep `src/prepare_features.py` available for custom pipeline class loading.

## Reproduce the pipeline

```powershell
python src/profile_data.py
python src/clean_data.py
python src/eda.py
python src/advanced_eda.py
python src/prepare_features.py
python src/train_models.py
python src/tune_models.py
python src/explain_model.py
streamlit run app.py
```

**Important:** Running `tune_models.py` again is for reproduction, not further test-set-driven model selection. Do not tune based on final holdout performance.

## Application pages

- Overview: dataset KPIs and charge distributions
- Customer analytics: smoking, age, BMI and demographic comparisons
- Cost concentration: Lorenz curve, top-decile shares and Gini coefficient
- Predict charges: validated form using the frozen trained pipeline
- Model performance: frozen holdout metrics and diagnostics
- Explainability: global permutation importance, SHAP and training-example contributions

## Repository layout

`app.py` is the dashboard entry point. `src/` contains the pipeline scripts and `app_utils.py`; `tests/` contains validation checks; `data/raw/` and `data/processed/` hold CSV data; `reports/` holds charts, CSV and JSON results; `models/` contains locally generated joblib artifacts (ignored by Git).

## Deployment

For local deployment, run `streamlit run app.py`. For Streamlit Community Cloud, push the repository, select `app.py`, and ensure the model artifact is provided by a **trusted** deployment build process (it is ignored by Git). Without the model, the predictor displays a clear missing-model warning; the analytics tabs still work. Review dataset redistribution permissions before publishing the CSV.

## Limitations and responsible use

This dataset represents US **medical charges**, not actual policy premiums, insurer claims, customer retention, or South African pricing. It is a small observational dataset with limited covariates and unknown collection representativeness. Results should not be used for medical advice, underwriting or insurance decisions. SHAP and feature importance do not establish causation. Model error and fairness may differ across groups. No calibrated prediction intervals are claimed.

## Portfolio summary

Built an end-to-end medical charge analytics and regression application on 1,337 cleaned observations, combining exploratory segmentation, cost concentration, leakage-aware model selection, a tuned Random Forest (holdout R² 0.914; MAE $2,117), SHAP explanations, automated tests and a six-tab Streamlit dashboard.
