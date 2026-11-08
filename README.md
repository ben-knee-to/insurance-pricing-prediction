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
