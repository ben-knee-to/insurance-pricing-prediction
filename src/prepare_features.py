"""Step 4: reproducible, train-only fitted preprocessing for insurance charges."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
NUMERIC = ['age', 'bmi', 'children']
CATEGORICAL = ['sex', 'smoker', 'region']
TARGET = 'charges'
SEED = 42

class InsuranceInteractions(BaseEstimator, TransformerMixin):
    """Stateless feature engineering; accepts a pandas DataFrame."""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self._check(X)
        return self

    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame with named columns')
        missing = set(FEATURES).difference(X.columns)
        if missing:
            raise ValueError(f'Missing input features: {sorted(missing)}')

    def transform(self, X):
        self._check(X)
        out = X.loc[:, FEATURES].copy()
        if self.enabled:
            smoker = out['smoker'].astype('string').str.strip().str.lower().eq('yes').astype(int)
            out['smoker_bmi'] = pd.to_numeric(out['bmi'], errors='coerce') * smoker
            out['smoker_age'] = pd.to_numeric(out['age'], errors='coerce') * smoker
        return out


def build_preprocessor(interactions=True):
    numeric = NUMERIC + (['smoker_bmi', 'smoker_age'] if interactions else [])
    column_transformer = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encoder', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL)
    ], remainder='drop', verbose_feature_names_out=True)
    return Pipeline([('interactions', InsuranceInteractions(enabled=interactions)),
                     ('preprocess', column_transformer)])


def prepare(input_csv=None, output_dir=None, artifact_dir=None):
    input_csv = Path(input_csv) if input_csv else ROOT / 'data/processed/insurance_clean.csv'
    output_dir = Path(output_dir) if output_dir else ROOT / 'data/processed/ml'
    artifact_dir = Path(artifact_dir) if artifact_dir else ROOT / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    required = FEATURES + [TARGET]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df[required].isna().any().any():
        raise ValueError('Clean input has missing values; rerun Step 2 validation')
    if df.duplicated(subset=required).any():
        raise ValueError('Clean input contains duplicate rows')
    if not np.isfinite(df[TARGET].to_numpy(dtype=float)).all() or (df[TARGET] <= 0).any():
        raise ValueError('Target must be positive and finite')
    df = df.loc[:, required].copy()
    df.insert(0, 'source_row_id', np.arange(len(df), dtype=int))
    train, test = train_test_split(df, test_size=0.20, random_state=SEED,
                                   stratify=df['smoker'])
    train = train.sort_values('source_row_id').reset_index(drop=True)
    test = test.sort_values('source_row_id').reset_index(drop=True)
    train.to_csv(output_dir / 'train.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)
    X_train, X_test = train[FEATURES], test[FEATURES]
    # Fit preprocessing on training rows ONLY. Test rows are transformed, never fitted.
    output = {}
    for name, interactions in [('base', False), ('interactions', True)]:
        pipeline = build_preprocessor(interactions=interactions)
        train_arr = pipeline.fit_transform(X_train)
        test_arr = pipeline.transform(X_test)
        joblib.dump(pipeline, artifact_dir / f'preprocessor_{name}.joblib')
        output[name] = {'n_features': int(train_arr.shape[1]),
                        'train_shape': list(train_arr.shape), 'test_shape': list(test_arr.shape),
                        'feature_names': pipeline.named_steps['preprocess'].get_feature_names_out().tolist()}
    report = {'seed': SEED, 'stratified_on': 'smoker', 'train_rows': len(train),
              'test_rows': len(test), 'train_smokers': int((train.smoker == 'yes').sum()),
              'test_smokers': int((test.smoker == 'yes').sum()),
              'train_target_mean': float(train.charges.mean()),
              'test_target_mean': float(test.charges.mean()),
              'train_source_ids': train.source_row_id.tolist(),
              'test_source_ids': test.source_row_id.tolist(), 'pipelines': output,
              'note': 'Holdout test data must not be used for tuning, feature selection or model comparison.'}
    (output_dir / 'split_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Train: {len(train)} rows | Holdout test: {len(test)} rows")
    print('Saved split, manifest and fitted train-only preprocessors.')
    return report

if __name__ == '__main__':
    prepare()
