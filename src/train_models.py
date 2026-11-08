"""Step 5: training-only cross-validation; never opens holdout test.csv."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prepare_features import ROOT, FEATURES, TARGET, SEED, build_preprocessor

MODELS = {
    'dummy': lambda: DummyRegressor(strategy='mean'),
    'linear': lambda: LinearRegression(),
    'ridge': lambda: Ridge(alpha=10.0),
    'random_forest': lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=SEED, n_jobs=1),
    'gradient_boosting': lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=SEED),
    'hist_gradient_boosting': lambda: HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED),
}

def build_model(name, interactions):
    return Pipeline([('features', build_preprocessor(interactions=interactions)),
                     ('regressor', MODELS[name]())])

def train(train_csv=None, report_dir=None, model_dir=None):
    train_csv = Path(train_csv) if train_csv else ROOT / 'data/processed/ml/train.csv'
    report_dir = Path(report_dir) if report_dir else ROOT / 'reports/ml'
    model_dir = Path(model_dir) if model_dir else ROOT / 'models'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'figures').mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(train_csv)
    if df['source_row_id'].duplicated().any() or df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError('Invalid training dataset')
    X, y = df[FEATURES], df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    rows, fold_rows = [], []
    for interaction in [False, True]:
        for name in MODELS:
            tag = 'interactions' if interaction else 'base'
            model = build_model(name, interaction)
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score='raise')
            for i in range(5):
                fold_rows.append({'model': name, 'features': tag, 'fold': i + 1,
                                  'mae': -scores['test_mae'][i], 'rmse': -scores['test_rmse'][i],
                                  'r2': scores['test_r2'][i]})
            rows.append({'model': name, 'features': tag,
                         'mae_mean': -scores['test_mae'].mean(), 'mae_std': scores['test_mae'].std(ddof=1),
                         'rmse_mean': -scores['test_rmse'].mean(), 'rmse_std': scores['test_rmse'].std(ddof=1),
                         'r2_mean': scores['test_r2'].mean(), 'r2_std': scores['test_r2'].std(ddof=1)})
            print(f'{name:24s} {tag:12s} MAE=${-scores["test_mae"].mean():,.0f} RMSE=${-scores["test_rmse"].mean():,.0f} R2={scores["test_r2"].mean():.3f}', flush=True)
    summary = pd.DataFrame(rows).sort_values(['mae_mean', 'rmse_mean']).reset_index(drop=True)
    summary.to_csv(report_dir / 'model_comparison.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(report_dir / 'cross_validation_results.csv', index=False)
    pivot = summary.pivot(index='model', columns='features', values='mae_mean')
    pivot['improvement_with_interactions_usd'] = pivot['base'] - pivot['interactions']
    pivot.to_csv(report_dir / 'feature_comparison.csv')
    winner = summary.iloc[0]
    model = build_model(winner['model'], winner['features'] == 'interactions')
    # Out-of-fold predictions for descriptive segment error analysis; no holdout access.
    oof = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    segments = []
    for variable in ['smoker', 'sex', 'region']:
        for label, idx in df.groupby(variable).groups.items():
            ids = df.index.get_indexer(idx)
            segments.append({'dimension': variable, 'segment': label, 'n': len(ids),
                             'oof_mae': mean_absolute_error(y.iloc[ids], oof[ids]),
                             'oof_rmse': np.sqrt(mean_squared_error(y.iloc[ids], oof[ids]))})
    for variable, groups in [('age_group', pd.cut(df.age, [0, 30, 45, 60, np.inf], labels=['<30','30-44','45-59','60+'])),
                             ('bmi_group', pd.cut(df.bmi, [0, 18.5, 25, 30, np.inf], labels=['<18.5','18.5-24.9','25-29.9','30+']))]:
        for label in groups.cat.categories:
            ids = np.flatnonzero((groups == label).to_numpy())
            if len(ids):
                segments.append({'dimension': variable, 'segment': str(label), 'n': len(ids),
                                 'oof_mae': mean_absolute_error(y.iloc[ids], oof[ids]),
                                 'oof_rmse': np.sqrt(mean_squared_error(y.iloc[ids], oof[ids]))})
    pd.DataFrame(segments).to_csv(report_dir / 'segment_performance.csv', index=False)
    pd.DataFrame({'source_row_id': df.source_row_id, 'actual': y, 'oof_prediction': oof,
                  'residual': y - oof}).to_csv(report_dir / 'oof_predictions.csv', index=False)
    model.fit(X, y)
    joblib.dump(model, model_dir / 'best_baseline_model.joblib')
    meta = {'selected_model': winner['model'], 'feature_set': winner['features'],
            'selection_metric': 'lowest mean 5-fold CV MAE', 'train_rows': len(df),
            'holdout_test_used': False, 'cv_seed': SEED,
            'caution': 'Exploratory CV winner; final tuning and holdout evaluation remain in Step 6.'}
    (report_dir / 'model_selection.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10, 6))
    top = summary.head(10).iloc[::-1]
    ax.barh(top['model'] + ' / ' + top['features'], top['mae_mean'])
    ax.set_xlabel('5-fold cross-validated MAE (USD)')
    ax.set_title('Insurance charges — model comparison (training only)')
    fig.tight_layout(); fig.savefig(report_dir / 'figures/model_comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, oof, s=10, alpha=0.4)
    lo, hi = min(y.min(), oof.min()), max(y.max(), oof.max())
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='black')
    ax.set(xlabel='Actual charges (USD)', ylabel='Out-of-fold predicted charges (USD)', title='Training-only out-of-fold predictions')
    fig.tight_layout(); fig.savefig(report_dir / 'figures/actual_vs_predicted.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(y - oof, bins=40)
    ax.set(xlabel='Actual minus predicted (USD)', ylabel='Count', title='Out-of-fold residuals')
    fig.tight_layout(); fig.savefig(report_dir / 'figures/residual_distribution.png', dpi=160); plt.close(fig)
    print('\nSelected candidate:', meta['selected_model'], meta['feature_set'])
    return summary

if __name__ == '__main__':
    train()
