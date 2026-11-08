"""Step 6: training-only tuning, followed by one final holdout evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prepare_features import ROOT, FEATURES, TARGET, SEED, build_preprocessor


def metrics(y, pred):
    return {'mae': float(mean_absolute_error(y, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'r2': float(r2_score(y, pred))}


def run(train_csv=None, test_csv=None, report_dir=None, model_dir=None):
    train_csv = Path(train_csv) if train_csv else ROOT / 'data/processed/ml/train.csv'
    test_csv = Path(test_csv) if test_csv else ROOT / 'data/processed/ml/test.csv'
    report_dir = Path(report_dir) if report_dir else ROOT / 'reports/ml/step6'
    model_dir = Path(model_dir) if model_dir else ROOT / 'models'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'figures').mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv)
    if train.source_row_id.duplicated().any():
        raise ValueError('Duplicate train row identifiers')
    X, y = train[FEATURES], train[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    searches = []
    for name, regressor, grid in [
        ('gradient_boosting', GradientBoostingRegressor(random_state=SEED),
         {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.05],
          'regressor__max_depth': [2, 3], 'regressor__min_samples_leaf': [3]}),
        ('random_forest', RandomForestRegressor(random_state=SEED, n_jobs=1),
         {'regressor__n_estimators': [150], 'regressor__max_features': [0.7, 1.0],
          'regressor__min_samples_leaf': [2, 4]})
    ]:
        for interactions in (False, True):
            pipe = Pipeline([('features', build_preprocessor(interactions=interactions)),
                             ('regressor', regressor)])
            search = GridSearchCV(pipe, grid, scoring='neg_mean_absolute_error', cv=cv,
                                  n_jobs=1, refit=True, return_train_score=False, error_score='raise')
            search.fit(X, y)
            row = {'model': name, 'features': 'interactions' if interactions else 'base',
                   'cv_mae': float(-search.best_score_), 'best_params': json.dumps(search.best_params_, sort_keys=True)}
            searches.append((row, search.best_estimator_))
            print(f"{name:20s} {row['features']:12s} CV MAE=${row['cv_mae']:,.2f}", flush=True)
    ranked = sorted(searches, key=lambda t: t[0]['cv_mae'])
    pd.DataFrame([r for r, _ in ranked]).to_csv(report_dir / 'tuning_comparison.csv', index=False)
    winner, fitted_model = ranked[0]
    # Diagnostic predictions on training folds only; selection is based on the CV searches above.
    oof = cross_val_predict(fitted_model, X, y, cv=cv, n_jobs=1)
    train_diag = train[['source_row_id', 'smoker', 'sex', 'region', 'age', 'bmi']].copy()
    train_diag['actual'] = y
    train_diag['oof_prediction'] = oof
    train_diag['absolute_error'] = np.abs(y.to_numpy() - oof)
    train_diag.to_csv(report_dir / 'training_oof_diagnostics.csv', index=False)
    # Final holdout: read only AFTER all candidate choices and hyperparameters are fixed.
    test = pd.read_csv(test_csv)
    if set(train.source_row_id) & set(test.source_row_id):
        raise ValueError('Train/test source row overlap')
    if test.source_row_id.duplicated().any():
        raise ValueError('Duplicate holdout row identifiers')
    prediction = fitted_model.predict(test[FEATURES])
    overall = metrics(test[TARGET], prediction)
    holdout = test[['source_row_id', 'smoker', 'sex', 'region', 'age', 'bmi']].copy()
    holdout['actual'] = test[TARGET]
    holdout['prediction'] = prediction
    holdout['residual'] = holdout['actual'] - holdout['prediction']
    holdout['absolute_error'] = holdout['residual'].abs()
    holdout.to_csv(report_dir / 'final_holdout_predictions.csv', index=False)
    segments = []
    groups = {'smoker': test['smoker'], 'sex': test['sex'], 'region': test['region'],
              'age_group': pd.cut(test.age, [0, 30, 45, 60, np.inf], labels=['under_30', '30_44', '45_59', '60_plus']),
              'bmi_group': pd.cut(test.bmi, [0, 18.5, 25, 30, np.inf], labels=['under_18_5', '18_5_24_9', '25_29_9', '30_plus'])}
    for dimension, group in groups.items():
        for label in group.dropna().unique():
            mask = (group == label).to_numpy()
            segment_metrics = (metrics(test.loc[mask, TARGET], prediction[mask]) if mask.sum() >= 2
                               else {'mae': float(np.abs(test.loc[mask, TARGET].to_numpy() - prediction[mask]).mean()),
                                     'rmse': float('nan'), 'r2': float('nan')})
            segments.append({'dimension': dimension, 'segment': str(label), 'n': int(mask.sum()),
                             **segment_metrics})
    pd.DataFrame(segments).to_csv(report_dir / 'holdout_segment_performance.csv', index=False)
    joblib.dump(fitted_model, model_dir / 'final_insurance_model.joblib')
    result = {'selected_model': winner['model'], 'feature_set': winner['features'],
              'best_params': json.loads(winner['best_params']), 'train_rows': len(train),
              'holdout_rows': len(test), 'selection_metric': 'minimum five-fold CV MAE',
              'selected_cv_mae': winner['cv_mae'], 'holdout_metrics': overall,
              'holdout_evaluated_once': True,
              'caution': 'Exploratory EDA used full data in earlier steps; do not claim a fully blinded research evaluation. No further tuning on this holdout.'}
    (report_dir / 'final_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(test[TARGET], prediction, alpha=.5, s=16)
    limit = max(float(test[TARGET].max()), float(np.max(prediction)))
    ax.plot([0, limit], [0, limit], 'k--')
    ax.set(xlabel='Actual medical charges (USD)', ylabel='Predicted charges (USD)', title='Final holdout: actual vs predicted')
    fig.tight_layout(); fig.savefig(report_dir / 'figures/holdout_actual_vs_predicted.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(holdout.residual, bins=30)
    ax.set(xlabel='Actual minus predicted (USD)', ylabel='Customers', title='Final holdout residuals')
    fig.tight_layout(); fig.savefig(report_dir / 'figures/holdout_residuals.png', dpi=150); plt.close(fig)
    print('FINAL HOLDOUT', json.dumps(result, indent=2), flush=True)
    return result

if __name__ == '__main__':
    run()
