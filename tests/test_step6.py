import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tune_models import metrics
from prepare_features import ROOT, FEATURES

def test_metrics_exact():
    m = metrics([1, 2, 3], [1, 2, 3])
    assert m['mae'] == 0 and m['rmse'] == 0 and m['r2'] == 1

def test_no_overlap():
    a = pd.read_csv(ROOT / 'data/processed/ml/train.csv')
    b = pd.read_csv(ROOT / 'data/processed/ml/test.csv')
    assert set(a.source_row_id).isdisjoint(set(b.source_row_id))
    assert len(a) == 1069 and len(b) == 268
    assert all(c in a.columns for c in FEATURES)

def test_saved_report_and_model():
    import json, joblib
    result = json.loads((ROOT / 'reports/ml/step6/final_evaluation.json').read_text())
    assert result['train_rows'] == 1069 and result['holdout_rows'] == 268
    model = joblib.load(ROOT / 'models/final_insurance_model.joblib')
    sample = pd.read_csv(ROOT / 'data/processed/ml/test.csv').head(3)
    pred = model.predict(sample[FEATURES])
    assert len(pred) == 3 and np.isfinite(pred).all()
