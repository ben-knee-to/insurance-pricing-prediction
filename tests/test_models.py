import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train_models import MODELS, build_model
from prepare_features import ROOT, FEATURES

def test_model_variants():
    for name in MODELS:
        for interactions in (False, True):
            model = build_model(name, interactions)
            assert model.named_steps['features'].named_steps['interactions'].enabled == interactions

def test_training_split_and_disjoint_holdout():
    train = pd.read_csv(ROOT / 'data/processed/ml/train.csv')
    test = pd.read_csv(ROOT / 'data/processed/ml/test.csv')
    assert len(train) == 1069 and len(test) == 268
    assert set(train.source_row_id).isdisjoint(test.source_row_id)

def test_pipeline_predicts_raw_rows():
    train = pd.read_csv(ROOT / 'data/processed/ml/train.csv')
    model = build_model('ridge', True)
    model.fit(train[FEATURES].head(100), train.charges.head(100))
    pred = model.predict(train[FEATURES].tail(5))
    assert pred.shape == (5,) and np.isfinite(pred).all()
