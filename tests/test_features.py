import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_features import InsuranceInteractions, build_preprocessor, prepare, FEATURES


def test_interactions():
    df = pd.DataFrame({'age': [40, 50], 'sex': ['female', 'male'], 'bmi': [30., 35.],
                       'children': [0, 2], 'smoker': ['no', 'yes'], 'region': ['northwest', 'southwest']})
    result = InsuranceInteractions().fit_transform(df)
    assert result['smoker_bmi'].tolist() == [0., 35.]
    assert result['smoker_age'].tolist() == [0, 50]


def test_unknown_category_supported():
    x = pd.read_csv(ROOT / 'data/processed/insurance_clean.csv')[FEATURES].head(30)
    pipe = build_preprocessor()
    pipe.fit(x)
    new = x.iloc[[0]].copy()
    new['region'] = 'new_region'
    assert pipe.transform(new).shape[1] == pipe.transform(x.iloc[[0]]).shape[1]


def test_split_and_saved_pipeline(tmp_path):
    report = prepare(output_dir=tmp_path / 'ml', artifact_dir=tmp_path / 'models')
    train = pd.read_csv(tmp_path / 'ml/train.csv')
    test = pd.read_csv(tmp_path / 'ml/test.csv')
    assert len(train) == 1069 and len(test) == 268
    assert not set(train.source_row_id) & set(test.source_row_id)
    assert set(train.source_row_id) | set(test.source_row_id) == set(range(1337))
    for name in ('base', 'interactions'):
        pipe = joblib.load(tmp_path / f'models/preprocessor_{name}.joblib')
        assert pipe.transform(test[FEATURES]).shape[0] == 268
        assert np.isfinite(pipe.transform(test[FEATURES])).all()
    assert report['pipelines']['interactions']['n_features'] == report['pipelines']['base']['n_features'] + 2


def test_reproducible_split(tmp_path):
    a = prepare(output_dir=tmp_path / 'a', artifact_dir=tmp_path / 'ma')
    b = prepare(output_dir=tmp_path / 'b', artifact_dir=tmp_path / 'mb')
    assert a['train_source_ids'] == b['train_source_ids']
    assert a['test_source_ids'] == b['test_source_ids']
