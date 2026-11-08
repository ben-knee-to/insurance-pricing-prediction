import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clean_data import clean_dataframe


def test_real_dataset_cleaning():
    raw = pd.read_csv(ROOT / 'data/raw/insurance.csv')
    clean, invalid, report = clean_dataframe(raw)
    assert report['input_rows'] == 1338
    assert report['duplicates_removed'] == 1
    assert report['invalid_rows_excluded'] == 0
    assert len(clean) == 1337
    assert invalid.empty
    assert not clean.isna().any().any()
    assert not clean.duplicated().any()


def test_bad_rows_audited_and_categories_normalized():
    raw = pd.DataFrame([
        [30, ' FEMALE ', 25, 0, ' NO ', ' Southwest ', 3000],
        [30, 'female', 25, 0, 'no', 'southwest', 3000],
        [40, 'unknown', -2, 1.5, 'yes', 'southeast', -10],
    ], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'])
    clean, invalid, report = clean_dataframe(raw)
    assert len(clean) == 1
    assert len(invalid) == 1
    assert report['duplicates_removed'] == 1
    assert clean.iloc[0]['sex'] == 'female'
    assert 'invalid_charges' in invalid.iloc[0]['reasons']


def test_schema_rejected():
    try:
        clean_dataframe(pd.DataFrame({'age': [20]}))
    except ValueError as exc:
        assert 'Unexpected schema' in str(exc)
    else:
        raise AssertionError('Schema validation must reject missing columns')
