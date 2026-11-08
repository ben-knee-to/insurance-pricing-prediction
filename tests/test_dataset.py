from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_columns():
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'insurance.csv')
    assert {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'} <= set(df.columns)
    assert len(df) > 0

def test_target_is_nonnegative():
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'insurance.csv')
    assert df['charges'].notna().all()
    assert (df['charges'] >= 0).all()
