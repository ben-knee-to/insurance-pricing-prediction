import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eda import load_and_validate, grouped_summary, build_reports


def test_cleaned_data_loads():
    df = load_and_validate()
    assert len(df) == 1337
    assert df['charges'].gt(0).all()


def test_group_counts_cover_all_rows():
    df = load_and_validate()
    assert grouped_summary(df, 'smoker')['count'].sum() == len(df)


def test_missing_columns_fail(tmp_path):
    p = tmp_path / 'bad.csv'
    pd.DataFrame({'age': [25]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match='Missing required columns'):
        load_and_validate(p)


def test_report_metrics():
    summary = build_reports(load_and_validate())
    assert summary['smoker_count'] + summary['non_smoker_count'] == summary['rows']
    assert summary['smoker_to_non_smoker_mean_ratio'] > 1
