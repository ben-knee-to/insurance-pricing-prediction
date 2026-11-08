import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from advanced_eda import load_data, concentration, add_segments, segment_summary


def test_clean_dataset():
    assert len(load_data()) == 1337


def test_concentration_equal_values():
    d = pd.DataFrame({'charges':[10.,10.,10.,10.]})
    summary, lorenz, ranked = concentration(d)
    assert summary['gini_coefficient'] == pytest.approx(0)
    assert summary['top_20pct_share'] == pytest.approx(.25)
    assert lorenz.charges_share.iloc[-1] == pytest.approx(1)
    assert ranked.cumulative_charge_share.iloc[-1] == pytest.approx(1)


def test_concentration_real():
    summary, _, _ = concentration(load_data())
    assert 0 < summary['top_10pct_share'] < summary['top_20pct_share'] < 1
    assert 0 <= summary['gini_coefficient'] <= 1


def test_segments_preserve_rows():
    d = load_data()
    s = add_segments(d)
    assert len(s) == len(d)
    assert s[['age_group','bmi_group']].notna().all().all()
    assert segment_summary(s, ['smoker','bmi_group'])['count'].sum() == len(d)


def test_invalid_duplicate(tmp_path):
    d = load_data()
    bad = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    p = tmp_path / 'bad.csv'
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match='duplicates'):
        load_data(p)
