"""Step 2: reproducible, conservative cleaning and data-quality checks."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/insurance.csv'
CLEAN = ROOT / 'data/processed/insurance_clean.csv'
REPORT = ROOT / 'reports/cleaning_report.json'
ISSUES = ROOT / 'reports/invalid_rows.csv'
EXPECTED = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
ALLOWED = {
    'sex': {'male', 'female'},
    'smoker': {'yes', 'no'},
    'region': {'northeast', 'northwest', 'southeast', 'southwest'},
}


def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (valid cleaned records, invalid records, audit report)."""
    if raw.columns.duplicated().any():
        raise ValueError('Duplicate column names in input')
    missing_columns = sorted(set(EXPECTED) - set(raw.columns))
    extra_columns = sorted(set(raw.columns) - set(EXPECTED))
    if missing_columns or extra_columns:
        raise ValueError(f'Unexpected schema: missing={missing_columns}, extra={extra_columns}')
    df = raw[EXPECTED].copy()
    original_count = len(df)
    # Preserve the raw row number for tracing rejected rows.
    df.insert(0, 'source_row', range(2, original_count + 2))
    for col in ALLOWED:
        df[col] = df[col].astype('string').str.strip().str.lower()
        df[col] = df[col].replace('', pd.NA)
    for col in ['age', 'bmi', 'children', 'charges']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Exact duplicates after harmless normalization; first occurrence retained.
    duplicates = df.duplicated(subset=EXPECTED, keep='first')
    duplicate_count = int(duplicates.sum())
    df = df.loc[~duplicates].copy()

    problems = pd.Series('', index=df.index, dtype='string')
    def flag(mask: pd.Series, reason: str) -> None:
        nonlocal problems
        problems = problems.mask(mask.fillna(False), problems + reason + ';')

    for col, values in ALLOWED.items():
        flag(~df[col].isin(values), f'invalid_{col}')
    for col in ['age', 'bmi', 'children', 'charges']:
        flag(df[col].isna(), f'missing_or_non_numeric_{col}')
    flag(df['age'].notna() & (~df['age'].between(0, 120) | (df['age'] % 1 != 0)), 'invalid_age')
    flag(df['bmi'].notna() & (df['bmi'] <= 0), 'invalid_bmi')
    flag(df['children'].notna() & ((df['children'] < 0) | (df['children'] % 1 != 0)), 'invalid_children')
    flag(df['charges'].notna() & (df['charges'] < 0), 'invalid_charges')

    invalid_mask = problems.str.len() > 0
    invalid = df.loc[invalid_mask].copy()
    invalid['reasons'] = problems.loc[invalid_mask]
    valid = df.loc[~invalid_mask, EXPECTED].copy().reset_index(drop=True)
    valid['age'] = valid['age'].astype('int64')
    valid['children'] = valid['children'].astype('int64')

    report = {
        'input_rows': original_count,
        'duplicates_removed': duplicate_count,
        'invalid_rows_excluded': int(invalid_mask.sum()),
        'clean_rows': len(valid),
        'missing_values_after_cleaning': {k: int(v) for k, v in valid.isna().sum().items()},
        'categories': {k: sorted(valid[k].dropna().unique().tolist()) for k in ALLOWED},
        'charges_summary_usd': {
            'min': float(valid['charges'].min()) if len(valid) else None,
            'median': float(valid['charges'].median()) if len(valid) else None,
            'max': float(valid['charges'].max()) if len(valid) else None,
        },
        'note': 'Extreme but valid values retained. Invalid rows excluded and audited, never silently imputed.',
    }
    assert original_count == duplicate_count + len(invalid) + len(valid)
    return valid, invalid, report


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f'Place original insurance.csv at {RAW}')
    raw = pd.read_csv(RAW)
    clean, invalid, report = clean_dataframe(raw)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN, index=False)
    invalid.to_csv(ISSUES, index=False)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nSaved {CLEAN}, {ISSUES}, {REPORT}')


if __name__ == '__main__':
    main()
