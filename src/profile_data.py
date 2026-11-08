from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data' / 'raw' / 'insurance.csv'
OUT = ROOT / 'reports'
OUT.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INPUT)
    required = {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')
    summary = {
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'missing_values': {k: int(v) for k, v in df.isna().sum().items()},
        'duplicate_rows': int(df.duplicated().sum()),
        'dtypes': {k: str(v) for k, v in df.dtypes.items()},
        'charges_mean': float(df['charges'].mean()),
        'charges_median': float(df['charges'].median()),
        'charges_min': float(df['charges'].min()),
        'charges_max': float(df['charges'].max()),
        'smoker_counts': {str(k): int(v) for k, v in df['smoker'].value_counts().items()},
    }
    (OUT / 'profile_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    df.describe(include='all').transpose().to_csv(OUT / 'descriptive_statistics.csv')
    pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values, 'missing': df.isna().sum().values}).to_csv(OUT / 'column_profile.csv', index=False)
    print(json.dumps(summary, indent=2))
    print(f'\nReports saved to: {OUT}')

if __name__ == '__main__':
    main()
