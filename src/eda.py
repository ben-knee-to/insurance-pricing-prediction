"""Step 3: reproducible descriptive EDA; run from any working directory."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed' / 'insurance_clean.csv'
REPORTS = ROOT / 'reports' / 'eda'
FIGURES = REPORTS / 'figures'
NUMERIC = ['age', 'bmi', 'children', 'charges']
CATEGORICAL = ['sex', 'smoker', 'region']
EXPECTED = NUMERIC + CATEGORICAL


def load_and_validate(path=DATA):
    df = pd.read_csv(path)
    missing_cols = set(EXPECTED) - set(df.columns)
    if missing_cols:
        raise ValueError(f'Missing required columns: {sorted(missing_cols)}')
    if df.empty or df[EXPECTED].isna().any().any():
        raise ValueError('Empty dataset or missing values in required fields')
    for col in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f'{col} must be numeric')
    if (df['charges'] <= 0).any() or (df['bmi'] <= 0).any():
        raise ValueError('Charges and BMI must be positive')
    if not df['smoker'].isin(['yes', 'no']).all():
        raise ValueError('Unexpected smoker categories')
    if df.duplicated().any():
        raise ValueError('Exact duplicate rows detected: rerun Step 2')
    return df


def grouped_summary(df, column):
    return (df.groupby(column, observed=True)['charges']
              .agg(count='count', mean='mean', median='median', std='std', min='min', max='max')
              .reset_index().sort_values(column).round(2))


def build_reports(df):
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df[NUMERIC].describe().round(3).to_csv(REPORTS / 'numeric_summary.csv')
    df[NUMERIC].corr(method='pearson').round(4).to_csv(REPORTS / 'correlations_pearson.csv')
    df[NUMERIC].corr(method='spearman').round(4).to_csv(REPORTS / 'correlations_spearman.csv')
    for col in CATEGORICAL + ['children']:
        grouped_summary(df, col).to_csv(REPORTS / f'charges_by_{col}.csv', index=False)
    by_age = df.assign(age_group=pd.cut(df.age, bins=[0, 25, 35, 45, 55, np.inf],
                            labels=['18-25','26-35','36-45','46-55','56+'], include_lowest=True))
    grouped_summary(by_age, 'age_group').to_csv(REPORTS / 'charges_by_age_group.csv', index=False)
    by_bmi = df.assign(bmi_group=pd.cut(df.bmi, bins=[0, 18.5, 25, 30, np.inf],
                            labels=['<18.5','18.5-<25','25-<30','30+'], right=False))
    grouped_summary(by_bmi, 'bmi_group').to_csv(REPORTS / 'charges_by_bmi_group.csv', index=False)
    smoker = grouped_summary(df, 'smoker').set_index('smoker')
    overall = df.charges
    summary = {
        'rows': int(len(df)), 'duplicate_rows': int(df.duplicated().sum()),
        'missing_cells': int(df.isna().sum().sum()),
        'mean_charges_usd': round(float(overall.mean()), 2),
        'median_charges_usd': round(float(overall.median()), 2),
        'min_charges_usd': round(float(overall.min()), 2),
        'max_charges_usd': round(float(overall.max()), 2),
        'smoker_count': int((df.smoker == 'yes').sum()),
        'non_smoker_count': int((df.smoker == 'no').sum()),
        'smoker_mean_charges_usd': round(float(smoker.loc['yes', 'mean']), 2),
        'non_smoker_mean_charges_usd': round(float(smoker.loc['no', 'mean']), 2),
        'smoker_to_non_smoker_mean_ratio': round(float(smoker.loc['yes', 'mean'] / smoker.loc['no', 'mean']), 3),
        'pearson_age_charges': round(float(df.age.corr(df.charges)), 4),
        'pearson_bmi_charges': round(float(df.bmi.corr(df.charges)), 4),
        'pearson_children_charges': round(float(df.children.corr(df.charges)), 4),
        'charge_skewness': round(float(df.charges.skew()), 3),
        'charge_95th_percentile_usd': round(float(df.charges.quantile(.95)), 2),
    }
    (REPORTS / 'eda_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def save_charts(df):
    sns.set_theme(style='whitegrid', palette='deep')
    def save(name):
        plt.tight_layout()
        plt.savefig(FIGURES / name, dpi=160, bbox_inches='tight')
        plt.close()
    plt.figure(figsize=(9,5))
    plt.hist(df['charges'], bins=35, edgecolor='white')
    plt.title('Distribution of medical charges'); plt.xlabel('Charges (USD)'); save('01_charges_distribution.png')
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='smoker', y='charges', order=['no','yes'])
    plt.title('Charges by smoking status'); plt.ylabel('Charges (USD)'); save('02_smoker_boxplot.png')
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=.65, s=30)
    plt.title('Age versus charges by smoking status'); save('03_age_vs_charges.png')
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=.65, s=30)
    plt.title('BMI versus charges by smoking status'); save('04_bmi_vs_charges.png')
    plt.figure(figsize=(9,5))
    sns.boxplot(data=df, x='region', y='charges', order=sorted(df.region.unique()))
    plt.title('Charges by region'); plt.xticks(rotation=15); save('05_region_boxplot.png')
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='children', y='charges')
    plt.title('Charges by number of children'); save('06_children_boxplot.png')
    plt.figure(figsize=(7,5))
    sns.heatmap(df[NUMERIC].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Pearson correlation (numeric variables)'); save('07_correlation_heatmap.png')
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='smoker', y='charges', hue='sex', order=['no','yes'])
    plt.title('Charges by smoking status and sex'); save('08_smoker_sex_boxplot.png')
    plt.figure(figsize=(9,5))
    
    for label in ['no', 'yes']:
        plt.hist(df.loc[df.smoker == label, 'charges'], bins=35, density=True, histtype='step', linewidth=2, label=label)
    plt.legend(title='smoker')
    plt.title('Charge distributions within smoking groups'); save('09_smoker_distributions.png')


def main():
    df = load_and_validate()
    summary = build_reports(df)
    save_charts(df)
    print(json.dumps(summary, indent=2))
    print(f'\nEDA outputs: {REPORTS}')

if __name__ == '__main__':
    main()
