"""Expanded descriptive EDA. Run: python src/advanced_eda.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/insurance_clean.csv'
OUT = ROOT / 'reports/eda/advanced'
FIG = OUT / 'figures'
REQUIRED = {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'}


def load_data(path=DATA):
    df = pd.read_csv(path)
    if not REQUIRED.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(REQUIRED - set(df.columns))}')
    if df.empty or df[list(REQUIRED)].isna().any().any():
        raise ValueError('Dataset is empty or has missing values')
    if df.duplicated().any():
        raise ValueError('Exact duplicates found; run Step 2 cleaning first')
    if not (df['charges'] > 0).all() or not (df['bmi'] > 0).all():
        raise ValueError('Nonpositive charges or BMI')
    if not df.smoker.isin(['yes', 'no']).all():
        raise ValueError('Unexpected smoker status')
    if not df.sex.isin(['male', 'female']).all():
        raise ValueError('Unexpected sex category')
    if not df.region.isin(['northeast','northwest','southeast','southwest']).all():
        raise ValueError('Unexpected region')
    if not (df.age.between(18, 100)).all() or not (df.children.between(0, 20)).all():
        raise ValueError('Unexpected age or children count')
    return df


def concentration(df):
    ordered = df.sort_values('charges', ascending=False).reset_index(drop=True)
    n = len(ordered)
    total = ordered.charges.sum()
    shares = {}
    for fraction in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(np.ceil(n * fraction)))
        shares[f'top_{int(fraction*100)}pct_share'] = float(ordered.charges.iloc[:k].sum() / total)
        shares[f'top_{int(fraction*100)}pct_count'] = k
    asc = np.sort(df.charges.to_numpy(dtype=float))
    gini = float((2 * np.arange(1, n+1) - n - 1) @ asc / (n * asc.sum()))
    lorenz = pd.DataFrame({'population_share': np.arange(n+1)/n,
                           'charges_share': np.r_[0, np.cumsum(asc)/asc.sum()]})
    ordered['cumulative_customer_share'] = np.arange(1, n+1)/n
    ordered['cumulative_charge_share'] = ordered.charges.cumsum()/total
    return {**shares, 'gini_coefficient': gini, 'total_charges_usd': float(total)}, lorenz, ordered


def add_segments(df):
    d = df.copy()
    d['age_group'] = pd.cut(d.age, [17, 29, 44, 59, np.inf], labels=['18-29','30-44','45-59','60+'])
    d['bmi_group'] = pd.cut(d.bmi, [0, 18.5, 25, 30, np.inf], right=False,
                            labels=['<18.5','18.5-<25','25-<30','30+'])
    d['smoker_bmi_segment'] = d.smoker + ' | ' + d.bmi_group.astype(str)
    d['age_smoker_segment'] = d.age_group.astype(str) + ' | ' + d.smoker
    return d


def segment_summary(d, columns):
    return (d.groupby(columns, observed=True).charges
            .agg(count='size', mean='mean', median='median', std='std', total='sum')
            .reset_index().sort_values('mean', ascending=False))


def plot_and_save():
    plt.tight_layout()
    plt.savefig(FIG / plot_and_save.filename, dpi=170, bbox_inches='tight')
    plt.close()


def run(path=DATA):
    df = load_data(path)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')
    conc, lorenz, pareto = concentration(df)
    lorenz.to_csv(OUT/'lorenz_curve.csv', index=False)
    pareto[['charges','cumulative_customer_share','cumulative_charge_share']].to_csv(OUT/'cost_concentration_ranked.csv', index=False)
    d = add_segments(df)
    group_cols = [['smoker','bmi_group'], ['age_group','smoker'], ['sex','smoker'],
                  ['region','smoker'], ['children','smoker'], ['smoker_bmi_segment']]
    names = ['smoking_bmi', 'age_smoking', 'sex_smoking', 'region_smoking', 'children_smoking', 'customer_segments']
    for cols, name in zip(group_cols, names):
        segment_summary(d, cols).round(3).to_csv(OUT/f'{name}.csv', index=False)
    # Descriptive subgroup associations, not causal effects or model feature importance.
    correlations = []
    for status, g in d.groupby('smoker'):
        for feature in ['age','bmi','children']:
            correlations.append({'smoker': status, 'feature': feature,
                                 'pearson_r': float(g[feature].corr(g.charges)), 'n': len(g)})
    pd.DataFrame(correlations).to_csv(OUT/'within_smoking_correlations.csv', index=False)
    # Aggregate descriptive profiles of top-decile observed charges.
    threshold = float(d.charges.quantile(.9))
    d['observed_top_decile'] = d.charges >= threshold
    profiles = []
    for col in ['smoker','sex','region','age_group','bmi_group']:
        g = d.groupby(col, observed=True).agg(n=('charges','size'), top_decile_n=('observed_top_decile','sum'))
        g['top_decile_rate'] = g.top_decile_n / g.n
        g['variable'] = col
        g['category'] = g.index.astype(str)
        profiles.append(g.reset_index(drop=True)[['variable','category','n','top_decile_n','top_decile_rate']])
    pd.concat(profiles, ignore_index=True).to_csv(OUT/'high_charge_customer_profiles.csv', index=False)
    plt.figure(figsize=(7,5))
    plt.plot(lorenz.population_share, lorenz.charges_share, label='Observed charges')
    plt.plot([0,1],[0,1],'--',label='Equal distribution')
    plt.xlabel('Cumulative share of customers (lowest charges first)')
    plt.ylabel('Cumulative share of charges')
    plt.title('Lorenz curve of medical charges'); plt.legend()
    plot_and_save.filename='10_lorenz_curve.png'; plot_and_save()
    plt.figure(figsize=(8,5))
    plt.plot(pareto.cumulative_customer_share, pareto.cumulative_charge_share)
    plt.axvline(.1, ls='--', color='grey'); plt.axvline(.2, ls='--', color='grey')
    plt.xlabel('Top fraction of customers (highest charges first)')
    plt.ylabel('Cumulative fraction of total charges')
    plt.title('Cost concentration / Pareto curve')
    plot_and_save.filename='11_cost_concentration.png'; plot_and_save()
    plt.figure(figsize=(9,5))
    sns.boxplot(data=d, x='bmi_group', y='charges', hue='smoker', showfliers=False)
    plt.title('Charges by BMI category and smoking status')
    plot_and_save.filename='12_smoking_bmi_interaction.png'; plot_and_save()
    plt.figure(figsize=(9,5))
    sns.lineplot(data=d, x='age_group', y='charges', hue='smoker', estimator='mean', errorbar=None)
    plt.ylabel('Mean charges (USD)'); plt.title('Age group and smoking interaction (descriptive)')
    plot_and_save.filename='13_age_smoking_interaction.png'; plot_and_save()
    plt.figure(figsize=(9,5))
    seg = segment_summary(d, ['smoker_bmi_segment'])
    sns.barplot(data=seg, x='mean', y='smoker_bmi_segment', color='steelblue')
    plt.xlabel('Mean charges (USD)'); plt.ylabel('Customer segment')
    plt.title('Average charges by smoking and BMI segment')
    plot_and_save.filename='14_customer_segments.png'; plot_and_save()
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=d, x='bmi', y='charges', hue='smoker', alpha=.45)
    plt.title('BMI vs charges, coloured by smoking status')
    plot_and_save.filename='15_bmi_smoker_scatter.png'; plot_and_save()
    summary = {'rows': len(d), 'cost_concentration': conc,
               'observed_top_decile_threshold_usd': threshold,
               'top_decile_observed_count': int(d.observed_top_decile.sum()),
               'notes': ['Full-sample exploratory summaries only; no predictive models fitted.',
                         'SHAP requires a fitted model and will be performed in Step 7.',
                         'Charges are medical charges, not premiums or insurer claims.',
                         'Differences are observational, not causal.']}
    (OUT/'advanced_eda_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Reports saved to:', OUT)
    return summary


if __name__ == '__main__':
    run()
