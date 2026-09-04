#!/usr/bin/env python3
"""Joint regression of last-layer CKA on URIEL syntactic distance and mean NLL
(Section 3.2, "Linguistic similarity modulates alignment"): nested-model ANOVA
over the 36 Goldfish 1000 MB pairs.

Reads results/pairwise_data.csv (produced by analyze_predictors.py).
Requires statsmodels.
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

df = pd.read_csv(Path(__file__).resolve().parent / 'results' / 'pairwise_data.csv')
print(f'n = {len(df)} pairs')

for col, label in [('uriel_syntactic', 'URIEL syntactic'), ('mean_nll', 'mean NLL')]:
    r, p = stats.pearsonr(df[col], df.cka_sgpt_final)
    rho, prho = stats.spearmanr(df[col], df.cka_sgpt_final)
    print(f'{label:16s} r={r:+.3f} (p={p:.4g})  rho={rho:+.3f} (p={prho:.4g})')

m_uriel = smf.ols('cka_sgpt_final ~ uriel_syntactic', data=df).fit()
m_nll = smf.ols('cka_sgpt_final ~ mean_nll', data=df).fit()
m_both = smf.ols('cka_sgpt_final ~ uriel_syntactic + mean_nll', data=df).fit()

for name, m in [('URIEL only', m_uriel), ('NLL only', m_nll), ('URIEL + NLL', m_both)]:
    print(f'\n--- {name}: R2={m.rsquared:.4f} adj={m.rsquared_adj:.4f} ---')
    print(m.summary2().tables[1].round(4))

print('\nNLL beyond URIEL:')
print(anova_lm(m_uriel, m_both).round(5))
print('\nURIEL beyond NLL:')
print(anova_lm(m_nll, m_both).round(5))

cols = ['cka_sgpt_final', 'uriel_syntactic', 'mean_nll']
z = (df[cols] - df[cols].mean()) / df[cols].std()
mz = smf.ols('cka_sgpt_final ~ uriel_syntactic + mean_nll', data=z).fit()
print('\nstandardized betas:', mz.params.drop('Intercept').round(3).to_dict())
print(f"r(URIEL, NLL) = {stats.pearsonr(df.uriel_syntactic, df.mean_nll)[0]:+.3f}")
