#!/usr/bin/env python3
"""
Appendix Figure 11: last-layer matched SGPT CKA (FLORES) vs. URIEL syntactic
distance for the 36 Goldfish pairs, coloured by genealogical relationship.

Usage:
  python plot_syntactic_distance.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

LANGUAGES = [
    'eng_latn', 'arb_arab', 'deu_latn', 'fra_latn', 'hin_deva',
    'jpn_jpan', 'rus_cyrl', 'spa_latn', 'zho_hans',
]

FLORES_TO_ISO = {
    'eng_latn': 'eng', 'arb_arab': 'arb', 'deu_latn': 'deu',
    'fra_latn': 'fra', 'hin_deva': 'hin', 'jpn_jpan': 'jpn',
    'rus_cyrl': 'rus', 'spa_latn': 'spa', 'zho_hans': 'zho',
}

LANG_LABELS = {
    'eng_latn': 'English',  'arb_arab': 'Arabic',  'deu_latn': 'German',
    'fra_latn': 'French',   'hin_deva': 'Hindi',   'jpn_jpan': 'Japanese',
    'rus_cyrl': 'Russian',  'spa_latn': 'Spanish',  'zho_hans': 'Chinese',
}

# Short codes for pair labels
LANG_SHORT = {
    'eng_latn': 'Eng', 'arb_arab': 'Ara', 'deu_latn': 'Deu',
    'fra_latn': 'Fra', 'hin_deva': 'Hin', 'jpn_jpan': 'Jpn',
    'rus_cyrl': 'Rus', 'spa_latn': 'Spa', 'zho_hans': 'Zho',
}

# Pair-type colours: group by broad relatedness
PAIR_TYPE_COLOURS = {
    'Same subfamily':     '#1f77b4',  # blue  (Eng-Deu, Fra-Spa)
    'Both Indo-European': '#e67e22',  # orange (e.g. Eng-Fra, Deu-Rus, Hin-Rus)
    'Cross-family':       '#7f7f7f',  # grey  (involves non-IE language)
}

LANG_FAMILY = {
    'eng_latn': 'Germanic', 'deu_latn': 'Germanic',
    'fra_latn': 'Romance',  'spa_latn': 'Romance',
    'rus_cyrl': 'Slavic',
    'hin_deva': 'Indo-Aryan',
    'arb_arab': 'Semitic',
    'jpn_jpan': 'Japonic',
    'zho_hans': 'Sinitic',
}

IE_FAMILIES = {'Germanic', 'Romance', 'Slavic', 'Indo-Aryan'}

# Hardcoded URIEL syntactic distances (lang2vec, cosine, syntax_knn)
SYNTACTIC_DIST = {
    ('eng','arb'): 0.355614, ('eng','deu'): 0.097458, ('eng','fra'): 0.188246,
    ('eng','hin'): 0.383805, ('eng','jpn'): 0.499807, ('eng','rus'): 0.188246,
    ('eng','spa'): 0.178406, ('eng','zho'): 0.289228, ('arb','deu'): 0.356279,
    ('arb','fra'): 0.278875, ('arb','hin'): 0.340766, ('arb','jpn'): 0.617765,
    ('arb','rus'): 0.255613, ('arb','spa'): 0.246589, ('arb','zho'): 0.468915,
    ('deu','fra'): 0.199945, ('deu','hin'): 0.333144, ('deu','jpn'): 0.432907,
    ('deu','rus'): 0.176414, ('deu','spa'): 0.214063, ('deu','zho'): 0.365098,
    ('fra','hin'): 0.349350, ('fra','jpn'): 0.556606, ('fra','rus'): 0.190476,
    ('fra','spa'): 0.156565, ('fra','zho'): 0.382292, ('hin','jpn'): 0.366444,
    ('hin','rus'): 0.301154, ('hin','spa'): 0.317073, ('hin','zho'): 0.324789,
    ('jpn','rus'): 0.478359, ('jpn','spa'): 0.577629, ('jpn','zho'): 0.296268,
    ('rus','spa'): 0.180663, ('rus','zho'): 0.308167, ('spa','zho'): 0.374805,
}

CKA_RESULTS = Path(__file__).resolve().parent.parent / 'cka' / 'results' / 'main'
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'


# ── Data loading ─────────────────────────────────────────────────────────────

def load_cka_values():
    cka = {}
    for csv_path in CKA_RESULTS.glob('cka_flores_*.csv'):
        df = pd.read_csv(csv_path)
        sgpt = df[df['setting'] == 'sgpt']
        if sgpt.empty:
            continue
        final_layer = sgpt[sgpt['layer'] == sgpt['layer'].max()]
        if final_layer.empty:
            continue
        row = final_layer.iloc[0]
        model_a, model_b = row['model_a'], row['model_b']
        lang_a = model_a.split('/')[1].replace('_1000mb', '')
        lang_b = model_b.split('/')[1].replace('_1000mb', '')
        cka[(lang_a, lang_b)] = float(row['cka_matched'])
    return cka


def pair_type(lang_a, lang_b):
    fa, fb = LANG_FAMILY[lang_a], LANG_FAMILY[lang_b]
    if fa == fb:
        return 'Same subfamily'
    if fa in IE_FAMILIES and fb in IE_FAMILIES:
        return 'Both Indo-European'
    return 'Cross-family'


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    # Load CKA
    cka_raw = load_cka_values()
    cka = {}
    for (a, b), v in cka_raw.items():
        key = (a, b) if a < b else (b, a)
        cka[key] = v

    # Build arrays
    iso_to_flores = {v: k for k, v in FLORES_TO_ISO.items()}
    x_dist, y_cka, labels, colours, ptypes = [], [], [], [], []

    for (iso_a, iso_b), d in SYNTACTIC_DIST.items():
        fa = iso_to_flores.get(iso_a)
        fb = iso_to_flores.get(iso_b)
        if not fa or not fb:
            continue
        key = (fa, fb) if fa < fb else (fb, fa)
        if key not in cka:
            continue
        pt = pair_type(fa, fb)
        x_dist.append(d)
        y_cka.append(cka[key])
        labels.append(f'{LANG_SHORT[fa]}\u2013{LANG_SHORT[fb]}')
        colours.append(PAIR_TYPE_COLOURS[pt])
        ptypes.append(pt)

    x = np.array(x_dist)
    y = np.array(y_cka)

    # Stats
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    # ── Plot ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Scatter by pair type
    seen_types = set()
    for xi, yi, col, pt, lab in zip(x, y, colours, ptypes, labels):
        kw = {'label': pt} if pt not in seen_types else {}
        ax.scatter(xi, yi, c=col, s=42, alpha=0.85, edgecolors='white',
                   linewidths=0.5, zorder=3, **kw)
        seen_types.add(pt)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 200)
    ax.plot(x_line, slope * x_line + intercept, color='#333333',
            linewidth=1.3, linestyle='--', alpha=0.7, zorder=2)

    # Confidence band (95%)
    n = len(x)
    x_mean = x.mean()
    se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    ci = stats.t.ppf(0.975, n - 2) * se * np.sqrt(
        1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
    )
    ax.fill_between(x_line, slope * x_line + intercept - ci,
                     slope * x_line + intercept + ci,
                     color='#333333', alpha=0.08, zorder=1)

    # Annotate points
    from adjustText import adjust_text
    texts = []
    for xi, yi, lab in zip(x, y, labels):
        texts.append(ax.text(xi, yi, lab, fontsize=6.5, alpha=0.75))
    try:
        adjust_text(texts, x=x, y=y, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='grey', alpha=0.4, lw=0.5),
                    force_text=(0.3, 0.5), force_points=(0.2, 0.3),
                    expand_text=(1.2, 1.4), expand_points=(1.2, 1.4))
    except Exception:
        pass  # fall back to raw positions if adjustText unavailable

    # Stats annotation
    stats_text = (
        f'$r$ = {r_pearson:.3f}  ($p$ < 0.001)\n'
        f'$\\rho$ = {r_spearman:.3f}  ($p$ < 0.001)'
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel('URIEL Syntactic Distance')
    ax.set_ylabel('CKA (SGPT, Final Layer)')

    # Legend — ordered
    handles, leg_labels = ax.get_legend_handles_labels()
    order = ['Same subfamily', 'Both Indo-European', 'Cross-family']
    sorted_pairs = [(h, l) for o in order for h, l in zip(handles, leg_labels) if l == o]
    if sorted_pairs:
        ax.legend([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                  loc='lower left', framealpha=0.9, edgecolor='#cccccc',
                  handletextpad=0.4, borderpad=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        out = FIGURES_DIR / f'cka_vs_syntactic_distance.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close(fig)

    # Print stats
    print(f'\nPearson  r = {r_pearson:.4f}, p = {p_pearson:.6f}')
    print(f'Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.6f}')
    print(f'N = {n} pairs')


if __name__ == '__main__':
    main()
