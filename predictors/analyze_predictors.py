#!/usr/bin/env python3
"""
Predictors of alignment (Appendix D, Table 7).

Correlates last-layer SGPT matched CKA (FLORES, from ../cka/results/main) with
four URIEL typological distances (lang2vec cosine distances, hard-coded below)
and with the mean FLORES NLL of the two models (results/nll_per_lang.csv from
compute_nll.py).  Writes results/correlation_summary.csv (Table 7),
results/pairwise_data.csv (input to regression_uriel_nll.py) and one scatter
plot per predictor under figures/.

Usage:
  python analyze_predictors.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

LANGUAGES = [
    'eng_latn', 'arb_arab', 'deu_latn', 'fra_latn', 'hin_deva',
    'jpn_jpan', 'rus_cyrl', 'spa_latn', 'zho_hans',
]

# Map FLORES codes to ISO 639-3 for URIEL+
FLORES_TO_ISO = {
    'eng_latn': 'eng', 'arb_arab': 'arb', 'deu_latn': 'deu',
    'fra_latn': 'fra', 'hin_deva': 'hin', 'jpn_jpan': 'jpn',
    'rus_cyrl': 'rus', 'spa_latn': 'spa', 'zho_hans': 'zho',
}

LANG_LABELS = {
    'eng_latn': 'English', 'arb_arab': 'Arabic', 'deu_latn': 'German',
    'fra_latn': 'French', 'hin_deva': 'Hindi', 'jpn_jpan': 'Japanese',
    'rus_cyrl': 'Russian', 'spa_latn': 'Spanish', 'zho_hans': 'Chinese',
}

CKA_RESULTS = Path(__file__).resolve().parent.parent / 'cka' / 'results' / 'main'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'


# ── Step 1: Load CKA ───────────────────────────────────────────────────────

def load_cka_values() -> dict[tuple[str, str], float]:
    """Load SGPT final-layer matched CKA from the FLORES CKA results."""
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
        # Extract language codes from model names
        model_a = row['model_a']  # e.g. goldfish-models/eng_latn_1000mb
        model_b = row['model_b']
        lang_a = model_a.split('/')[1].replace('_1000mb', '')
        lang_b = model_b.split('/')[1].replace('_1000mb', '')
        cka[(lang_a, lang_b)] = float(row['cka_matched'])
    print(f'Loaded CKA for {len(cka)} pairs')
    return cka


# ── Step 2: URIEL distances (hardcoded from lang2vec) ────────────────────────
# Cosine distances computed from URIEL feature vectors via lang2vec v1.1.2.
# Feature sets: fam (genetic), syntax_knn (syntactic), geo (geographic),
# inventory_knn (inventory).
# Keys use ISO 639-3 codes; mapped to FLORES codes below.

_URIEL_RAW = {
    'genetic': {
        ('eng','arb'): 1.000000, ('eng','deu'): 0.455669, ('eng','fra'): 0.903775,
        ('eng','hin'): 0.874012, ('eng','jpn'): 1.000000, ('eng','rus'): 0.833333,
        ('eng','spa'): 0.903775, ('eng','zho'): 1.000000, ('arb','deu'): 1.000000,
        ('arb','fra'): 1.000000, ('arb','hin'): 1.000000, ('arb','jpn'): 1.000000,
        ('arb','rus'): 1.000000, ('arb','spa'): 1.000000, ('arb','zho'): 1.000000,
        ('deu','fra'): 0.882149, ('deu','hin'): 0.845697, ('deu','jpn'): 1.000000,
        ('deu','rus'): 0.795876, ('deu','spa'): 0.882149, ('deu','zho'): 1.000000,
        ('fra','hin'): 0.890891, ('fra','jpn'): 1.000000, ('fra','rus'): 0.855662,
        ('fra','spa'): 0.250000, ('fra','zho'): 1.000000, ('hin','jpn'): 1.000000,
        ('hin','rus'): 0.811018, ('hin','spa'): 0.890891, ('hin','zho'): 1.000000,
        ('jpn','rus'): 1.000000, ('jpn','spa'): 1.000000, ('jpn','zho'): 1.000000,
        ('rus','spa'): 0.855662, ('rus','zho'): 1.000000, ('spa','zho'): 1.000000,
    },
    'syntactic': {
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
    },
    'geographic': {
        ('eng','arb'): 0.041104, ('eng','deu'): 0.002358, ('eng','fra'): 0.000744,
        ('eng','hin'): 0.089030, ('eng','jpn'): 0.143530, ('eng','rus'): 0.041865,
        ('eng','spa'): 0.004072, ('eng','zho'): 0.115831, ('arb','deu'): 0.024994,
        ('arb','fra'): 0.035020, ('arb','hin'): 0.021957, ('arb','jpn'): 0.119802,
        ('arb','rus'): 0.027313, ('arb','spa'): 0.036482, ('arb','zho'): 0.071586,
        ('deu','fra'): 0.001265, ('deu','hin'): 0.069398, ('deu','jpn'): 0.137879,
        ('deu','rus'): 0.032940, ('deu','spa'): 0.004135, ('deu','zho'): 0.103549,
        ('fra','hin'): 0.085816, ('fra','jpn'): 0.151448, ('fra','rus'): 0.044091,
        ('fra','spa'): 0.001612, ('fra','zho'): 0.119643, ('hin','jpn'): 0.059597,
        ('hin','rus'): 0.021122, ('hin','spa'): 0.094479, ('hin','zho'): 0.022601,
        ('jpn','rus'): 0.051657, ('jpn','spa'): 0.171703, ('jpn','zho'): 0.010528,
        ('rus','spa'): 0.057190, ('rus','zho'): 0.025769, ('spa','zho'): 0.136779,
    },
    'inventory': {
        ('eng','arb'): 0.268012, ('eng','deu'): 0.237230, ('eng','fra'): 0.259132,
        ('eng','hin'): 0.290909, ('eng','jpn'): 0.346115, ('eng','rus'): 0.352504,
        ('eng','spa'): 0.361821, ('eng','zho'): 0.302723, ('arb','deu'): 0.353502,
        ('arb','fra'): 0.313197, ('arb','hin'): 0.248749, ('arb','jpn'): 0.384210,
        ('arb','rus'): 0.363005, ('arb','spa'): 0.348024, ('arb','zho'): 0.358467,
        ('deu','fra'): 0.164694, ('deu','hin'): 0.313507, ('deu','jpn'): 0.339599,
        ('deu','rus'): 0.296647, ('deu','spa'): 0.450195, ('deu','zho'): 0.326425,
        ('fra','hin'): 0.277654, ('fra','jpn'): 0.333891, ('fra','rus'): 0.293286,
        ('fra','spa'): 0.373109, ('fra','zho'): 0.289689, ('hin','jpn'): 0.394551,
        ('hin','rus'): 0.329379, ('hin','spa'): 0.430197, ('hin','zho'): 0.321072,
        ('jpn','rus'): 0.322355, ('jpn','spa'): 0.392823, ('jpn','zho'): 0.340088,
        ('rus','spa'): 0.304275, ('rus','zho'): 0.369874, ('spa','zho'): 0.424945,
    },
}

# ISO 639-3 -> FLORES code
_ISO_TO_FLORES = {v: k for k, v in FLORES_TO_ISO.items()}


def get_uriel_distances() -> dict[str, dict[tuple[str, str], float]]:
    """Return hardcoded URIEL distances keyed by FLORES language codes."""
    all_distances = {}
    for dtype, raw in _URIEL_RAW.items():
        distances = {}
        for (iso_a, iso_b), d in raw.items():
            fa = _ISO_TO_FLORES.get(iso_a)
            fb = _ISO_TO_FLORES.get(iso_b)
            if fa and fb:
                distances[(fa, fb)] = d
        all_distances[dtype] = distances
        print(f'  {dtype}: {len(distances)} pairs')
    return all_distances


# ── Step 3: Load NLL ─────────────────────────────────────────────────────────

def load_nll() -> dict[str, float]:
    """Load per-language mean NLL from results/nll_per_lang.csv."""
    nll_path = RESULTS_DIR / 'nll_per_lang.csv'
    if not nll_path.exists():
        raise FileNotFoundError(
            f'{nll_path} not found. Run compute_nll.py first '
            f'(bash run_nll.sh).'
        )
    df = pd.read_csv(nll_path)
    nll = {row['lang']: float(row['mean_nll']) for _, row in df.iterrows()}
    print(f'Loaded NLL for {len(nll)} languages')
    return nll


# ── Step 4: Correlation analysis ─────────────────────────────────────────────

def pair_label(lang_a, lang_b):
    return f'{LANG_LABELS[lang_a][:3]}-{LANG_LABELS[lang_b][:3]}'


def correlate_and_plot(
    x_vals, y_vals, labels,
    x_label, y_label, title, out_path,
):
    """Scatter plot with regression line + Pearson/Spearman stats."""
    x = np.array(x_vals)
    y = np.array(y_vals)

    # Remove NaN
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    labels = [l for l, v in zip(labels, valid) if v]

    if len(x) < 3:
        print(f'  Skipping {title}: only {len(x)} valid points')
        return None

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=30, alpha=0.7, zorder=3)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7, linewidth=1.5)

    # Annotate points
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)

    stats_text = (f'Pearson r={r_pearson:.3f} (p={p_pearson:.3f})\n'
                  f'Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})')
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')

    return {
        'pearson_r': r_pearson, 'pearson_p': p_pearson,
        'spearman_r': r_spearman, 'spearman_p': p_spearman,
        'n': len(x),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load CKA
    print('Loading CKA values...')
    cka = load_cka_values()

    # Normalise pair keys: ensure (a, b) where a < b alphabetically
    def norm_key(a, b):
        return (a, b) if a < b else (b, a)

    cka_norm = {}
    for (a, b), v in cka.items():
        cka_norm[norm_key(a, b)] = v

    # All pairs
    pairs = sorted(cka_norm.keys())
    cka_vals = [cka_norm[p] for p in pairs]
    pair_labels = [pair_label(*p) for p in pairs]

    # ── URIEL distances ────────────────────────────────────────────────────
    print('\nLoading URIEL distances...')
    uriel_dists = get_uriel_distances()

    summary_rows = []
    for dtype, distances in uriel_dists.items():
        dist_vals = []
        cka_for_dist = []
        labels_for_dist = []
        for p in pairs:
            d = distances.get(p, distances.get((p[1], p[0]), np.nan))
            if not np.isnan(d) and p in cka_norm:
                dist_vals.append(d)
                cka_for_dist.append(cka_norm[p])
                labels_for_dist.append(pair_label(*p))

        print(f'\n--- URIEL+ {dtype} distance vs CKA ---')
        result = correlate_and_plot(
            dist_vals, cka_for_dist, labels_for_dist,
            x_label=f'URIEL+ {dtype} distance',
            y_label='CKA (SGPT, final layer)',
            title=f'CKA vs URIEL+ {dtype} distance',
            out_path=FIGURES_DIR / f'cka_vs_uriel_{dtype}.png',
        )
        if result:
            result['predictor'] = f'uriel_{dtype}'
            summary_rows.append(result)

    # ── NLL analysis ────────────────────────────────────────────────────────
    print('\nLoading NLL values...')
    try:
        nll = load_nll()

        # For each pair (a, b), use mean NLL = (NLL_a + NLL_b) / 2
        nll_vals = []
        cka_for_nll = []
        labels_for_nll = []
        for p in pairs:
            if p[0] in nll and p[1] in nll:
                mean_nll = (nll[p[0]] + nll[p[1]]) / 2
                nll_vals.append(mean_nll)
                cka_for_nll.append(cka_norm[p])
                labels_for_nll.append(pair_label(*p))

        print(f'\n--- Mean NLL vs CKA ---')
        result = correlate_and_plot(
            nll_vals, cka_for_nll, labels_for_nll,
            x_label='Mean NLL (avg of both models)',
            y_label='CKA (SGPT, final layer)',
            title='CKA vs Mean Sentence NLL',
            out_path=FIGURES_DIR / 'cka_vs_mean_nll.png',
        )
        if result:
            result['predictor'] = 'mean_nll'
            summary_rows.append(result)

        # Also try: |NLL_a - NLL_b| (NLL difference)
        nll_diff_vals = []
        cka_for_diff = []
        labels_for_diff = []
        for p in pairs:
            if p[0] in nll and p[1] in nll:
                nll_diff_vals.append(abs(nll[p[0]] - nll[p[1]]))
                cka_for_diff.append(cka_norm[p])
                labels_for_diff.append(pair_label(*p))

        print(f'\n--- |NLL_a - NLL_b| vs CKA ---')
        result = correlate_and_plot(
            nll_diff_vals, cka_for_diff, labels_for_diff,
            x_label='|NLL difference| between models',
            y_label='CKA (SGPT, final layer)',
            title='CKA vs NLL Difference',
            out_path=FIGURES_DIR / 'cka_vs_nll_diff.png',
        )
        if result:
            result['predictor'] = 'nll_diff'
            summary_rows.append(result)

    except FileNotFoundError as e:
        print(f'\nSkipping NLL analysis: {e}')

    # ── Summary table ───────────────────────────────────────────────────────
    if summary_rows:
        print('\n' + '=' * 70)
        print('SUMMARY: Correlation between predictors and CKA alignment')
        print('=' * 70)
        print(f'{"Predictor":<20} {"Pearson r":>10} {"p-value":>10} '
              f'{"Spearman ρ":>12} {"p-value":>10} {"N":>5}')
        print('-' * 70)
        for row in summary_rows:
            print(f'{row["predictor"]:<20} {row["pearson_r"]:>10.3f} '
                  f'{row["pearson_p"]:>10.4f} {row["spearman_r"]:>12.3f} '
                  f'{row["spearman_p"]:>10.4f} {row["n"]:>5}')

        # Save summary CSV
        summary_path = RESULTS_DIR / 'correlation_summary.csv'
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'predictor', 'pearson_r', 'pearson_p',
                'spearman_r', 'spearman_p', 'n',
            ])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'\nSummary saved to: {summary_path}')

    # ── Save CKA + distance combined table ──────────────────────────────────
    combined_path = RESULTS_DIR / 'pairwise_data.csv'
    with open(combined_path, 'w', newline='') as f:
        fieldnames = ['lang_a', 'lang_b', 'cka_sgpt_final']
        for dtype in uriel_dists:
            fieldnames.append(f'uriel_{dtype}')
        fieldnames.extend(['nll_a', 'nll_b', 'mean_nll', 'nll_diff'])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        nll = {}
        try:
            nll = load_nll()
        except FileNotFoundError:
            pass

        for p in pairs:
            row = {
                'lang_a': p[0], 'lang_b': p[1],
                'cka_sgpt_final': cka_norm[p],
            }
            for dtype, distances in uriel_dists.items():
                d = distances.get(p, distances.get((p[1], p[0]), ''))
                row[f'uriel_{dtype}'] = d
            if p[0] in nll and p[1] in nll:
                row['nll_a'] = nll[p[0]]
                row['nll_b'] = nll[p[1]]
                row['mean_nll'] = (nll[p[0]] + nll[p[1]]) / 2
                row['nll_diff'] = abs(nll[p[0]] - nll[p[1]])
            writer.writerow(row)
    print(f'Combined data saved to: {combined_path}')


if __name__ == '__main__':
    main()
