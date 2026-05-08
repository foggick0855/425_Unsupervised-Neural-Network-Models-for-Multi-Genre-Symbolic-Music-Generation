"""
Unified performance comparison table (Tasks 1–3 + baselines).
Human scores excluded (Task 4 skipped).

Usage:
  python src/evaluation/compare_all.py
Outputs:
  outputs/plots/comparison_table.json
  outputs/plots/comparison_table.png  (bar chart, dpi=300)
  outputs/plots/comparison_table.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import evaluate_directory, aggregate

OUTPUTS  = Path(__file__).parent.parent.parent / 'outputs'
GEN_DIR  = OUTPUTS / 'generated_midis'
PLOT_DIR = OUTPUTS / 'plots'

# ---------------------------------------------------------------------------
# 1. Reference MIDI (first MAESTRO val file)
# ---------------------------------------------------------------------------
def _reference():
    try:
        from src.preprocessing.midi_parser import get_split
        val = get_split('validation')
        return val[0]['path'] if val else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 2. Evaluate generated MIDIs for each model
# ---------------------------------------------------------------------------
def eval_dir(subdir):
    path = GEN_DIR / subdir
    if not path.exists():
        return {}
    results = evaluate_directory(str(path), _reference())
    return aggregate(results) if results else {}

# ---------------------------------------------------------------------------
# 3. Pull best val loss from model checkpoints
# ---------------------------------------------------------------------------
def _ckpt_val_loss(name):
    path = OUTPUTS / 'models' / name
    if not path.exists():
        return None
    try:
        ckpt = torch.load(str(path), map_location='cpu')
        return float(ckpt.get('val_loss', float('nan')))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 4. Build table
# ---------------------------------------------------------------------------
def build_table():
    ref = _reference()

    # Baselines from saved JSON (already uses pitch_histogram_distance)
    baseline_path = PLOT_DIR / 'baseline_metrics.json'
    baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    # Model evaluations (fresh — uses corrected pitch_histogram_distance)
    ae_metrics  = eval_dir('ae')
    vae_metrics = eval_dir('vae')
    tr_metrics  = eval_dir('transformer')

    # Losses from checkpoints
    ae_loss  = _ckpt_val_loss('ae_best.pth')
    vae_loss = _ckpt_val_loss('vae_best.pth')
    tr_loss  = _ckpt_val_loss('transformer_best.pth')

    # Perplexity from report
    ppl_path = PLOT_DIR / 'perplexity_report.json'
    best_ppl = None
    if ppl_path.exists():
        ppl_data = json.loads(ppl_path.read_text())
        best_ppl = ppl_data.get('best_perplexity')

    def _fmt(v, decimals=4):
        return round(v, decimals) if v is not None else None

    rows = {}

    for k, m in baselines.items():
        rows[k] = {
            'val_loss':                 None,
            'perplexity':               None,
            'rhythm_diversity':         _fmt(m.get('rhythm_diversity')),
            'repetition_ratio':         _fmt(m.get('repetition_ratio')),
            'pitch_histogram_distance': _fmt(m.get('pitch_histogram_distance')),
            'genre_control':            'Weak' if 'Markov' in k else 'None',
        }

    rows['Task 1: AE'] = {
        'val_loss':                 _fmt(ae_loss),
        'perplexity':               None,
        'rhythm_diversity':         _fmt(ae_metrics.get('rhythm_diversity')),
        'repetition_ratio':         _fmt(ae_metrics.get('repetition_ratio')),
        'pitch_histogram_distance': _fmt(ae_metrics.get('pitch_histogram_distance')),
        'genre_control':            'Single Genre',
    }

    rows['Task 2: VAE'] = {
        'val_loss':                 _fmt(vae_loss),
        'perplexity':               None,
        'rhythm_diversity':         _fmt(vae_metrics.get('rhythm_diversity')),
        'repetition_ratio':         _fmt(vae_metrics.get('repetition_ratio')),
        'pitch_histogram_distance': _fmt(vae_metrics.get('pitch_histogram_distance')),
        'genre_control':            'Moderate',
    }

    rows['Task 3: Transformer'] = {
        'val_loss':                 _fmt(tr_loss),
        'perplexity':               _fmt(best_ppl, 2),
        'rhythm_diversity':         _fmt(tr_metrics.get('rhythm_diversity')),
        'repetition_ratio':         _fmt(tr_metrics.get('repetition_ratio')),
        'pitch_histogram_distance': _fmt(tr_metrics.get('pitch_histogram_distance')),
        'genre_control':            'Strong',
    }

    return rows

# ---------------------------------------------------------------------------
# 5. Print table
# ---------------------------------------------------------------------------
def print_table(rows):
    cols = ['val_loss', 'perplexity', 'rhythm_diversity',
            'repetition_ratio', 'pitch_histogram_distance', 'genre_control']
    headers = ['Model', 'Val Loss', 'Perplexity', 'Rhythm Div.',
               'Repetition R.', 'Pitch Hist. Dist.', 'Genre Control']

    col_w  = [max(len(headers[0]), max(len(k) for k in rows)) + 2]
    col_w += [max(len(h), 10) + 2 for h in headers[1:]]

    def fmt(v):
        if v is None:
            return '-'
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)

    sep = '+' + '+'.join('-' * w for w in col_w) + '+'
    print(sep)
    header_row = '|' + '|'.join(
        f' {h:<{col_w[i]-1}}' for i, h in enumerate(headers)
    ) + '|'
    print(header_row)
    print(sep)
    for name, m in rows.items():
        vals = [name] + [fmt(m.get(c)) for c in cols]
        row = '|' + '|'.join(
            f' {v:<{col_w[i]-1}}' for i, v in enumerate(vals)
        ) + '|'
        print(row)
    print(sep)

# ---------------------------------------------------------------------------
# 6. Bar chart
# ---------------------------------------------------------------------------
def plot_table(rows):
    models = list(rows.keys())
    metrics = [
        ('rhythm_diversity',         'Rhythm Diversity\n(higher = better)',     1.0),
        ('repetition_ratio',         'Repetition Ratio\n(0.1–0.5 coherent)',    1.0),
        ('pitch_histogram_distance', 'Pitch Hist. Distance\n(lower = better)',  2.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.tab10.colors[:len(models)]

    for ax, (metric, label, ylim) in zip(axes, metrics):
        values = [rows[m].get(metric) or 0 for m in models]
        bars = ax.bar(range(len(models)), values, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=7)
        ax.set_ylim(0, ylim)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel('')

    fig.suptitle('Performance Comparison (Tasks 1–3 vs Baselines)', fontsize=11)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / 'comparison_table.png', dpi=300)
    fig.savefig(PLOT_DIR / 'comparison_table.pdf')
    plt.close(fig)
    print(f'Bar chart -> {PLOT_DIR / "comparison_table.png"}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Building comparison table...')
    rows = build_table()

    print_table(rows)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / 'comparison_table.json'
    out.write_text(json.dumps(rows, indent=2))
    print(f'\nTable JSON -> {out}')

    plot_table(rows)

    # Also refresh evaluation_report.json with corrected metric key
    fresh = {
        'AE (Task 1)':          aggregate(evaluate_directory(str(GEN_DIR / 'ae'),          _reference()) or [{}]),
        'VAE (Task 2)':         aggregate(evaluate_directory(str(GEN_DIR / 'vae'),         _reference()) or [{}]),
        'Transformer (Task 3)': aggregate(evaluate_directory(str(GEN_DIR / 'transformer'), _reference()) or [{}]),
    }
    (PLOT_DIR / 'evaluation_report.json').write_text(json.dumps(fresh, indent=2))
    print(f'Evaluation report -> {PLOT_DIR / "evaluation_report.json"}')
