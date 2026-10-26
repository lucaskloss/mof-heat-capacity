"""Generate schematic and illustrative attention maps for the report.

Run with: python docs/report/generate_transformer_figures.py
The attention weights use synthetic features, not a trained model.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mof-matplotlib'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def save(fig, name):
    directory = Path(__file__).resolve().parent / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        target = directory / f'{name}.{suffix}'
        fig.savefig(target, dpi=250, bbox_inches='tight')
        print(f'Wrote {target}')
    plt.close(fig)


def attention_flow():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    boxes = [
        ('Input tokens', r'$X$', '#dcecf8'),
        ('Learned projections', r'$Q=XW_Q$' '\n' r'$K=XW_K,\quad V=XW_V$', '#dceee4'),
        ('Compatibility scores', r'$S=QK^{\mathrm{T}}/\sqrt{d_k}+M$', '#fff0ce'),
        ('Normalize each row', r'$A=\mathrm{softmax}_{\mathrm{row}}(S)$', '#eee4f5'),
        ('Mix information', r'$Y=AV$', '#dcecf8'),
    ]
    for i, (title, formula, color) in enumerate(boxes):
        x = i * 2.6
        ax.add_patch(FancyBboxPatch((x, 0), 2.3, 1.25,
                     boxstyle='round,pad=0.04', facecolor=color, edgecolor='#69818f'))
        ax.text(x + 1.15, 1.75, title, ha='center', fontsize=10, weight='bold')
        ax.text(x + 1.15, 0.625, formula, ha='center', va='center', fontsize=11)
        if i < len(boxes) - 1:
            ax.annotate('', xy=(x + 2.58, 0.625), xytext=(x + 2.33, 0.625),
                        arrowprops={'arrowstyle': '->', 'color': '#455563', 'lw': 1.5})
    ax.text(6.35, -0.6,
            'Each output token is a weighted sum of value vectors; the weights depend on the input.',
            ha='center', fontsize=10, color='#455563')
    ax.set(xlim=(-0.2, 13.0), ylim=(-0.9, 2.15))
    ax.axis('off')
    fig.tight_layout()
    save(fig, 'transformer-attention-flow')


def attention_maps():
    rng = np.random.default_rng(12)
    tokens = rng.normal(size=(4, 3))
    queries = tokens @ rng.normal(size=(3, 2))
    keys = tokens @ rng.normal(size=(3, 2))
    scores = queries @ keys.T / np.sqrt(2)
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.3), layout='constrained')
    for ax, causal in zip(axes, (False, True)):
        masked_scores = scores.copy()
        if causal:
            masked_scores[np.triu_indices(4, 1)] = -np.inf
        weights = np.exp(masked_scores - masked_scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        if not np.allclose(weights.sum(axis=1), 1):
            raise ValueError('Attention rows must sum to one.')
        im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{weights[i, j]:.2f}', ha='center', va='center',
                        color='white' if weights[i, j] > 0.55 else '#20333e', fontsize=10)
        ax.set(xticks=range(4), yticks=range(4),
               xticklabels=[f'$t_{i}$' for i in range(1, 5)],
               yticklabels=[f'$t_{i}$' for i in range(1, 5)],
               xlabel='Key / value token', ylabel='Query token',
               title='Causal attention' if causal else 'Unmasked attention')
    fig.colorbar(im, ax=axes, shrink=0.8, label='Attention weight')
    save(fig, 'transformer-attention-maps')


def main():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    attention_flow()
    attention_maps()


if __name__ == '__main__':
    main()
