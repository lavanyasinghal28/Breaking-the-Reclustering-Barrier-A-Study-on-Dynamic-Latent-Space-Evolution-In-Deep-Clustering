import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PLOT_DIR


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path}")


def _add_intervention_lines(ax, epochs_max, interval, label='BRB epoch'):
    first = True
    for e in range(interval, epochs_max + 1, interval):
        ax.axvline(x=e, color='grey', ls='--', alpha=0.35,
                   label=label if first else "")
        first = False


def plot_non_geometric_single_run(history, title_prefix, filename, interval=None):
    epochs = history['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['loss'], color='#8c564b', linewidth=2)
    ax1.set_title(f"{title_prefix} - CE Loss", fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    if interval:
        _add_intervention_lines(ax1, max(epochs), interval)

    ax2.plot(epochs, [a * 100 for a in history['accuracy']], color='#17becf', linewidth=2)
    ax2.set_title(f"{title_prefix} - Classification Accuracy", fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    if interval:
        _add_intervention_lines(ax2, max(epochs), interval)
        ax2.legend()

    fig.suptitle(title_prefix, fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, filename)


def plot_non_geometric_comparison(results, reset_interval,
                                 filename='comparison_non_geometric.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'Non-Geometric (plain)': '#d62728',
        'Non-Geometric + BRB': '#2ca02c',
    }

    for label, history in results.items():
        c = colors.get(label)
        ax1.plot(history['epoch'], history['loss'], label=label, color=c, linewidth=2.3)
        ax2.plot(history['epoch'], [a * 100 for a in history['accuracy']],
                 label=label, color=c, linewidth=2.3)

    first_label = next(iter(results.keys()))
    max_ep = max(results[first_label]['epoch'])
    _add_intervention_lines(ax1, max_ep, reset_interval)
    _add_intervention_lines(ax2, max_ep, reset_interval)

    ax1.set_title('Non-Geometric Classification Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Non-Geometric Classification Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Non-Geometric Classification: Plain vs BRB',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, filename)
