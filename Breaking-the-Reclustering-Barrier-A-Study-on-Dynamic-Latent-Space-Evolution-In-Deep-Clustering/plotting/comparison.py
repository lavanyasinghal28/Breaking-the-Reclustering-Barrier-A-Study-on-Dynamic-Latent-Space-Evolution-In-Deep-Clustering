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
    print(f"  Saved → {path}")


# ── Color scheme (consistent across all plots) ──────────────────────────────

COLORS = {
    'DCN (plain)':        '#d62728',   # red
    'DCN + Reclustering': '#ff7f0e',   # orange
    'DCN + BRB':          '#2ca02c',   # green
    'DCN + LDAR':         '#1f77b4',   # blue
    'DCN + LSAR':         '#9467bd',   # purple
    'DCN + FWR':          '#8c564b',   # brown
}


def _color_for(label):
    return COLORS.get(label, None)


def _add_intervention_lines(ax, epochs_max, interval, label='Intervention epoch'):
    """Draw vertical dashed lines at intervention epochs."""
    first = True
    for e in range(interval, epochs_max + 1, interval):
        ax.axvline(x=e, color='grey', ls='--', alpha=0.35,
                   label=label if first else "")
        first = False


# ── Single run plot ──────────────────────────────────────────────────────────

def plot_single_run(history, title_prefix, filename, interval=None):
    """Two-panel plot (loss + accuracy) for a single experiment."""
    epochs = history['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['loss'], color='#d62728', linewidth=2)
    ax1.set_title(f"{title_prefix} — Training Loss", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Joint Loss")
    ax1.grid(True, alpha=0.3)
    if interval:
        _add_intervention_lines(ax1, max(epochs), interval)

    ax2.plot(epochs, [a * 100 for a in history['accuracy']],
             color='#1f77b4', linewidth=2)
    ax2.set_title(f"{title_prefix} — Clustering Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    if interval:
        _add_intervention_lines(ax2, max(epochs), interval, 'Intervention')
        ax2.legend()

    fig.suptitle(title_prefix, fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, filename)


# ── Combined comparison ─────────────────────────────────────────────────────

def plot_comparison(results, reset_interval, filename="comparison_all.png",
                    title="Demonstrating the Reclustering Barrier on MNIST (DCN)"):
    """2-panel loss + accuracy comparison of all modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for label, history in results.items():
        c = _color_for(label)
        ax1.plot(history['epoch'], history['loss'], label=label, color=c, linewidth=2)
        ax2.plot(history['epoch'], [a * 100 for a in history['accuracy']],
                 label=label, color=c, linewidth=2)

    first_label = next(iter(results.keys()))
    max_ep = max(results[first_label]['epoch'])
    _add_intervention_lines(ax1, max_ep, reset_interval)
    _add_intervention_lines(ax2, max_ep, reset_interval)

    ax1.set_title("Training Loss Comparison", fontsize=14)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Joint Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("Clustering Accuracy Comparison", fontsize=14)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(title,
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, filename)


def plot_accuracy_only_comparison(results, reset_interval):
    """Single-panel accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in results.items():
        ax.plot(history['epoch'],
                [a * 100 for a in history['accuracy']],
                label=label, color=_color_for(label), linewidth=2.5)
    first_label = next(iter(results.keys()))
    _add_intervention_lines(ax, max(results[first_label]['epoch']), reset_interval)
    ax.set_title("Reclustering Barrier — Accuracy Comparison (MNIST, DCN)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Clustering Accuracy (%)", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "comparison_accuracy.png")


def plot_loss_only_comparison(results, reset_interval):
    """Single-panel loss comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in results.items():
        ax.plot(history['epoch'], history['loss'],
                label=label, color=_color_for(label), linewidth=2.5)
    first_label = next(iter(results.keys()))
    _add_intervention_lines(ax, max(results[first_label]['epoch']), reset_interval)
    ax.set_title("Reclustering Barrier — Loss Comparison (MNIST, DCN)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Joint Loss", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "comparison_loss.png")


# ── Cluster quality ──────────────────────────────────────────────────────────

def plot_cluster_quality_comparison(results, reset_interval):
    """3-panel: Intra-CD, Inter-CD, Clustering Loss."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    first_label = next(iter(results.keys()))
    max_ep = max(results[first_label]['epoch'])

    for label, history in results.items():
        c = _color_for(label)
        ax1.plot(history['epoch'], history['intra_cd'],  label=label, color=c, linewidth=2)
        ax2.plot(history['epoch'], history['inter_cd'],  label=label, color=c, linewidth=2)
        ax3.plot(history['epoch'], history['cluster_loss'], label=label, color=c, linewidth=2)

    for ax in (ax1, ax2, ax3):
        _add_intervention_lines(ax, max_ep, reset_interval, 'Intervention')

    ax1.set_title("Intra-Cluster Distance (lower = tighter)", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Intra-CD")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("Inter-Cluster Distance (higher = more separated)", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Inter-CD")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3.set_title("Clustering Loss (lower = better)", fontsize=13)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("CL")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    fig.suptitle("Cluster Quality Metrics Comparison (MNIST, DCN)",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, "comparison_cluster_quality.png")


def plot_intra_cd_comparison(results, reset_interval):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in results.items():
        ax.plot(history['epoch'], history['intra_cd'],
                label=label, color=_color_for(label), linewidth=2.5)
    first_label = next(iter(results.keys()))
    _add_intervention_lines(ax, max(results[first_label]['epoch']), reset_interval)
    ax.set_title("Intra-Cluster Distance (lower = tighter clusters)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean distance to assigned center", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "comparison_intra_cd.png")


def plot_inter_cd_comparison(results, reset_interval):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in results.items():
        ax.plot(history['epoch'], history['inter_cd'],
                label=label, color=_color_for(label), linewidth=2.5)
    first_label = next(iter(results.keys()))
    _add_intervention_lines(ax, max(results[first_label]['epoch']), reset_interval)
    ax.set_title("Inter-Cluster Distance (higher = better separation)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean pairwise center distance", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "comparison_inter_cd.png")


def plot_cluster_loss_comparison(results, reset_interval):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in results.items():
        ax.plot(history['epoch'], history['cluster_loss'],
                label=label, color=_color_for(label), linewidth=2.5)
    first_label = next(iter(results.keys()))
    _add_intervention_lines(ax, max(results[first_label]['epoch']), reset_interval)
    ax.set_title("Clustering Loss (lower = better)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean squared distance to assigned center", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "comparison_cluster_loss.png")


def plot_pairwise_method_vs_plain(method_label, plain_history, method_history,
                                  reset_interval, filename):
    """Two-panel comparison (loss + accuracy) of plain DCN vs one reset method."""
    results = {
        'DCN (plain)': plain_history,
        method_label: method_history,
    }
    plot_comparison(
        results,
        reset_interval,
        filename=filename,
        title=f"Plain DCN vs {method_label} (MNIST, DCN)",
    )


def plot_reset_methods_comparison(reset_results, reset_interval,
                                  filename="comparison_reset_methods.png"):
    """Compare BRB, LDAR, LSAR, and FWR together (loss + accuracy)."""
    plot_comparison(
        reset_results,
        reset_interval,
        filename=filename,
        title="Reset Method Comparison: BRB vs LDAR vs LSAR vs FWR",
    )
