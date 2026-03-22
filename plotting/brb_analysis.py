import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .comparison import _save


def plot_d1d2_histogram(brb_snapshots, reset_interval):
    """
    Overlaid d1/d2 distance-ratio histograms before & after each BRB event.
    """
    befores = brb_snapshots['d1d2_before']
    afters  = brb_snapshots['d1d2_after']
    if not befores:
        return

    bins = np.linspace(0, 1, 60)

    # ── Single-event plot (first BRB) ──
    ep_b, r_before = befores[0]
    ep_a, r_after  = afters[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r_before, bins=bins, alpha=0.55, color='#d62728',
            label=f'Before BRB (epoch {ep_b})', density=True,
            edgecolor='white', linewidth=0.4)
    ax.hist(r_after, bins=bins, alpha=0.55, color='#2ca02c',
            label=f'After  BRB (epoch {ep_a})', density=True,
            edgecolor='white', linewidth=0.4)
    ax.set_xlabel('d₁ / d₂  (nearest / 2nd-nearest centroid distance)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('BRB Increases Assignment Uncertainty\n'
                 '(d₁/d₂ shifts toward 1 after soft reset)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=np.median(r_before), ls='--', color='#d62728', lw=1.5,
               label=f'Median before = {np.median(r_before):.3f}')
    ax.axvline(x=np.median(r_after),  ls='--', color='#2ca02c', lw=1.5,
               label=f'Median after  = {np.median(r_after):.3f}')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, 'brb_d1d2_histogram_first.png')

    # ── Multi-event summary ──
    if len(befores) >= 2:
        n_events = min(len(befores), 4)
        fig, axes = plt.subplots(1, n_events, figsize=(6 * n_events, 5),
                                 sharey=True)
        if n_events == 1:
            axes = [axes]
        for idx in range(n_events):
            ep_b, rb = befores[idx]
            ep_a, ra = afters[idx]
            ax = axes[idx]
            ax.hist(rb, bins=bins, alpha=0.55, color='#d62728', density=True,
                    label='Before', edgecolor='white', linewidth=0.3)
            ax.hist(ra, bins=bins, alpha=0.55, color='#2ca02c', density=True,
                    label='After', edgecolor='white', linewidth=0.3)
            ax.set_title(f'BRB @ epoch {ep_b}', fontsize=12)
            ax.set_xlabel('d₁/d₂')
            if idx == 0:
                ax.set_ylabel('Density')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        fig.suptitle('d₁/d₂ Distance-Ratio Histograms Across BRB Events',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        _save(fig, 'brb_d1d2_histogram_multi.png')


def plot_embedding_panels(brb_snapshots, method='pca'):
    """
    3-panel scatter plot: Pre-BRB → During BRB → Post-BRB.
    Centroids marked with black ★.
    """
    snaps = brb_snapshots['embeddings']
    if len(snaps) < 3:
        return

    chosen = [snaps[0], snaps[len(snaps) // 2], snaps[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    cmap = plt.cm.tab10

    for ax, (ep, tag, embs, labs, centers) in zip(axes, chosen):
        combined = np.vstack([embs, centers])
        if method == 'tsne':
            proj = TSNE(n_components=2, perplexity=30, random_state=42,
                        init='pca', learning_rate='auto')
            combined_2d = proj.fit_transform(combined)
        else:
            proj = PCA(n_components=2)
            combined_2d = proj.fit_transform(combined)

        embs_2d    = combined_2d[:len(embs)]
        centers_2d = combined_2d[len(embs):]

        ax.scatter(embs_2d[:, 0], embs_2d[:, 1],
                   c=labs, cmap=cmap, s=4, alpha=0.5)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                   marker='*', s=250, c='black', edgecolors='white',
                   linewidths=1.0, zorder=10, label='Centroids')
        ax.set_title(tag, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=9)

    proj_label = 'PCA' if method == 'pca' else 't-SNE'
    fig.suptitle(f'Embedding Space Evolution Across BRB ({proj_label} projection)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, f'brb_embeddings_{method}.png')


def plot_nmi_curve(history_brb, reset_interval):
    """NMI vs epoch with vertical lines at BRB events."""
    epochs = history_brb['epoch']
    nmi    = history_brb['nmi']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, nmi, color='#9467bd', linewidth=2.5, label='NMI')

    first = True
    for e in range(reset_interval, max(epochs) + 1, reset_interval):
        ax.axvline(x=e, color='#d62728', ls='--', lw=1.3, alpha=0.6,
                   label='BRB event' if first else '')
        first = False

    ax.set_title('NMI vs Epoch — DCN + BRB\n'
                 '(BRB improves NMI after each reset)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalised Mutual Information', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'brb_nmi_curve.png')


def plot_embedding_and_nmi(brb_snapshots, history_brb, reset_interval,
                           method='pca'):
    """
    Combined 4-panel: [0-2] embedding scatter, [3] NMI curve.
    """
    snaps = brb_snapshots['embeddings']
    if len(snaps) < 3:
        return

    chosen = [snaps[0], snaps[len(snaps) // 2], snaps[-1]]
    cmap = plt.cm.tab10

    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])

    for idx, (ep, tag, embs, labs, centers) in enumerate(chosen):
        ax = fig.add_subplot(gs[0, idx])
        combined = np.vstack([embs, centers])
        if method == 'tsne':
            proj = TSNE(n_components=2, perplexity=30, random_state=42,
                        init='pca', learning_rate='auto')
            c2d = proj.fit_transform(combined)
        else:
            proj = PCA(n_components=2)
            c2d = proj.fit_transform(combined)
        e2d, ct2d = c2d[:len(embs)], c2d[len(embs):]
        ax.scatter(e2d[:, 0], e2d[:, 1], c=labs, cmap=cmap, s=4, alpha=0.5)
        ax.scatter(ct2d[:, 0], ct2d[:, 1], marker='*', s=250,
                   c='black', edgecolors='white', linewidths=1, zorder=10,
                   label='Centroids')
        ax.set_title(tag, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8)

    # NMI panel
    ax_nmi = fig.add_subplot(gs[0, 3])
    epochs = history_brb['epoch']
    ax_nmi.plot(epochs, history_brb['nmi'], color='#9467bd', linewidth=2.5,
                label='NMI')
    first = True
    for e in range(reset_interval, max(epochs) + 1, reset_interval):
        ax_nmi.axvline(x=e, color='#d62728', ls='--', lw=1.3, alpha=0.6,
                       label='BRB event' if first else '')
        first = False
    ax_nmi.set_title('NMI vs Epoch', fontsize=12)
    ax_nmi.set_xlabel('Epoch'); ax_nmi.set_ylabel('NMI')
    ax_nmi.legend(fontsize=9); ax_nmi.grid(True, alpha=0.3)

    proj_label = 'PCA' if method == 'pca' else 't-SNE'
    fig.suptitle(f'Embedding Evolution + NMI ({proj_label}) — DCN + BRB',
                 fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    _save(fig, f'brb_embeddings_nmi_{method}.png')
