"""
train.py — Main entry point.

Runs three experiments on MNIST with DCN:
  1. Plain DCN (no interventions)
  2. DCN + Reclustering only
  3. DCN + Full BRB (soft reset + reclustering)

All hyperparameters live in config.py.
To add a new clustering algorithm, create its model in models/ and its
training loop in training/, then wire it into this file.
"""
import copy
import os
import json
import torch

from config import (
    BATCH_SIZE, PRETRAIN_EPOCHS, CLUSTER_EPOCHS,
    RESET_INTERVAL, ALPHA, INPUT_DIM, EMBEDDING_DIM,
    N_CLUSTERS, PLOT_DIR,
)
from dataset import get_mnist_dataloaders
from models import SimpleAutoencoder, DCN
from training import (
    pretrain_autoencoder,
    train_dcn_plain,
    train_dcn_with_reclustering,
    train_dcn_with_brb,
)
from plotting import (
    plot_single_run,
    plot_comparison,
    plot_accuracy_only_comparison,
    plot_loss_only_comparison,
    plot_cluster_quality_comparison,
    plot_intra_cd_comparison,
    plot_inter_cd_comparison,
    plot_cluster_loss_comparison,
    plot_d1d2_histogram,
    plot_embedding_panels,
    plot_nmi_curve,
    plot_embedding_and_nmi,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    print("Loading MNIST …")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=BATCH_SIZE)

    # ── Shared pre-trained autoencoder ───────────────────────────────────
    base_ae = SimpleAutoencoder(input_dim=INPUT_DIM,
                                embedding_dim=EMBEDDING_DIM).to(device)
    pretrain_autoencoder(base_ae, train_loader,
                         pretrain_epochs=PRETRAIN_EPOCHS, device=device)
    pretrained_state = copy.deepcopy(base_ae.state_dict())

    def _fresh_dcn():
        """Clone AE from pretrained checkpoint and wrap in DCN."""
        ae = SimpleAutoencoder(input_dim=INPUT_DIM,
                               embedding_dim=EMBEDDING_DIM).to(device)
        ae.load_state_dict(copy.deepcopy(pretrained_state))
        return DCN(ae, n_clusters=N_CLUSTERS,
                   embedding_dim=EMBEDDING_DIM).to(device)

    results = {}

    # ── Experiment 1 — Plain DCN ─────────────────────────────────────────
    hist_plain = train_dcn_plain(
        _fresh_dcn(), train_loader, test_loader,
        epochs=CLUSTER_EPOCHS, device=device,
    )
    results['DCN (plain)'] = hist_plain
    plot_single_run(hist_plain, "DCN (plain)", "dcn_plain.png")

    # ── Experiment 2 — DCN + Reclustering ────────────────────────────────
    hist_recluster = train_dcn_with_reclustering(
        _fresh_dcn(), train_loader, test_loader,
        epochs=CLUSTER_EPOCHS, recluster_interval=RESET_INTERVAL,
        device=device,
    )
    results['DCN + Reclustering'] = hist_recluster
    plot_single_run(hist_recluster, "DCN + Reclustering",
                    "dcn_reclustering.png", interval=RESET_INTERVAL)

    # ── Experiment 3 — DCN + Full BRB ────────────────────────────────────
    hist_brb, brb_snapshots = train_dcn_with_brb(
        _fresh_dcn(), train_loader, test_loader,
        epochs=CLUSTER_EPOCHS, reset_interval=RESET_INTERVAL,
        alpha=ALPHA, device=device,
    )
    results['DCN + BRB'] = hist_brb
    plot_single_run(hist_brb, "DCN + BRB",
                    "dcn_brb.png", interval=RESET_INTERVAL)

    # ── Comparison plots ─────────────────────────────────────────────────
    plot_comparison(results, RESET_INTERVAL)
    plot_accuracy_only_comparison(results, RESET_INTERVAL)
    plot_loss_only_comparison(results, RESET_INTERVAL)

    # ── Cluster quality ──────────────────────────────────────────────────
    plot_cluster_quality_comparison(results, RESET_INTERVAL)
    plot_intra_cd_comparison(results, RESET_INTERVAL)
    plot_inter_cd_comparison(results, RESET_INTERVAL)
    plot_cluster_loss_comparison(results, RESET_INTERVAL)

    # ── BRB analysis ─────────────────────────────────────────────────────
    print("\n  Generating BRB analysis plots...")
    plot_d1d2_histogram(brb_snapshots, RESET_INTERVAL)
    plot_nmi_curve(hist_brb, RESET_INTERVAL)
    plot_embedding_panels(brb_snapshots, method='pca')
    plot_embedding_panels(brb_snapshots, method='tsne')
    plot_embedding_and_nmi(brb_snapshots, hist_brb, RESET_INTERVAL, method='pca')
    plot_embedding_and_nmi(brb_snapshots, hist_brb, RESET_INTERVAL, method='tsne')

    # ── Save metrics JSON ────────────────────────────────────────────────
    os.makedirs(PLOT_DIR, exist_ok=True)
    metrics_path = os.path.join(PLOT_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw metrics saved → {metrics_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for label, hist in results.items():
        best_acc = max(hist['accuracy']) * 100
        final_acc = hist['accuracy'][-1] * 100
        final_loss = hist['loss'][-1]
        print(f"  {label:25s} | Best Acc: {best_acc:6.2f}%  "
              f"| Final Acc: {final_acc:6.2f}%  "
              f"| Final Loss: {final_loss:.4f}")
    print(f"{'='*60}")
    print(f"\n  All plots saved in '{PLOT_DIR}/' directory.")


if __name__ == "__main__":
    main()