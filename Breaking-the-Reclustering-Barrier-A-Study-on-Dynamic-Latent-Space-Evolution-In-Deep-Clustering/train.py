"""Main entry point for reset-method comparison experiments on MNIST (DCN)."""
import copy
import os
import json
import torch

from config import (
    BATCH_SIZE, PRETRAIN_EPOCHS,
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
    train_dcn_with_reset_method,
)
from plotting import (
    plot_single_run,
    plot_comparison,
    plot_pairwise_method_vs_plain,
    plot_reset_methods_comparison,
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
    exp_cluster_epochs = 100

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

    # ── Experiment 1: Plain DCN ──────────────────────────────────────────
    hist_plain = train_dcn_plain(
        _fresh_dcn(), train_loader, test_loader,
        epochs=exp_cluster_epochs, device=device,
    )
    results['DCN (plain)'] = hist_plain
    plot_single_run(hist_plain, "DCN (plain)", "dcn_plain.png")

    # ── Experiment 2: DCN + Reclustering ─────────────────────────────────
    hist_recluster = train_dcn_with_reclustering(
        _fresh_dcn(), train_loader, test_loader,
        epochs=exp_cluster_epochs, recluster_interval=RESET_INTERVAL,
        device=device,
    )
    results['DCN + Reclustering'] = hist_recluster
    plot_single_run(hist_recluster, "DCN + Reclustering",
                    "dcn_reclustering.png", interval=RESET_INTERVAL)

    # ── Experiment 3: DCN + BRB (original) ───────────────────────────────
    hist_brb, brb_snapshots = train_dcn_with_brb(
        _fresh_dcn(), train_loader, test_loader,
        epochs=exp_cluster_epochs, reset_interval=RESET_INTERVAL,
        alpha=ALPHA, device=device,
    )
    results['DCN + BRB'] = hist_brb
    plot_single_run(hist_brb, "DCN + BRB",
                    "dcn_brb.png", interval=RESET_INTERVAL)

    # ── Experiment 4: DCN + LDAR ─────────────────────────────────────────
    hist_ldar = train_dcn_with_reset_method(
        _fresh_dcn(), train_loader, test_loader,
        method_name='ldar', epochs=exp_cluster_epochs,
        reset_interval=RESET_INTERVAL, device=device,
        ldar_kwargs={'alpha_base': 0.8, 'gamma': 1.0},
    )
    results['DCN + LDAR'] = hist_ldar
    plot_single_run(hist_ldar, "DCN + LDAR", "dcn_ldar.png", interval=RESET_INTERVAL)

    # ── Experiment 5: DCN + LSAR ─────────────────────────────────────────
    hist_lsar = train_dcn_with_reset_method(
        _fresh_dcn(), train_loader, test_loader,
        method_name='lsar', epochs=exp_cluster_epochs,
        reset_interval=RESET_INTERVAL, device=device,
        lsar_kwargs={
            'alpha_max': 0.90,
            'alpha_min': 0.75,
            'window': 5,
            'eps': 1e-3,
            'stagnation_factor': 0.88,
        },
    )
    results['DCN + LSAR'] = hist_lsar
    plot_single_run(hist_lsar, "DCN + LSAR", "dcn_lsar.png", interval=RESET_INTERVAL)

    # ── Experiment 6: DCN + FWR ──────────────────────────────────────────
    hist_fwr = train_dcn_with_reset_method(
        _fresh_dcn(), train_loader, test_loader,
        method_name='fwr', epochs=exp_cluster_epochs,
        reset_interval=RESET_INTERVAL, device=device,
        fwr_kwargs={'n_batches': 8, 'lambda_': 3.0, 'alpha_floor': 0.3},
    )
    results['DCN + FWR'] = hist_fwr
    plot_single_run(hist_fwr, "DCN + FWR", "dcn_fwr.png", interval=RESET_INTERVAL)

    # ── Main comparison plots ────────────────────────────────────────────
    plot_comparison(results, RESET_INTERVAL)
    plot_accuracy_only_comparison(results, RESET_INTERVAL)
    plot_loss_only_comparison(results, RESET_INTERVAL)

    # ── Individual method vs plain plots ─────────────────────────────────
    plot_pairwise_method_vs_plain(
        'DCN + BRB', hist_plain, hist_brb, RESET_INTERVAL,
        filename='plain_vs_brb.png',
    )
    plot_pairwise_method_vs_plain(
        'DCN + LDAR', hist_plain, hist_ldar, RESET_INTERVAL,
        filename='plain_vs_ldar.png',
    )
    plot_pairwise_method_vs_plain(
        'DCN + LSAR', hist_plain, hist_lsar, RESET_INTERVAL,
        filename='plain_vs_lsar.png',
    )
    plot_pairwise_method_vs_plain(
        'DCN + FWR', hist_plain, hist_fwr, RESET_INTERVAL,
        filename='plain_vs_fwr.png',
    )

    # ── All reset methods together ───────────────────────────────────────
    reset_results = {
        'DCN + BRB': hist_brb,
        'DCN + LDAR': hist_ldar,
        'DCN + LSAR': hist_lsar,
        'DCN + FWR': hist_fwr,
    }
    plot_reset_methods_comparison(
        reset_results,
        RESET_INTERVAL,
        filename='comparison_reset_methods.png',
    )
    plot_comparison(
        reset_results,
        RESET_INTERVAL,
        filename='comparison_reset_methods_loss_accuracy.png',
        title='Loss and Accuracy: BRB vs LDAR vs LSAR vs FWR',
    )

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
    print(f"\n  Raw metrics saved -> {metrics_path}")

    final_accuracies = {
        label: {
            'final_accuracy': float(hist['accuracy'][-1]),
            'best_accuracy': float(max(hist['accuracy'])),
        }
        for label, hist in results.items()
    }
    final_acc_path = os.path.join(PLOT_DIR, "final_accuracies.json")
    with open(final_acc_path, 'w') as f:
        json.dump(final_accuracies, f, indent=2)
    print(f"  Final accuracy summary saved -> {final_acc_path}")

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