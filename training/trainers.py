import torch
import torch.nn as nn

from config import LEARNING_RATE, CLUSTER_LOSS_WT
from evaluation import evaluate, compute_cluster_quality, compute_nmi
from evaluation import collect_d1d2_ratios, collect_embeddings
from utils import apply_soft_reset_to_network


def _train_one_epoch(dcn_model, train_loader, optimizer, mse_loss_fn, device,
                     cluster_loss_weight=CLUSTER_LOSS_WT):
    """Run one training epoch; return average joint loss."""
    dcn_model.train()
    total_loss = 0

    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        z, x_recon = dcn_model.autoencoder(batch_x)

        loss_recon = mse_loss_fn(x_recon, batch_x)

        distances = torch.cdist(z, dcn_model.centers)
        assigned_centers_idx = torch.argmin(distances, dim=1)
        loss_cluster = mse_loss_fn(z, dcn_model.centers[assigned_centers_idx])

        loss = loss_recon + cluster_loss_weight * loss_cluster

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ── Mode 1: Plain DCN ───────────────────────────────────────────────────────

def train_dcn_plain(dcn_model, train_loader, test_loader, epochs=50,
                    device='cpu'):
    """
    Vanilla DCN: K-Means init once, then joint training with NO interventions.
    Demonstrates the reclustering barrier — accuracy plateaus early.
    """
    optimizer = torch.optim.Adam(dcn_model.parameters(), lr=LEARNING_RATE)
    mse_loss_fn = nn.MSELoss()

    dcn_model.init_kmeans(train_loader, device)

    history = {'epoch': [], 'loss': [], 'accuracy': [],
               'intra_cd': [], 'inter_cd': [], 'cluster_loss': []}

    print(f"\n{'='*60}")
    print(f"  DCN Plain (no reclustering, no BRB) — {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        avg_loss = _train_one_epoch(dcn_model, train_loader, optimizer,
                                    mse_loss_fn, device)
        acc = evaluate(dcn_model, test_loader, device)
        intra, inter, cl = compute_cluster_quality(dcn_model, test_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | IntraCD: {intra:.4f} | "
                  f"InterCD: {inter:.4f} | CL: {cl:.4f}")

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        history['intra_cd'].append(intra)
        history['inter_cd'].append(inter)
        history['cluster_loss'].append(cl)

    return history


# ── Mode 2: DCN + Reclustering only ─────────────────────────────────────────

def train_dcn_with_reclustering(dcn_model, train_loader, test_loader,
                                epochs=50, recluster_interval=10, device='cpu'):
    """
    DCN with periodic K-Means re-initialization of centers but WITHOUT
    soft-resetting the encoder weights.  Shows the 'reclustering barrier'.
    """
    optimizer = torch.optim.Adam(dcn_model.parameters(), lr=LEARNING_RATE)
    mse_loss_fn = nn.MSELoss()

    dcn_model.init_kmeans(train_loader, device)

    history = {'epoch': [], 'loss': [], 'accuracy': [],
               'intra_cd': [], 'inter_cd': [], 'cluster_loss': []}

    print(f"\n{'='*60}")
    print(f"  DCN + Reclustering (interval={recluster_interval}) — {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % recluster_interval == 0:
            print(f"  [Epoch {epoch}] Reclustering centers (no weight reset)...")
            dcn_model.init_kmeans(train_loader, device)
            if dcn_model.centers in optimizer.state:
                optimizer.state[dcn_model.centers] = {}

        avg_loss = _train_one_epoch(dcn_model, train_loader, optimizer,
                                    mse_loss_fn, device)
        acc = evaluate(dcn_model, test_loader, device)
        intra, inter, cl = compute_cluster_quality(dcn_model, test_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | IntraCD: {intra:.4f} | "
                  f"InterCD: {inter:.4f} | CL: {cl:.4f}")

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        history['intra_cd'].append(intra)
        history['inter_cd'].append(inter)
        history['cluster_loss'].append(cl)

    return history


# ── Mode 3: DCN + Full BRB ──────────────────────────────────────────────────

def train_dcn_with_brb(dcn_model, train_loader, test_loader, epochs=50,
                       reset_interval=10, alpha=0.7, device='cpu'):
    """
    Full BRB: periodic soft reset + reclustering + optimizer momentum reset.

    Returns
    -------
    history : dict          – epoch-level metrics
    brb_snapshots : dict    – per-BRB-event snapshots for visualisation
    """
    optimizer = torch.optim.Adam(dcn_model.parameters(), lr=LEARNING_RATE)
    mse_loss_fn = nn.MSELoss()

    dcn_model.init_kmeans(train_loader, device)

    history = {'epoch': [], 'loss': [], 'accuracy': [], 'nmi': [],
               'intra_cd': [], 'inter_cd': [], 'cluster_loss': []}

    brb_snapshots = {
        'd1d2_before': [],
        'd1d2_after':  [],
        'embeddings':  [],
    }

    print(f"\n{'='*60}")
    print(f"  DCN + BRB (interval={reset_interval}, α={alpha}) — {epochs} epochs")
    print(f"{'='*60}")

    # Initial state snapshot
    brb_snapshots['embeddings'].append(
        (0, 'Pre-BRB (init)',
         *collect_embeddings(dcn_model, test_loader, device)))

    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % reset_interval == 0:
            print(f"  [Epoch {epoch}] BRB Triggered!")

            ratios_pre = collect_d1d2_ratios(dcn_model, test_loader, device)
            brb_snapshots['d1d2_before'].append((epoch, ratios_pre))

            # 1. Soft reset encoder weights
            apply_soft_reset_to_network(dcn_model.autoencoder, alpha=alpha)

            brb_snapshots['embeddings'].append(
                (epoch, f'During BRB (epoch {epoch})',
                 *collect_embeddings(dcn_model, test_loader, device)))

            # 2. Reclustering on perturbed embeddings
            dcn_model.init_kmeans(train_loader, device)
            # 3. Reset optimizer momentum
            if dcn_model.centers in optimizer.state:
                optimizer.state[dcn_model.centers] = {}

            ratios_post = collect_d1d2_ratios(dcn_model, test_loader, device)
            brb_snapshots['d1d2_after'].append((epoch, ratios_post))

        avg_loss = _train_one_epoch(dcn_model, train_loader, optimizer,
                                    mse_loss_fn, device)
        acc = evaluate(dcn_model, test_loader, device)
        nmi = compute_nmi(dcn_model, test_loader, device)
        intra, inter, cl = compute_cluster_quality(dcn_model, test_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | NMI: {nmi:.4f} | "
                  f"IntraCD: {intra:.4f} | InterCD: {inter:.4f}")

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        history['nmi'].append(nmi)
        history['intra_cd'].append(intra)
        history['inter_cd'].append(inter)
        history['cluster_loss'].append(cl)

    # Final snapshot
    brb_snapshots['embeddings'].append(
        (epochs, 'Post-BRB (final)',
         *collect_embeddings(dcn_model, test_loader, device)))

    return history, brb_snapshots
