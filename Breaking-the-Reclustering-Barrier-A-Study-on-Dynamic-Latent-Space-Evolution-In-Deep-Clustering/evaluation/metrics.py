import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def cluster_accuracy(y_true, y_pred):
    """
    Hungarian-algorithm matching between predicted clusters and true labels.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size


@torch.no_grad()
def evaluate(dcn_model, test_loader, device):
    """Compute clustering accuracy on a test set."""
    dcn_model.eval()
    all_preds, all_labels = [], []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        z = dcn_model.autoencoder.encode(batch_x)
        distances = torch.cdist(z, dcn_model.centers)
        preds = torch.argmin(distances, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())
    return cluster_accuracy(np.array(all_labels), np.array(all_preds))


@torch.no_grad()
def compute_cluster_quality(dcn_model, dataloader, device):
    """
    Returns (intra_cd, inter_cd, cl):
      - intra_cd : mean distance to assigned center (compactness)
      - inter_cd : mean pairwise distance between centers (separation)
      - cl       : mean squared distance to assigned center
    """
    dcn_model.eval()
    centers = dcn_model.centers

    total_intra = 0.0
    total_cl = 0.0
    n_samples = 0

    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        z = dcn_model.autoencoder.encode(batch_x)
        dists = torch.cdist(z, centers)
        min_dists, _ = dists.min(dim=1)
        total_intra += min_dists.sum().item()
        total_cl += (min_dists ** 2).sum().item()
        n_samples += z.size(0)

    intra_cd = total_intra / n_samples
    cl = total_cl / n_samples

    center_dists = torch.cdist(centers, centers)
    k = centers.size(0)
    mask = torch.triu(torch.ones(k, k, device=device), diagonal=1).bool()
    inter_cd = center_dists[mask].mean().item()

    return intra_cd, inter_cd, cl


@torch.no_grad()
def compute_nmi(dcn_model, dataloader, device):
    """NMI between predicted cluster assignments and true labels."""
    dcn_model.eval()
    all_preds, all_labels = [], []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        z = dcn_model.autoencoder.encode(batch_x)
        dists = torch.cdist(z, dcn_model.centers)
        preds = torch.argmin(dists, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())
    return normalized_mutual_info_score(all_labels, all_preds)


@torch.no_grad()
def collect_d1d2_ratios(dcn_model, dataloader, device, max_samples=5000):
    """
    d1/d2 ratio (nearest / 2nd-nearest center distance) per sample.
    Ratio close to 1 = uncertain assignment.
    """
    dcn_model.eval()
    ratios = []
    n = 0
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        z = dcn_model.autoencoder.encode(batch_x)
        dists = torch.cdist(z, dcn_model.centers)
        topk = torch.topk(dists, k=2, dim=1, largest=False)
        d1 = topk.values[:, 0]
        d2 = topk.values[:, 1]
        ratios.append((d1 / d2.clamp(min=1e-8)).cpu().numpy())
        n += z.size(0)
        if n >= max_samples:
            break
    return np.concatenate(ratios)[:max_samples]


@torch.no_grad()
def collect_embeddings(dcn_model, dataloader, device, max_samples=5000):
    """Snapshot of (embeddings, labels, centers) for visualisation."""
    dcn_model.eval()
    embs, labs = [], []
    n = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        z = dcn_model.autoencoder.encode(batch_x)
        embs.append(z.cpu().numpy())
        labs.append(batch_y.numpy())
        n += z.size(0)
        if n >= max_samples:
            break
    return (np.concatenate(embs)[:max_samples],
            np.concatenate(labs)[:max_samples],
            dcn_model.centers.cpu().numpy())
