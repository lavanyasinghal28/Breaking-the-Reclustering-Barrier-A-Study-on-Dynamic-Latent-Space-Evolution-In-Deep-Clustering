import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class DCN(nn.Module):
    """
    Deep Clustering Network: wraps an autoencoder and maintains learnable
    cluster centers as an nn.Parameter.
    """
    def __init__(self, autoencoder, n_clusters=10, embedding_dim=10):
        super(DCN, self).__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.centers = nn.Parameter(torch.zeros(n_clusters, embedding_dim))

    @torch.no_grad()
    def init_kmeans(self, dataloader, device):
        """Pass all data through the AE, run K-Means, set centers."""
        print("  Running K-Means to initialize centers...")
        self.autoencoder.eval()
        embeddings = []

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            z = self.autoencoder.encode(batch_x)
            embeddings.append(z.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(embeddings)

        self.centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(device)
        self.autoencoder.train()
