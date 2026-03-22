import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from brb_utils import apply_soft_reset_to_network

def cluster_accuracy(y_true, y_pred):
    """
    standard metric for unsupervised clustering.
    Finds the best matching between predicted clusters and true labels.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size

class DCN(nn.Module):
    def __init__(self, autoencoder, n_clusters=10, embedding_dim=10):
        super(DCN, self).__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        # Cluster centers are learnable parameters, just like network weights!
        self.centers = nn.Parameter(torch.zeros(n_clusters, embedding_dim))
        
    @torch.no_grad()
    def init_kmeans(self, dataloader, device):
        """Passes all data through the AE and runs K-Means to find starting centers."""
        print("Running K-Means to initialize/re-initialize centers...")
        self.autoencoder.eval()
        embeddings = []
        
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            z = self.autoencoder.encode(batch_x)
            embeddings.append(z.cpu().numpy())
            
        embeddings = np.concatenate(embeddings, axis=0)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(embeddings)
        
        # Teleport our PyTorch cluster centers to the new K-Means centers
        self.centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        self.autoencoder.train()

def train_dcn_with_brb(
    dcn_model, 
    train_loader, 
    test_loader, 
    epochs=50, 
    reset_interval=10, 
    alpha=0.7, 
    device='cpu'
):
    """
    The main training loop implementing Deep Clustering Network (DCN) + BRB.
    """
    optimizer = torch.optim.Adam(dcn_model.parameters(), lr=1e-3)
    mse_loss_fn = nn.MSELoss()
    
    # Initialize clusters for the first time
    dcn_model.init_kmeans(train_loader, device)
    
    # Lists to store metrics for our graphs later!
    history = {'epoch': [], 'loss': [], 'accuracy': []}

    print("\n--- Starting DCN Training ---")
    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % reset_interval == 0:
            print(f"\n[Epoch {epoch}] BRB Triggered! Breaking the Reclustering Barrier...")
            
            # 1. Soft Reset the Autoencoder Weights
            apply_soft_reset_to_network(dcn_model.autoencoder, alpha=alpha)
            
            # 2. Re-run K-Means on the newly shifted embeddings
            dcn_model.init_kmeans(train_loader, device)
            
            # 3. Wipe the optimizer's memory for the centers so they don't get pulled backwards
            optimizer.state[dcn_model.centers] = {}

        dcn_model.train()
        total_loss = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            z, x_recon = dcn_model.autoencoder(batch_x)
            
            # 1. Reconstruction Loss (Make sure it still looks like a digit)
            loss_recon = mse_loss_fn(x_recon, batch_x)
            
            # 2. Clustering Loss (Pull embeddings toward their closest center)
            # Calculate distance from every point to every center
            distances = torch.cdist(z, dcn_model.centers)
            # Find the closest center for each point
            assigned_centers_idx = torch.argmin(distances, dim=1)
            # Calculate MSE between the point and its assigned center
            loss_cluster = mse_loss_fn(z, dcn_model.centers[assigned_centers_idx])
            
            # Joint Loss (Usually clustering loss is scaled down, e.g., by 0.1)
            loss = loss_recon + (0.1 * loss_cluster)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        dcn_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                z = dcn_model.autoencoder.encode(batch_x)
                distances = torch.cdist(z, dcn_model.centers)
                preds = torch.argmin(distances, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                
        acc = cluster_accuracy(np.array(all_labels), np.array(all_preds))
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
        
        # Save metrics for plotting
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        
    return history