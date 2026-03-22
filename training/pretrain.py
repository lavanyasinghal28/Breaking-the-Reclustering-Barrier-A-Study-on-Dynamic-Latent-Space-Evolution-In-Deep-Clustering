import torch
import torch.nn as nn

from config import PRETRAIN_EPOCHS, LEARNING_RATE


def pretrain_autoencoder(autoencoder, train_loader, pretrain_epochs=None,
                         lr=None, device='cpu'):
    """
    Pre-train the autoencoder on reconstruction loss only.
    Gives the embedding space meaningful structure before clustering.
    """
    pretrain_epochs = pretrain_epochs or PRETRAIN_EPOCHS
    lr = lr or LEARNING_RATE

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    print(f"\n{'='*60}")
    print(f"  Autoencoder Pre-training ({pretrain_epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, pretrain_epochs + 1):
        autoencoder.train()
        total_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            _, x_recon = autoencoder(batch_x)
            loss = mse_loss_fn(x_recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Pretrain Epoch {epoch:03d} | Recon Loss: {avg_loss:.6f}")

    print("  Pre-training complete.\n")
    return autoencoder
