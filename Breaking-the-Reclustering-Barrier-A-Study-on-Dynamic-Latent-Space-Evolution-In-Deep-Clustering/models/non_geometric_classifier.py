import torch.nn as nn


class NonGeometricClassifier(nn.Module):
    """Encoder + MLP head classifier (no distance/prototype geometry in objective)."""

    def __init__(self, autoencoder, embedding_dim=10, hidden_dim=64, n_classes=10):
        super().__init__()
        self.autoencoder = autoencoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        return logits
