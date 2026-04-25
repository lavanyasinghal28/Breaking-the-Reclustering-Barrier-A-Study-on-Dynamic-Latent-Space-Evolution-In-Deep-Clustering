import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=784, embedding_dim=10):
        super(SimpleAutoencoder, self).__init__()

        # Encoder: 784 -> 256 -> 128 -> 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        # Decoder: 10 -> 128 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return z, reconstruction
