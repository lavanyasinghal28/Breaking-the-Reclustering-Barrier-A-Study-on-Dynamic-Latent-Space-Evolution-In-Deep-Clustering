import torch
import torch.nn as nn

from config import LEARNING_RATE
from utils import apply_soft_reset_to_network


def _classification_train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


@torch.no_grad()
def _classification_accuracy(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1)

        total += batch_y.size(0)
        correct += (preds == batch_y).sum().item()

    return correct / total if total else 0.0


def train_non_geometric_classifier_plain(classifier_model, train_loader, test_loader,
                                        epochs=100, device='cpu'):
    """Train non-geometric encoder-head classifier with standard CE loss."""
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    history = {'epoch': [], 'loss': [], 'accuracy': []}

    print(f"\n{'='*60}")
    print(f"  Non-Geometric Classifier (plain) - {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        avg_loss = _classification_train_epoch(
            classifier_model, train_loader, optimizer, criterion, device
        )
        acc = _classification_accuracy(classifier_model, test_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)

    return history


def train_non_geometric_classifier_with_brb(classifier_model, train_loader, test_loader,
                                            epochs=100, reset_interval=10,
                                            alpha=0.7, device='cpu'):
    """
    Train non-geometric classifier with BRB-style periodic soft reset.

    This applies soft reset to encoder weights and continues CE training.
    """
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    history = {'epoch': [], 'loss': [], 'accuracy': []}

    print(f"\n{'='*60}")
    print(f"  Non-Geometric Classifier + BRB (interval={reset_interval}, a={alpha}) - {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % reset_interval == 0:
            print(f"  [Epoch {epoch}] BRB reset on classifier encoder")
            apply_soft_reset_to_network(classifier_model.autoencoder, alpha=alpha)

        avg_loss = _classification_train_epoch(
            classifier_model, train_loader, optimizer, criterion, device
        )
        acc = _classification_accuracy(classifier_model, test_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)

    return history
