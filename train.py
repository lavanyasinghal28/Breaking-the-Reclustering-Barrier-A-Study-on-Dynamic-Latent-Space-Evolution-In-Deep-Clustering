import torch
import matplotlib.pyplot as plt

from dataset import get_mnist_dataloaders
from autoencoder import SimpleAutoencoder
from dec import DEC, train_dec_with_brb

def plot_results(history, reset_interval, epochs):
    """
    Plots the training loss and accuracy. 
    Adds vertical dashed lines where BRB was triggered so you can 
    visually see the network breaking out of the plateau!
    """
    epochs_range = history['epoch']
    
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['loss'], label='Joint Loss', color='red', linewidth=2)
    plt.title("DCN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    
    for i in range(reset_interval, epochs, reset_interval):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Clustering Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['accuracy'], label='Clustering Accuracy', color='blue', linewidth=2)
    plt.title("MNIST Clustering Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    for i in range(reset_interval, epochs, reset_interval):
        plt.axvline(x=i, color='gray', linestyle='--', label='BRB Soft Reset' if i == reset_interval else "")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("brb_results.png", dpi=300)
    print("\nTraining complete! Graph saved as 'brb_results.png'")
    plt.show()

def main():
    # 1. Setup Device (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 60            # Run for 60 epochs to see the plateau and the break
    RESET_INTERVAL = 15    # Trigger BRB every 15 epochs
    ALPHA = 0.7            # Keep 70% of old weights, inject 30% noise

    # 3. Load Data
    print("Loading MNIST data...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=BATCH_SIZE)

    # 4. Initialize the Neural Network and the DEC engine
    autoencoder = SimpleAutoencoder(input_dim=784, embedding_dim=10).to(device)
    dec_model = DEC(autoencoder, n_clusters=10, embedding_dim=10).to(device)

    # 5. Start the Training!
    history = train_dec_with_brb(
        dec_model=dec_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        reset_interval=RESET_INTERVAL,
        alpha=ALPHA,
        device=device
    )

    plot_results(history, RESET_INTERVAL, EPOCHS)

if __name__ == "__main__":
    main()