import os
import torch
from model import VAE, weights_init
from train import train
from train import LEARNING_RATE
from load_data import load_data
from generate import generate_samples
from visualize import save_reconstructions, save_losses

BATCH_SIZE = 64
EPOCHS = 50
LATENT_SIZES = [64, 128, 256, 512]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    train_loader, test_loader = load_data(BATCH_SIZE)

    for latent_dim in LATENT_SIZES:
        print(f"Training VAE with latent_dim={latent_dim}")
        vae = VAE(latent_dim).to(DEVICE)
        vae.apply(weights_init)
        optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

        # Оновлення: прибираємо verbose
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)

        losses = []

        for epoch in range(1, EPOCHS + 1):
            train(epoch, train_loader, vae, optimizer, DEVICE, beta=1.0)

            # Зберігаємо результати після кожного епоху
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(DEVICE)
                    recon, _, _ = vae(data)
                    save_reconstructions(
                        data, recon, epoch, latent_dim, RESULTS_DIR)
                    break

            # Обчислюємо середній loss і оновлюємо scheduler
            avg_loss = sum(losses) / len(losses) if losses else 0
            scheduler.step(avg_loss)

            # Логування learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Current Learning Rate: {current_lr:.6e}")

            losses.append(avg_loss)

        save_losses(losses, os.path.join(
            RESULTS_DIR, f"losses_latent{latent_dim}.png"))


if __name__ == "__main__":
    main()
