import torch
import torch.nn as nn
from model import VAE

LEARNING_RATE = 1e-4


def loss_function(recon_x, x, mu, logvar, beta=1.5):
    x = (x + 1) / 2  # Масштабування до [0, 1]
    # recon_loss = nn.MSELoss()(recon_x, x)
    recon_loss = nn.BCELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                               logvar.exp()) / x.size(0)
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train(epoch, train_loader, vae, optimizer, device, beta=1.0, max_norm=1.0):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss, recon_loss, kl_loss = loss_function(
            recon_batch, data, mu, logvar, beta)
        loss.backward()

        # Додаємо обмеження на градієнти
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm)

        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch}, Total Loss: {train_loss / len(train_loader)
          :.6f}, Recon Loss: {recon_loss:.6f}, KL Loss: {kl_loss:.6f}")
    print(f"Epoch {epoch}, mu mean: {mu.mean().item()
          :.6f}, logvar mean: {logvar.mean().item():.6f}")
