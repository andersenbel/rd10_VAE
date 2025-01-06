import torch
from model import VAE


def generate_samples(latent_dim, vae, device):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        samples = vae.decoder(z).cpu()
        return samples
