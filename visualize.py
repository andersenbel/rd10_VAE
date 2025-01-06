import matplotlib.pyplot as plt


def save_reconstructions(original, reconstructed, epoch, latent_dim, save_dir):
    original = ((original[:8] * 0.5) + 0.5).permute(0, 2, 3, 1).cpu().numpy()
    reconstructed = reconstructed[:8].permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(15, 5))
    for i in range(8):
        axes[0, i].imshow(original[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')

    fig.suptitle(f"Latent Dim: {latent_dim}, Epoch: {epoch}")
    plt.savefig(f"{save_dir}/recon_latent{latent_dim}_epoch{epoch}.png")
    plt.close()


def save_losses(losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
