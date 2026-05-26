import matplotlib.pyplot as plt
import os

def plot_uncertainty_map(image, entropy, save_path="uncertainty.png"):
    """
    image: [3, H, W] tensor
    entropy: [1, H, W] tensor
    """
    img_np = image.permute(1, 2, 0).cpu().numpy()
    ent_np = entropy[0].cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_np)
    # Overlay entropy
    cax = ax.imshow(ent_np, cmap='inferno', alpha=0.6)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
