import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def plot_qualitative_grid(images, gts, preds_dict, save_path="grid.png"):
    """
    images: [B, C, H, W]
    gts: [B, H, W]
    preds_dict: dict of {model_name: [B, H, W]}
    """
    B = images.shape[0]
    num_cols = 2 + len(preds_dict) # Image, GT, Model1, Model2...
    
    fig, axes = plt.subplots(B, num_cols, figsize=(3*num_cols, 3*B))
    if B == 1:
        axes = np.expand_dims(axes, 0)
        
    model_names = list(preds_dict.keys())
    
    for b in range(B):
        # Image
        img = images[b].permute(1, 2, 0).cpu().numpy()
        axes[b, 0].imshow(img)
        axes[b, 0].axis('off')
        if b == 0: axes[b, 0].set_title("Image")
        
        # GT
        gt = gts[b].cpu().numpy()
        axes[b, 1].imshow(img)
        axes[b, 1].imshow(gt, alpha=0.5, cmap='jet')
        axes[b, 1].axis('off')
        if b == 0: axes[b, 1].set_title("Ground Truth")
        
        # Models
        for i, name in enumerate(model_names):
            pred = preds_dict[name][b].cpu().numpy()
            axes[b, 2+i].imshow(img)
            axes[b, 2+i].imshow(pred, alpha=0.5, cmap='jet')
            axes[b, 2+i].axis('off')
            if b == 0: axes[b, 2+i].set_title(name)
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
