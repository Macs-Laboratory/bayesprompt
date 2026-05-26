import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..core.config import load_config
from ..core.registry import MODELS, DATASETS
from torch.utils.data import DataLoader
from ..models.uncertainty import compute_mc_predictions

logger = logging.getLogger(__name__)

def run_visualization(config_path, overrides):
    logger.info("Starting True Visualization Runner...")
    cfg = load_config(config_path, overrides)
    
    out_dir = os.path.join(cfg.output.root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Model and Data
    DatasetClass = DATASETS.get(cfg.dataset.name)
    dataset = DatasetClass(
        root_dir=cfg.dataset.root, 
        split="test", 
        modality=cfg.dataset.target_modality,
        fast_dev_run=cfg.dataset.fast_dev_run,
        image_size=cfg.dataset.image_size,
        split_file=cfg.dataset.split_file
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    ModelClass = MODELS.get(cfg.model.name)
    model = ModelClass(
        num_classes=cfg.dataset.num_classes, 
        embed_dim=cfg.model.feature_dim, 
        prompt_dim=cfg.model.prompt_dim,
        lightweight_backend=cfg.model.lightweight_backend,
    ).to(cfg.training.device)
    
    model.eval()
    
    batch = next(iter(loader))
    images = batch['image'].to(cfg.training.device)
    masks = batch['mask'].to(cfg.training.device)
    
    with torch.no_grad():
        mc_logits = model(images, num_samples=cfg.adaptation.mc_samples_eval, deterministic=False)
        mean_probs, entropy = compute_mc_predictions(mc_logits)
        
    img_np = images[0, 0].cpu().numpy()
    gt_np = masks[0].cpu().numpy()
    pred_np = torch.argmax(mean_probs[0], dim=0).cpu().numpy()
    entropy_np = entropy[0].cpu().numpy()
    error_np = (pred_np != gt_np).astype(np.float32)
    
    # 1. 9-Panel Qualitative Grid
    fig, axes = plt.subplots(1, 9, figsize=(30, 4))
    
    titles = ["Image", "Ground Truth", "U-Net", "nnU-Net", "SegFormer", "CDUN", "MedSAM", "SAM-Adapter", "BayesPrompt"]
    
    # We only have BayesPrompt prediction in this simple runner. Mark others as N/A.
    blank_np = np.zeros_like(img_np)
    panels = [img_np, gt_np, blank_np, blank_np, blank_np, blank_np, blank_np, blank_np, pred_np]
    
    for ax, title, data in zip(axes, titles, panels):
        if title not in ["Image", "Ground Truth", "BayesPrompt"]:
            ax.imshow(blank_np, cmap='gray')
            ax.text(0.5, 0.5, 'N/A', color='red', fontsize=24, ha='center', va='center', transform=ax.transAxes)
        else:
            ax.imshow(data, cmap='gray' if title == "Image" else 'jet')
        ax.set_title(title)
        ax.axis('off')
        
    out_file1 = os.path.join(out_dir, "qualitative_grid.png")
    plt.savefig(out_file1, bbox_inches='tight')
    plt.close(fig)
    
    # 2. 6-Panel Uncertainty Map
    fig2, axes2 = plt.subplots(1, 6, figsize=(24, 4))
    
    axes2[0].imshow(img_np, cmap='gray')
    axes2[0].set_title("Input Image")
    
    axes2[1].imshow(pred_np, cmap='jet')
    axes2[1].set_title("Prediction")
    
    axes2[2].imshow(gt_np, cmap='jet')
    axes2[2].set_title("Ground Truth")
    
    im3 = axes2[3].imshow(entropy_np, cmap='hot')
    axes2[3].set_title("Predictive Entropy")
    fig2.colorbar(im3, ax=axes2[3], fraction=0.046, pad=0.04)
    
    axes2[4].imshow(error_np, cmap='Reds')
    axes2[4].set_title("Error Map")
    
    axes2[5].imshow(img_np, cmap='gray')
    axes2[5].imshow(entropy_np, cmap='hot', alpha=0.5)
    axes2[5].set_title("Entropy Overlay")
    
    for ax in axes2:
        ax.axis('off')
        
    out_file2 = os.path.join(out_dir, "uncertainty_map.png")
    plt.savefig(out_file2, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Real Reliability Diagram
    try:
        from sklearn.calibration import calibration_curve
        probs_flat = mean_probs[0, 1].cpu().numpy().flatten()
        gt_flat = gt_np.flatten()
        fraction_of_positives, mean_predicted_value = calibration_curve(gt_flat, probs_flat, n_bins=10)
        
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax3.plot(mean_predicted_value, fraction_of_positives, 's-', label='BayesPrompt')
        ax3.set_title("Reliability Diagram")
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Accuracy")
        ax3.legend()
        
        out_file3 = os.path.join(out_dir, "reliability_diagram.png")
        plt.savefig(out_file3, bbox_inches='tight')
        plt.close(fig3)
    except ImportError:
        logger.warning("sklearn not installed, skipping real reliability diagram.")
