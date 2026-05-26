import os
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def run_visualization(config_path: str, unknown_args: list):
    logger.info("Generating improved qualitative grid visualization (Template Mode)...")
    
    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    methods = [
        "Image", "Ground Truth", "U-Net", "nnU-Net", "SegFormer", 
        "CDUN", "MedSAM", "SAM-Adapter", "BayesPrompt"
    ]
    
    # 2 rows (1 MRI, 1 US), 9 columns
    fig, axes = plt.subplots(2, len(methods), figsize=(20, 5))
    
    # Generate blank mock images to simulate the structure without requiring actual data
    for i, modality in enumerate(["MRI", "Ultrasound"]):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            ax.imshow(np.zeros((100, 100)), cmap='gray')
            ax.axis('off')
            
            # Simulate Ground Truth boundary in green and Prediction in magenta
            if j == 1:
                ax.plot([25, 75], [25, 25], color='green', linewidth=2)
            elif j > 1:
                ax.plot([25, 75], [25, 25], color='green', linewidth=2, alpha=0.5)
                ax.plot([25, 75], [30, 30], color='magenta', linewidth=2)
            
            # Simulate zoomed ROI in bottom right
            if j > 0:
                ax.text(70, 90, "ROI", color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
            
            if i == 0:
                ax.set_title(method, fontsize=10)
            if j == 0:
                ax.text(-20, 50, modality, rotation=90, va='center', fontsize=12, fontweight='bold')
                
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    png_path = os.path.join(out_dir, "qualitative_grid.png")
    pdf_path = os.path.join(out_dir, "qualitative_grid.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved qualitative grids to {png_path} and {pdf_path}")
    logger.info("*(Note: To generate real figures, ensure proper checkpoint and split configuration.)*")
