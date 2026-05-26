import torch
from tqdm import tqdm
from .metrics import compute_dice_iou
from ..losses.calibration import compute_ece
from ..models.uncertainty import compute_mc_predictions
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

    def evaluate(self, test_loader, mode='mc'):
        """
        mode: 'mc' for Bayesian MC sampling, 'deterministic' for fast prompt.
        """
        self.model.eval()
        
        all_dice = []
        all_iou = []
        all_ece = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating ({mode})"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if mode == 'mc':
                    # MC Sampling (R=8)
                    mc_logits = self.model(
                        images, 
                        num_samples=self.config.adaptation.mc_samples_eval, 
                        deterministic=False
                    )
                    probs, _ = compute_mc_predictions(mc_logits)
                else:
                    # Deterministic
                    logits = self.model(images, num_samples=1, deterministic=True)
                    probs = torch.softmax(logits, dim=1)
                
                metrics = compute_dice_iou(probs, masks)
                ece = compute_ece(probs, masks)
                
                all_dice.extend(metrics['dice'])
                all_iou.extend(metrics['iou'])
                all_ece.append(ece)
                
        results = {
            'dice_mean': np.mean(all_dice),
            'dice_std': np.std(all_dice),
            'iou_mean': np.mean(all_iou),
            'ece_mean': np.mean(all_ece)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Dice: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
        logger.info(f"IoU: {results['iou_mean']:.4f}")
        logger.info(f"ECE: {results['ece_mean']:.4f}")
        
        return results
