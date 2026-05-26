import torch
import torch.optim as optim
from ..losses.bayesian import BMPALoss
import logging
import os

logger = logging.getLogger(__name__)

class TargetAdapter:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = BMPALoss(lambda_kl=config.adaptation.lambda_kl)

    def fast_prompt(self, support_loader):
        logger.info("Running Fast Prompt...")
        self.model.eval()
        
        all_images = []
        all_masks = []
        for batch in support_loader:
            all_images.append(batch['image'])
            all_masks.append(batch['mask'])
            
        images = torch.cat(all_images, dim=0).to(self.device)
        masks = torch.cat(all_masks, dim=0).to(self.device)
        
        if masks.dim() == 3:
            masks = torch.nn.functional.one_hot(masks, num_classes=self.config.dataset.num_classes).permute(0, 3, 1, 2)
            
        # Fast Prompt is deterministic
        prompts = self.model.compute_prompts(images, masks, deterministic=True)
        logger.info(f"Prompts computed with shape {prompts.shape}")
        return prompts

    def bayesian_finetune(self, support_loader):
        logger.info("Running Bayesian Fine-Tuning...")
        self.model.train()
        
        # Optimize only variational parameters
        trainable_params = [self.model.head.mu, self.model.head.rho]
        optimizer = optim.AdamW(trainable_params, lr=self.config.optimizer.lr)
        
        all_images = []
        all_masks = []
        for batch in support_loader:
            all_images.append(batch['image'])
            all_masks.append(batch['mask'])
        
        images = torch.cat(all_images, dim=0).to(self.device)
        masks = torch.cat(all_masks, dim=0).to(self.device)
        
        if masks.dim() == 3:
            masks_onehot = torch.nn.functional.one_hot(masks, num_classes=self.config.dataset.num_classes).permute(0, 3, 1, 2)
        else:
            masks_onehot = masks
            
        with torch.no_grad():
            self.model.compute_prompts(images, masks_onehot, num_mc=self.config.adaptation.mc_samples_train)

        self.model.head.initialize_posterior_from_prior()

        for i in range(self.config.adaptation.iterations):
            optimizer.zero_grad()
            
            mc_logits = self.model(images, num_samples=self.config.adaptation.mc_samples_train, deterministic=False)
            
            kl_tensor = self.model.head.kl_to_prior()
            
            loss, l_seg, l_kl = self.criterion(mc_logits, masks, kl_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (i+1) % 5 == 0:
                logger.info(f"Iteration {i+1} | Loss: {loss.item():.4f} | Seg: {l_seg.item():.4f} | KL: {l_kl.item():.4f}")
        
        logger.info("Bayesian Fine-Tuning completed.")
