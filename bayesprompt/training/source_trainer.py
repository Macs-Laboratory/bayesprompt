import torch
import torch.optim as optim
from tqdm import tqdm
from ..core.checkpointing import save_checkpoint
from ..losses.segmentation import SegLoss
import logging
import os

logger = logging.getLogger(__name__)

class SourceTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        trainable_params = self.model.trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=config.optimizer.lr, 
            weight_decay=config.optimizer.weight_decay
        )
        self.criterion = SegLoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training Source"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Deterministic forward pass for source training
            logits = self.model(images, deterministic=True)
            
            loss = self.criterion(logits, masks)
            loss.backward()
            
            # Grad clip
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.grad_clip)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def train(self, dataloader):
        epochs = self.config.training.source_epochs
        logger.info(f"Starting Source Training for {epochs} epochs...")
        os.makedirs(self.config.output.root, exist_ok=True)
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                save_checkpoint(self.model, self.optimizer, epoch, f"{self.config.output.root}/{self.config.output.experiment_name}_source_{epoch+1}.pt")
                
        # Save prior
        self.model.head.load_source_prior()
        save_checkpoint(self.model, self.optimizer, epochs, f"{self.config.output.root}/{self.config.output.experiment_name}_source_final.pt")
        logger.info("Source training completed and prior saved.")
