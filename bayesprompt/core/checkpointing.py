import torch
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer = None) -> int:
    if not os.path.exists(path):
        logger.warning(f"Checkpoint not found at {path}")
        return 0
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model_state'])
    if optimizer and state.get('optimizer_state'):
        optimizer.load_state_dict(state['optimizer_state'])
    logger.info(f"Loaded checkpoint from {path} (epoch {state.get('epoch', 0)})")
    return state.get('epoch', 0)
