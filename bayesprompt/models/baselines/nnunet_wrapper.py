import torch.nn as nn
from ...core.decorators import register_model
import logging

logger = logging.getLogger(__name__)

@register_model("nnUNet")
class nnUNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        logger.warning("nnUNet is a wrapper. Please install nnUNetv2 and link the model here.")
        
    def forward(self, x):
        raise NotImplementedError("Link to nnUNet inference.")

@register_model("MedSAM")
class MedSAMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        logger.warning("MedSAM is a wrapper. Please install MedSAM and link here.")
        
    def forward(self, x):
        raise NotImplementedError("Link to MedSAM inference.")
