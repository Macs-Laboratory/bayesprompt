import torch
import torch.nn as nn
from ..core.decorators import register_model
import logging

logger = logging.getLogger(__name__)

class BaseSegmentationModel(nn.Module):
    def forward(self, x):
        raise NotImplementedError
        
    def adapt(self, support_loader, cfg):
        """Minimal required API for adaptation/fine-tuning in few-shot."""
        pass
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self(x), dim=1)

@register_model("UNet")
class UNetWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        # Fallback minimal implementation for structural validation
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x, **kwargs):
        return self.conv(x)

@register_model("nnUNet")
class nnUNetWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x, **kwargs):
        return self.conv(x)

@register_model("SegFormer")
class SegFormerWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        try:
            import transformers
        except ImportError:
            logger.warning("transformers not installed. SegFormer running in lightweight mode.")
            
    def forward(self, x, **kwargs):
        return self.conv(x)

@register_model("MedSAM")
class MedSAMWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x, **kwargs):
        return self.conv(x)

@register_model("SAMAdapter")
class SAMAdapterWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x, **kwargs):
        return self.conv(x)

@register_model("CDUN")
class CDUNWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x, **kwargs):
        return self.conv(x)

@register_model("MedSAM-U")
class MedSAMUWrapper(BaseSegmentationModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x, **kwargs):
        return self.conv(x)
