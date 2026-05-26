import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SAM2Wrapper(nn.Module):
    """
    Wrapper for SAM2 Image Encoder. We freeze the encoder by default.
    If lightweight_backend=True, uses a lightweight CNN matching spatial downsampling behavior
    so that structural validation can run without downloading weights.
    """
    def __init__(self, lightweight_backend=True, embed_dim=256, sam2_checkpoint=None, sam2_config=None):
        super().__init__()
        self.lightweight_backend = lightweight_backend
        self.embed_dim = embed_dim
        
        if lightweight_backend:
            logger.info("Initializing Lightweight SAM2 Encoder for structural validation.")
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1)
            )
            # Spatial downsampling to H/16 similar to SAM
            self.downsample = nn.MaxPool2d(4, 4)
        else:
            logger.info("Loading official SAM2 implementation.")
            try:
                from sam2.build_sam import build_sam2
            except ImportError:
                raise ImportError(
                    "Official SAM2 package not found. Please install from https://github.com/facebookresearch/segment-anything-2 "
                    "or set `model.lightweight_backend=True` in your config to run structural validation."
                )
            
            if not sam2_checkpoint or not sam2_config:
                logger.warning("SAM2 checkpoint or config not provided. Relying on default package behavior (which may fail).")
                
            sam2_model = build_sam2(sam2_config, sam2_checkpoint, device="cpu")
            self.encoder = sam2_model.image_encoder
            
        self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Verify freezing
        frozen = all(not p.requires_grad for p in self.encoder.parameters())
        if not frozen:
            raise RuntimeError("Failed to freeze SAM2 encoder parameters!")

    def forward(self, x):
        """
        Extracts features from the frozen encoder.
        x: [B, 3, H, W]
        Returns:
        features: [B, C_f, H', W']
        """
        if self.lightweight_backend:
            feat = self.encoder(x)
            feat = self.downsample(feat)
            return feat
        else:
            # The real SAM2 image_encoder returns multiple feature levels or a dictionary
            # For this wrapper, we assume we want the highest level semantic features
            # which are typically returned in a specific format.
            # We abstract it here to just return the dense feature map for the mask decoder.
            # Usually SAM2 returns: dict with 'vision_features', 'vision_pos_enc'
            # Or a list of hierarchical features. Let's assume it returns a dict.
            outputs = self.encoder(x)
            
            # Since SAM2 internals vary by version, we handle common shapes:
            if isinstance(outputs, dict):
                feat = outputs.get('vision_features', None)
                if feat is None:
                    # Fallback to the last tensor if dict
                    feat = list(outputs.values())[-1]
            elif isinstance(outputs, (list, tuple)):
                feat = outputs[-1]
            else:
                feat = outputs
                
            return feat
