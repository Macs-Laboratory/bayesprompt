import torch
import torch.nn as nn
import logging
from .sam2_wrapper import SAM2Wrapper
from .bayesian_head import BayesianParameterModule
from .prompt_module import ProbabilisticPromptModule
from .decoder_adapter import PromptConditionedMaskDecoder
from ..core.decorators import register_model

logger = logging.getLogger(__name__)

@register_model("BayesPrompt")
class BayesPromptModel(nn.Module):
    def __init__(self, num_classes=2, embed_dim=256, prompt_dim=256, lightweight_backend=True, sam2_checkpoint=None, sam2_config=None):
        super().__init__()
        
        self.encoder = SAM2Wrapper(lightweight_backend=lightweight_backend, embed_dim=embed_dim, sam2_checkpoint=sam2_checkpoint, sam2_config=sam2_config)
        
        decoder = PromptConditionedMaskDecoder(embed_dim=embed_dim, prompt_dim=prompt_dim, num_classes=num_classes)
        self.head = BayesianParameterModule(decoder)
        
        self.prompt_module = ProbabilisticPromptModule(embed_dim, prompt_dim)
        self.current_prompts = None

    def freeze_encoder(self):
        self.encoder._freeze_encoder()
        
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
        
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.trainable_parameters())
        
    def trainable_parameter_ratio(self):
        trainable = self.count_trainable_parameters()
        total = sum(p.numel() for p in self.parameters())
        return trainable / max(1, total)

    def extract_features(self, x):
        return self.encoder(x)

    def forward(self, x, prompts=None, num_samples=1, deterministic=False):
        """
        x: [B, 3, H, W]
        """
        features = self.extract_features(x)
        
        if prompts is None:
            if self.current_prompts is not None:
                prompts = self.current_prompts
            else:
                logger.warning("No prompts provided and no cached prompts found. Proceeding with promptless inference (may degrade performance).")

        logits = self.head(features, prompts=prompts, num_samples=num_samples, deterministic=deterministic)
        
        # Upsample to original resolution
        if num_samples > 1:
            R, B, C, H_feat, W_feat = logits.shape
            logits = logits.view(R*B, C, H_feat, W_feat)
            logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
            logits = logits.view(R, B, C, x.shape[-2], x.shape[-1])
        else:
            logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return logits

    def compute_prompts(self, support_images, support_masks, num_mc=4, deterministic=False):
        """
        Extracts features, runs forward pass to get uncertainty, and computes Prompts.
        """
        with torch.no_grad():
            features = self.extract_features(support_images)
            
            # Predict to get uncertainty at feature resolution
            logits = self.head(features, prompts=None, num_samples=1 if deterministic else num_mc, deterministic=deterministic)
            
            if deterministic:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.softmax(logits, dim=2).mean(dim=0)
            
            # Downsample masks to feature resolution
            _, _, H_f, W_f = features.shape
            masks_down = torch.nn.functional.interpolate(support_masks.float(), size=(H_f, W_f), mode='nearest')
            
            # support_masks is assumed to be [B, C, H, W] one-hot in PPM
            prompts = self.prompt_module(features, masks_down, probs)
            self.current_prompts = prompts
        return prompts
        
    def predict(self, x):
        """Posterior mean prediction (deterministic)."""
        self.eval()
        with torch.no_grad():
            logits = self(x, num_samples=1, deterministic=True)
            return torch.softmax(logits, dim=1)
            
    def predict_mc(self, x, num_samples=8):
        """MC posterior prediction."""
        self.eval()
        with torch.no_grad():
            logits = self(x, num_samples=num_samples, deterministic=False)
            probs = torch.softmax(logits, dim=2)
            return probs.mean(dim=0)
            
    def save_source_prior(self, path):
        self.head.load_source_prior() # copies posterior to prior
        torch.save(self.state_dict(), path)
        
    def load_source_prior(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.freeze_encoder()
