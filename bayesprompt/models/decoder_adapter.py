import torch
import torch.nn as nn
from .attention_injection import KVPromptInjector

class PromptConditionedMaskDecoder(nn.Module):
    """
    A robust, reproducible mask decoder adapter for BayesPrompt.
    
    Instead of brittle monkey-patching of the official SAM2 attention layers,
    we provide this clean adapter which acts as the target for our lightweight
    Bayesian parameter adaptation. It takes the frozen features from the encoder,
    and applies standard Cross-Attention where queries are the image features,
    and Keys/Values are projected from the Probabilistic Prompts.
    """
    def __init__(self, embed_dim=256, prompt_dim=256, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-attention over features
        self.feature_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # Cross-Attention for Prompts
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.injector = KVPromptInjector(prompt_dim, embed_dim)
        
        # Optional Self-Attention residual block to refine features after prompt injection
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Final segmentation head
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        )

    def forward(self, features, prompts):
        """
        features: [B, C_f, H, W]
        prompts: [B, C_classes, D_prompt] or [1, C_classes, D_prompt]
        """
        B, C_f, H, W = features.shape
        
        # Flatten features for attention: [B, H*W, C_f]
        feat_flat = self.feature_proj(features).flatten(2).transpose(1, 2)
        feat_flat = self.norm1(feat_flat)
        
        if prompts is not None:
            # Expand prompts if it's 1-shot aggregated [1, C_classes, D_prompt] -> [B, C_classes, D_prompt]
            if prompts.shape[0] == 1 and B > 1:
                prompts = prompts.expand(B, -1, -1)
                
            # Direct mapping to K, V without fallback tensors
            K, V = self.injector(prompts) # [B, C_classes, C_f]
            
            # Cross attention: Query = features, Key/Value = Prompts
            attn_out, _ = self.cross_attn(query=feat_flat, key=K, value=V)
            
            # Residual connection
            feat_flat = feat_flat + attn_out
            
        feat_flat = self.norm2(feat_flat)
        
        # Self-attention block
        self_attn_out, _ = self.self_attn(query=feat_flat, key=feat_flat, value=feat_flat)
        feat_flat = feat_flat + self_attn_out
            
        # Reshape back to spatial
        feat_spatial = feat_flat.transpose(1, 2).view(B, C_f, H, W)
        
        # Decode
        logits = self.output_conv(feat_spatial)
        
        return logits
