import torch
import torch.nn as nn

class PromptInjector(nn.Module):
    """Base class for Prompt Injectors."""
    def forward(self, prompts):
        raise NotImplementedError

class KVPromptInjector(PromptInjector):
    """
    Directly maps Prompts to Keys and Values for cross-attention.
    This avoids the conceptually awkward 'fallback_K' logic.
    """
    def __init__(self, prompt_dim: int, embed_dim: int):
        super().__init__()
        self.proj_k = nn.Linear(prompt_dim, embed_dim)
        self.proj_v = nn.Linear(prompt_dim, embed_dim)
        
    def forward(self, prompts):
        """
        prompts: [B, C_classes, D_prompt]
        returns:
            K: [B, C_classes, embed_dim]
            V: [B, C_classes, embed_dim]
        """
        K = self.proj_k(prompts)
        V = self.proj_v(prompts)
        return K, V

class AttentionBiasPromptInjector(PromptInjector):
    def __init__(self):
        super().__init__()
        # TODO: Implement attention bias injection if integrating directly into SAM2's core attention
        pass

class PrefixPromptInjector(PromptInjector):
    def __init__(self):
        super().__init__()
        # TODO: Implement prefix tuning logic
        pass
