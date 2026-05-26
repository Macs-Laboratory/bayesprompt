import torch

def compute_predictive_entropy(probs: torch.Tensor):
    """
    probs: [B, C, H, W] expected to be the mean of predicted probabilities from MC samples.
    Returns: [B, 1, H, W] entropy map
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
    return entropy

def compute_mc_predictions(mc_logits: torch.Tensor):
    """
    mc_logits: [R, B, C, H, W]
    Returns:
    mean_probs: [B, C, H, W]
    entropy: [B, 1, H, W]
    """
    mc_probs = torch.softmax(mc_logits, dim=2)
    mean_probs = mc_probs.mean(dim=0)
    entropy = compute_predictive_entropy(mean_probs)
    return mean_probs, entropy
