import torch

def compute_ece(probs, labels, n_bins=15, ignore_index=None, foreground_only=False):
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    probs: [B, C, H, W]
    labels: [B, H, W]
    """
    B, C, H, W = probs.shape
    
    # Flatten
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    
    mask = torch.ones_like(labels, dtype=torch.bool)
    if ignore_index is not None:
        mask = mask & (labels != ignore_index)
        
    if foreground_only:
        # Assuming background is class 0
        mask = mask & (labels > 0)
        
    probs = probs[mask]
    labels = labels[mask]
    
    if len(labels) == 0:
        return torch.tensor(0.0, device=probs.device)
        
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    
    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()
