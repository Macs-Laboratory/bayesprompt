import logging
import os
import torch
import json
from ..core.config import load_config
from ..datasets.registry import DATASETS
from ..models.registry import MODELS
from ..evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def run_evaluation(config_path, overrides):
    logger.info("Starting Standalone Evaluation Runner...")
    cfg = load_config(config_path, overrides)
    
    # Setup Dataset
    DatasetClass = DATASETS.get(cfg.dataset.name)
    test_dataset = DatasetClass(
        root_dir=cfg.dataset.root,
        split="test",
        modality=cfg.dataset.target_modality,
        fast_dev_run=cfg.dataset.fast_dev_run,
        image_size=cfg.dataset.image_size,
        split_file=cfg.dataset.split_file
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Setup Model
    ModelClass = MODELS.get(cfg.model.name)
    model = ModelClass(
        num_classes=cfg.dataset.num_classes,
        embed_dim=cfg.model.feature_dim,
        prompt_dim=cfg.model.prompt_dim,
        lightweight_backend=cfg.model.lightweight_backend
    ).to(cfg.training.device)
    
    # Evaluate
    evaluator = Evaluator(model, cfg, device=cfg.training.device)
    
    # If the user sets adaptation.mode to fast_prompt, run deterministic. Else MC.
    mode = "deterministic" if cfg.adaptation.mode == "fast_prompt" else "mc"
    results = evaluator.evaluate(test_loader, mode=mode)
    
    # Save standalone results
    out_dir = os.path.join(cfg.output.root, "eval")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "standalone_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Evaluation completed. Metrics saved to {out_file}")
