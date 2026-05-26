import logging
from ..core.config import load_config
from ..core.reproducibility import set_seed
from ..core.checkpointing import load_checkpoint
from ..models.registry import MODELS
from ..datasets.registry import DATASETS
from ..datasets.fewshot import FewShotSampler
from ..training.target_adapter import TargetAdapter
from ..evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def run_adaptation_for_seed(cfg, seed):
    set_seed(seed)
    
    DatasetClass = DATASETS.get(cfg.dataset.name)
    target_dataset = DatasetClass(
        root_dir=cfg.dataset.root, 
        split="train", 
        modality=cfg.dataset.target_modality,
        fast_dev_run=cfg.dataset.fast_dev_run,
        image_size=cfg.dataset.image_size,
        split_file=cfg.dataset.split_file
    )
    
    sampler = FewShotSampler(
        target_dataset, 
        k=cfg.fewshot.k, 
        num_classes=cfg.dataset.num_classes,
        seed=seed,
        patient_level=cfg.fewshot.patient_level
    )
    support_set = sampler.sample_support_set()
    support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    
    test_dataset = DatasetClass(
        root_dir=cfg.dataset.root, 
        split="test", 
        modality=cfg.dataset.target_modality,
        fast_dev_run=cfg.dataset.fast_dev_run,
        image_size=cfg.dataset.image_size,
        split_file=cfg.dataset.split_file
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    ModelClass = MODELS.get(cfg.model.name)
    model = ModelClass(
        num_classes=cfg.dataset.num_classes, 
        embed_dim=cfg.model.feature_dim, 
        prompt_dim=cfg.model.prompt_dim,
        lightweight_backend=cfg.model.lightweight_backend,
        sam2_checkpoint=cfg.model.sam2_checkpoint,
        sam2_config=cfg.model.sam2_config
    )
    
    if cfg.checkpoint_path:
        load_checkpoint(model, cfg.checkpoint_path)
        
    adapter = TargetAdapter(model, cfg, device=cfg.training.device)
    
    if cfg.adaptation.mode == "fast_prompt":
        adapter.fast_prompt(support_loader)
        eval_mode = "deterministic"
    else:
        adapter.bayesian_finetune(support_loader)
        eval_mode = "mc"
        
    evaluator = Evaluator(model, cfg, device=cfg.training.device)
    results = evaluator.evaluate(test_loader, mode=eval_mode)
    return results

def main(config_path="configs/default.yaml", overrides=None):
    if overrides is None:
        overrides = []
        
    logger.info(f"Loading config from {config_path}")
    logger.info(f"Applying overrides: {overrides}")
    
    cfg = load_config(config_path, overrides=overrides)
    
    from .experiment_runner import ExperimentRunner
    runner = ExperimentRunner(cfg)
    runner.run()

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    main(config_path, overrides)
