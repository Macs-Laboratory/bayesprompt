import logging
from ..core.config import load_config
from ..core.reproducibility import set_seed
from ..models.registry import MODELS
from ..datasets.registry import DATASETS
from ..training.source_trainer import SourceTrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def main(config_path="configs/default.yaml", overrides=None):
    if overrides is None:
        overrides = []
    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path, overrides=overrides)
    set_seed(42)
    
    DatasetClass = DATASETS.get(cfg.dataset.name)
    train_dataset = DatasetClass(
        root_dir=cfg.dataset.root, 
        split="train", 
        modality=cfg.dataset.source_modality,
        fast_dev_run=cfg.dataset.fast_dev_run,
        image_size=cfg.dataset.image_size,
        split_file=cfg.dataset.split_file
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.optimizer.batch_size, shuffle=True)
    
    ModelClass = MODELS.get(cfg.model.name)
    model = ModelClass(
        num_classes=cfg.dataset.num_classes, 
        embed_dim=cfg.model.feature_dim, 
        prompt_dim=cfg.model.prompt_dim,
        lightweight_backend=cfg.model.lightweight_backend,
        sam2_checkpoint=cfg.model.sam2_checkpoint,
        sam2_config=cfg.model.sam2_config
    )
    
    trainer = SourceTrainer(model, cfg, device=cfg.training.device)
    trainer.train(train_loader)

if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    main(config, overrides)
