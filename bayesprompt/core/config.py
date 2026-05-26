from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf
import os

@dataclass
class DatasetConfig:
    name: str = "AMOS"
    root: str = "./data/amos"
    source_modality: str = "CT"
    target_modality: str = "MRI"
    image_size: int = 512
    num_classes: int = 2
    include_background: bool = True
    split_file: str = "splits.json"
    fast_dev_run: bool = False

@dataclass
class FewShotConfig:
    k: int = 3
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    require_positive: bool = True
    patient_level: bool = True

@dataclass
class AdaptationConfig:
    mode: str = "bayesian_finetune"  # fast_prompt or bayesian_finetune
    iterations: int = 20
    lambda_kl: float = 0.1
    mc_samples_train: int = 4
    mc_samples_eval: int = 8
    recompute_prompts: bool = False
    uncertainty_weighting: bool = True

@dataclass
class ModelConfig:
    name: str = "BayesPrompt"
    sam2_checkpoint: str = ""
    sam2_config: str = ""
    lightweight_backend: bool = True
    prompt_dim: int = 256
    feature_dim: int = 256
    trainable_components: List[str] = field(default_factory=lambda: ["decoder"])
    injector: str = "kv"
    include_background_prompt: bool = False

@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    grad_clip: float = 1.0
    amp: bool = True

@dataclass
class TrainingConfig:
    source_epochs: int = 100
    num_workers: int = 4
    distributed: bool = False
    device: str = "cuda"

@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["dice", "iou", "ece"])
    save_predictions: bool = True
    save_uncertainty: bool = True
    save_visualizations: bool = True
    ece_bins: int = 15

@dataclass
class OutputConfig:
    root: str = "./outputs"
    experiment_name: str = "default_exp"

@dataclass
class BayesPromptConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    checkpoint_path: Optional[str] = None

def load_config(config_path: str = None, overrides: List[str] = None) -> BayesPromptConfig:
    """
    Loads config using OmegaConf to merge defaults, yaml file, and cli overrides.
    """
    schema = OmegaConf.structured(BayesPromptConfig)
    
    if config_path and os.path.exists(config_path):
        yaml_conf = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(schema, yaml_conf)
    else:
        cfg = schema
        
    if overrides:
        # overrides is expected to be a list of "key=value" strings
        override_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_conf)
        
    return cfg
