import logging
from .run_crossmodality import run_crossmodality
from .run_ablation import run_ablation
from .run_tables import run_tables
from .run_visualization import run_visualization

logger = logging.getLogger(__name__)

def run_reproduce(config_path, overrides):
    logger.info("Starting Reproduce Pipeline...")
    
    # 1. Run Experiments
    logger.info("Executing Cross-Modality Evaluation...")
    run_crossmodality(config_path, overrides)
    
    logger.info("Executing Ablations...")
    run_ablation(config_path, overrides)
    
    # 2. Render Outputs
    logger.info("Rendering Tables...")
    run_tables(config_path, overrides)
    
    logger.info("Rendering Visualizations...")
    run_visualization(config_path, overrides)
    
    logger.info("Reproduce Pipeline Completed Successfully.")
