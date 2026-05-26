import logging
import os
import shutil

logger = logging.getLogger(__name__)

def run_cleanup(config_path, overrides):
    logger.info("Starting Cleanup Runner...")
    
    targets = [
        "outputs",
        
        "bayesprompt/__pycache__",
        "bayesprompt/core/__pycache__",
        "bayesprompt/datasets/__pycache__",
        "bayesprompt/models/__pycache__",
        "bayesprompt/training/__pycache__",
        "bayesprompt/experiments/__pycache__",
        "bayesprompt/evaluation/__pycache__",
        "bayesprompt/visualization/__pycache__",
        
    ]
    
    for target in targets:
        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
                logger.info(f"Deleted directory: {target}")
            else:
                os.remove(target)
                logger.info(f"Deleted file: {target}")
                
    logger.info("Cleanup completed successfully.")
