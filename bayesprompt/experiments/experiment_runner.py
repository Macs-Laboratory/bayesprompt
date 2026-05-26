import logging
import scipy.stats as stats
import numpy as np
import pandas as pd
import os
from .run_target_adaptation import run_adaptation_for_seed

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Orchestrates the k-shot, multi-seed evaluation required by the paper.
    Computes 95% Confidence Intervals over the repeated seeds.
    """
    def __init__(self, config):
        self.config = config

    def run(self):
        results = []
        
        logger.info(f"Running Experiment: {self.config.output.experiment_name}")
        logger.info(f"Mode: {self.config.adaptation.mode}")
        logger.info(f"K-Shot: {self.config.fewshot.k}")
        logger.info(f"Seeds: {self.config.fewshot.seeds}")
        
        for seed in self.config.fewshot.seeds:
            logger.info(f"========== RUNNING SEED {seed} ==========")
            res = run_adaptation_for_seed(self.config, seed)
            res['seed'] = seed
            results.append(res)
            
        df = pd.DataFrame(results)
        
        # Calculate Mean and 95% CI
        summary = {"num_seeds": len(df)}
        for metric in ['dice', 'iou', 'ece']:
            # The evaluator outputs dice_mean, iou_mean, ece_mean
            col_name = f"{metric}_mean"
            if col_name in df.columns:
                n = len(df)
                mean_val = df[col_name].mean()
                if n > 1:
                    std_val = df[col_name].std(ddof=0)
                    ci95 = 1.96 * (std_val / np.sqrt(n))
                else:
                    ci95 = 0.0
                    
                summary[metric] = mean_val
                summary[f"{metric}_ci95"] = ci95
                
                logger.info(f"{metric}: {mean_val*100:.2f} ± {ci95*100:.2f}")
            
        # Provide seed level DataFrame along with summary for callers
        return {
            "summary": summary,
            "seed_df": df
        }
