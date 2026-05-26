import logging
import os
import json
import pandas as pd
from ..core.config import load_config
from .experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

def run_ablation(config_path, overrides):
    logger.info("Starting True Deep Ablation Runner...")
    
    cfg = load_config(config_path, overrides)
    out_dir = os.path.join(cfg.output.root, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    
    directions = [
        {"source": "US", "target": "MRI"},
        {"source": "MRI", "target": "US"}
    ]
    
    ablations = [
        {"name": "Full BayesPrompt", "flags": ["adaptation.mode=bayesian_finetune"]},
        {"name": "w/o BMPA", "flags": ["adaptation.mode=bayesian_finetune", "adaptation.disable_bmpa=true"]},
        {"name": "w/o PPM", "flags": ["adaptation.mode=bayesian_finetune", "adaptation.disable_ppm=true"]},
        {"name": "w/o uncertainty weighting", "flags": ["adaptation.mode=bayesian_finetune", "adaptation.uncertainty_weighting=false"]},
        {"name": "Prompt-only", "flags": ["adaptation.mode=bayesian_finetune", "adaptation.prompt_only=true"]},
        {"name": "Fast Prompt only", "flags": ["adaptation.mode=fast_prompt"]},
        {"name": "Bayesian Fine-tuning", "flags": ["adaptation.mode=bayesian_finetune"]},
        {"name": "Source-only", "flags": ["adaptation.mode=source_only"]},
        {"name": "posterior mean inference", "flags": ["adaptation.mode=bayesian_finetune", "adaptation.inference_mode=mean"]},
        {"name": "MC posterior inference", "flags": ["adaptation.mode=bayesian_finetune", "adaptation.inference_mode=mc"]}
    ]
    
    k = 3
    
    if cfg.dataset.fast_dev_run:
        directions = [directions[0]]
        ablations = [ablations[0], ablations[1]]
    
    summary_list = []
    all_seeds_df_list = []
    
    for d in directions:
        for ab in ablations:
            logger.info(f"--- Ablation | {d['source']}->{d['target']} | {ab['name']} ---")
            
            curr_overrides = overrides.copy() if overrides else []
            curr_overrides.extend([
                f"dataset.source_modality={d['source']}",
                f"dataset.target_modality={d['target']}",
                f"fewshot.k={k}"
            ])
            curr_overrides.extend(ab["flags"])
            
            run_cfg = load_config(config_path, curr_overrides)
            runner = ExperimentRunner(run_cfg)
            res = runner.run()
            metrics = res["summary"]
            seed_df = res["seed_df"]
            
            seed_df["ablation_name"] = ab["name"]
            seed_df["direction"] = f"{d['source']}->{d['target']}"
            seed_df["k"] = k
            all_seeds_df_list.append(seed_df)
            
            summary_list.append({
                "ablation_name": ab["name"],
                "direction": f"{d['source']}->{d['target']}",
                "k": k,
                "dice_mean": metrics.get("dice", 0),
                "dice_ci95": metrics.get("dice_ci95", 0),
                "iou_mean": metrics.get("iou", 0),
                "iou_ci95": metrics.get("iou_ci95", 0),
                "ece_mean": metrics.get("ece", 0),
                "ece_ci95": metrics.get("ece_ci95", 0)
            })
            
    summary_df = pd.DataFrame(summary_list)
    combined_seeds_df = pd.concat(all_seeds_df_list, ignore_index=True) if all_seeds_df_list else pd.DataFrame()
    
    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_list, f, indent=4)
        
    csv_path = os.path.join(out_dir, "metrics_per_seed.csv")
    combined_seeds_df.to_csv(csv_path, index=False)
    
    logger.info(f"True Ablation pipeline completed. Results saved to {out_dir}")
