import logging
import os
import json
import pandas as pd
from ..core.config import load_config
from .experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

def run_crossmodality(config_path, overrides):
    logger.info("Starting True Deep Cross-Modality Runner...")
    
    cfg = load_config(config_path, overrides)
    out_dir = os.path.join(cfg.output.root, "crossmod")
    os.makedirs(out_dir, exist_ok=True)
    
    directions = [
        {"dataset_group": "External", "source": "CT", "target": "MRI"},
        {"dataset_group": "External", "source": "MRI", "target": "CT"},
        {"dataset_group": "Internal", "source": "US", "target": "MRI"},
        {"dataset_group": "Internal", "source": "MRI", "target": "US"}
    ]
    
    methods = [
        {"name": "U-Net", "model": "UNet", "mode": "fast_prompt"},
        {"name": "nnU-Net", "model": "nnUNet", "mode": "fast_prompt"},
        {"name": "SegFormer", "model": "SegFormer", "mode": "fast_prompt"},
        {"name": "CDUN", "model": "CDUN", "mode": "fast_prompt"},
        {"name": "MedSAM", "model": "MedSAM", "mode": "fast_prompt"},
        {"name": "SAM-Adapter", "model": "SAMAdapter", "mode": "fast_prompt"},
        {"name": "SAM2 Fine-tune", "model": "BayesPrompt", "mode": "bayesian_finetune", "disable_bmpa": "true", "disable_ppm": "true"},
        {"name": "Prompt-only", "model": "BayesPrompt", "mode": "bayesian_finetune", "prompt_only": "true"},
        {"name": "BayesPrompt Fast", "model": "BayesPrompt", "mode": "fast_prompt"},
        {"name": "BayesPrompt Bayesian", "model": "BayesPrompt", "mode": "bayesian_finetune"}
    ]
    
    k_values = [1, 3, 5, 10]
    
    if cfg.dataset.fast_dev_run:
        directions = [directions[0]]
        methods = [methods[-3], methods[-2], methods[-1]]
        k_values = [3]
    
    summary_list = []
    all_seeds_df_list = []
    
    for d in directions:
        for m in methods:
            for k in k_values:
                logger.info(f"--- Crossmod | {d['source']}->{d['target']} | {m['name']} | k={k} ---")
                
                curr_overrides = overrides.copy() if overrides else []
                curr_overrides.extend([
                    f"dataset.source_modality={d['source']}",
                    f"dataset.target_modality={d['target']}",
                    f"model.name={m['model']}",
                    f"adaptation.mode={m['mode']}",
                    f"fewshot.k={k}"
                ])
                
                if "disable_bmpa" in m:
                    curr_overrides.append(f"adaptation.disable_bmpa={m['disable_bmpa']}")
                if "disable_ppm" in m:
                    curr_overrides.append(f"adaptation.disable_ppm={m['disable_ppm']}")
                if "prompt_only" in m:
                    curr_overrides.append(f"adaptation.prompt_only={m['prompt_only']}")
                
                run_cfg = load_config(config_path, curr_overrides)
                runner = ExperimentRunner(run_cfg)
                res = runner.run()
                metrics = res["summary"]
                seed_df = res["seed_df"]
                
                # Add contextual columns to the seed_df
                seed_df["method"] = m["name"]
                seed_df["direction"] = f"{d['source']}->{d['target']}"
                seed_df["k"] = k
                all_seeds_df_list.append(seed_df)
                
                summary_list.append({
                    "dataset_group": d["dataset_group"],
                    "direction": f"{d['source']}->{d['target']}",
                    "method": m["name"],
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
    
    # Save Summary
    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_list, f, indent=4)
        
    csv_path = os.path.join(out_dir, "metrics_per_seed.csv")
    combined_seeds_df.to_csv(csv_path, index=False)
    
    logger.info(f"True Cross-Modality pipeline completed. Results saved to {out_dir}")
