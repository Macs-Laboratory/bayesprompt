import logging
import os
from ..core.config import load_config
from ..evaluation.table_builder import TableBuilder

logger = logging.getLogger(__name__)

def run_tables(config_path, overrides):
    logger.info("Starting True Table Reproduction Runner...")
    cfg = load_config(config_path, overrides)
    
    out_dir = os.path.join(cfg.output.root, "tables")
    os.makedirs(out_dir, exist_ok=True)
    
    builder = TableBuilder(out_dir)
    
    crossmod_summary = os.path.join(cfg.output.root, "crossmod", "summary.json")
    ablation_summary = os.path.join(cfg.output.root, "ablation", "summary.json")
    
    crossmod_tex = builder.build_crossmod_table(crossmod_summary)
    with open(os.path.join(out_dir, "table_crossmod_dice.tex"), "w") as f:
        f.write(crossmod_tex)
        
    ablation_tex = builder.build_ablation_table(ablation_summary)
    with open(os.path.join(out_dir, "table_ablation.tex"), "w") as f:
        f.write(ablation_tex)
        
    calib_tex = builder.build_calibration_stability_table(crossmod_summary)
    with open(os.path.join(out_dir, "table_calibration_stability.tex"), "w") as f:
        f.write(calib_tex)
        
    eff_tex = builder.build_efficiency_table(crossmod_summary)
    with open(os.path.join(out_dir, "table_efficiency.tex"), "w") as f:
        f.write(eff_tex)
        
    logger.info(f"Generated dynamic LaTeX tables at {out_dir}")
