import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

BASELINE_GROUPS = {
    "General Segmentation": ["U-Net", "nnU-Net", "SegFormer"],
    "Cross-Domain Adaptation": ["CDUN", "VP-SFDA", "MAUP"],
    "SAM-Family": ["SAM2 Zero-shot", "SAM2 Fine-tune", "MedSAM", "SAM-Adapter", "SAM Few-shot", "MedSAM-U", "Prompt-only"],
    "Proposed": ["BayesPrompt"]
}

def run_check_baselines(metrics_dir: str, out_path: str):
    logger.info(f"Checking baseline coverage in {metrics_dir}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    summary_path = os.path.join(metrics_dir, "summary.json")
    present_methods = set()
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                df = pd.DataFrame(json.load(f))
                if 'method' in df.columns:
                    present_methods = set(df['method'].unique())
        except Exception as e:
            logger.warning(f"Failed to read {summary_path}: {e}")
            
    report = ["# Baseline Coverage Report\n"]
    
    for group, methods in BASELINE_GROUPS.items():
        report.append(f"## {group}")
        for m in methods:
            status = "✅ Present" if m in present_methods else "❌ Missing (or External Template)"
            report.append(f"- **{m}**: {status}")
        report.append("")
        
    report.append("*(Note: Missing methods may be implemented as external templates and are expected to be absent from internal metrics logs until externally reproduced metrics are manually appended.)*")
    
    with open(out_path, 'w') as f:
        f.write("\n".join(report))
        
    logger.info(f"✅ Baseline check complete. Report written to {out_path}")
