import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def run_build_tables(metrics_dir: str, out_dir: str, template: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Building tables from {metrics_dir} -> {out_dir}")
    
    summary_path = os.path.join(metrics_dir, "summary.json")
    has_data = os.path.exists(summary_path) and not template
    
    df = pd.DataFrame()
    if has_data:
        try:
            with open(summary_path, 'r') as f:
                df = pd.DataFrame(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to read {summary_path}: {e}")
            has_data = False

    # 1. Table 1: Cross-modality
    table1 = _build_table1(df, has_data)
    with open(os.path.join(out_dir, "table1_crossmodality.tex"), "w") as f:
        f.write(table1)
        
    # 2. Table 2: Calibration
    table2 = _build_table2(df, has_data)
    with open(os.path.join(out_dir, "table2_calibration_stability.tex"), "w") as f:
        f.write(table2)
        
    # 3. Table 3: Ablation
    table3 = _build_table3(df, has_data)
    with open(os.path.join(out_dir, "table3_ablation.tex"), "w") as f:
        f.write(table3)
        
    logger.info("✅ Tables generated successfully.")

def _build_table1(df, has_data):
    tex = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|cccc|cccc|cccc|cccc}",
        r"\hline",
        r"\multirow{2}{*}{Method} & \multicolumn{4}{c|}{AMOS CT $\rightarrow$ MRI} & \multicolumn{4}{c|}{AMOS MRI $\rightarrow$ CT} & \multicolumn{4}{c|}{RCT US $\rightarrow$ MRI} & \multicolumn{4}{c}{RCT MRI $\rightarrow$ US} \\",
        r" & 1-shot & 3-shot & 5-shot & 10-shot & 1-shot & 3-shot & 5-shot & 10-shot & 1-shot & 3-shot & 5-shot & 10-shot & 1-shot & 3-shot & 5-shot & 10-shot \\",
        r"\hline"
    ]
    methods = ["U-Net", "nnU-Net", "SegFormer", "CDUN", "VP-SFDA", "MAUP", "SAM2 Zero-shot", "SAM2 Fine-tune", "MedSAM", "SAM-Adapter", "SAM Few-shot", "MedSAM-U", "Prompt-only", "BayesPrompt"]
    if has_data and 'method' in df.columns:
        for m in [x for x in methods if x in df['method'].unique()]:
            row = f"{m} "
            for d in ["CT->MRI", "MRI->CT", "US->MRI", "MRI->US"]:
                for k in [1, 3, 5, 10]:
                    sub = df[(df['method'] == m) & (df['direction'] == d) & (df['k'] == k)]
                    if len(sub) > 0:
                        val = sub.iloc[0]
                        row += f"& {val['dice_mean']*100:.1f} $\\pm$ {val['dice_ci95']*100:.1f} "
                    else:
                        row += "& - "
            tex.append(row + r"\\")
    else:
        tex.append(r"\multicolumn{17}{c}{Data not available - run experiments or use \texttt{--template}} \\")
        
    tex.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\caption{Table 1: Comprehensive Cross-Modality Dice}",
        r"\end{table*}"
    ])
    return "\n".join(tex)

def _build_table2(df, has_data):
    tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\hline",
        r"\multirow{2}{*}{Method} & \multicolumn{3}{c|}{RCT US $\rightarrow$ MRI} & \multicolumn{3}{c}{RCT MRI $\rightarrow$ US} \\",
        r" & Dice (1-shot) & Dice (3-shot) & ECE (3-shot) & Dice (1-shot) & Dice (3-shot) & ECE (3-shot) \\",
        r"\hline"
    ]
    methods = ["SAM2 Fine-tune", "Prompt-only", "BayesPrompt"]
    if has_data and 'method' in df.columns:
        for m in [x for x in methods if x in df['method'].unique()]:
            row = f"{m} "
            for d in ["US->MRI", "MRI->US"]:
                sub_1 = df[(df['method'] == m) & (df['direction'] == d) & (df['k'] == 1)]
                sub_3 = df[(df['method'] == m) & (df['direction'] == d) & (df['k'] == 3)]
                dice_1 = f"{sub_1.iloc[0]['dice_mean']*100:.1f} $\\pm$ {sub_1.iloc[0]['dice_ci95']*100:.1f}" if len(sub_1) > 0 else "-"
                dice_3 = f"{sub_3.iloc[0]['dice_mean']*100:.1f} $\\pm$ {sub_3.iloc[0]['dice_ci95']*100:.1f}" if len(sub_3) > 0 else "-"
                ece_3 = f"{sub_3.iloc[0]['ece_mean']*100:.2f} $\\pm$ {sub_3.iloc[0]['ece_ci95']*100:.2f}" if len(sub_3) > 0 else "-"
                row += f"& {dice_1} & {dice_3} & {ece_3} "
            tex.append(row + r"\\")
    else:
        tex.append(r"\multicolumn{7}{c}{Data not available} \\")
        
    tex.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Table 2: Calibration and Stability}",
        r"\end{table}"
    ])
    return "\n".join(tex)

def _build_table3(df, has_data):
    tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l|cc|cc}",
        r"\hline",
        r"\multirow{2}{*}{Ablation} & \multicolumn{2}{c|}{RCT US $\rightarrow$ MRI} & \multicolumn{2}{c}{RCT MRI $\rightarrow$ US} \\",
        r" & Dice & ECE & Dice & ECE \\",
        r"\hline"
    ]
    if has_data and 'ablation_name' in df.columns:
        for n in df['ablation_name'].unique():
            row = f"{n} "
            for d in ["US->MRI", "MRI->US"]:
                sub = df[(df['ablation_name'] == n) & (df['direction'] == d)]
                if len(sub) > 0:
                    val = sub.iloc[0]
                    row += f"& {val['dice_mean']*100:.1f} $\\pm$ {val['dice_ci95']*100:.1f} & {val['ece_mean']*100:.2f} $\\pm$ {val['ece_ci95']*100:.2f} "
                else:
                    row += "& - & - "
            tex.append(row + r"\\")
    else:
        tex.append(r"\multicolumn{5}{c}{Data not available} \\")
        
    tex.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Table 3: Ablation by Direction}",
        r"\end{table}"
    ])
    return "\n".join(tex)
