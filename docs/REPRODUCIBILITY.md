# Reproducibility Guide

This document outlines the hardware, environment, and procedural steps required for pipeline and public benchmark reproduction from the paper *Uncertainty-Aware Bayesian Prompt Adaptation for Robust Cross-Modality Medical Segmentation*.

## 1. Hardware & Environment

- **Source Training:** The source models were trained using 4x NVIDIA RTX A6000 GPUs to handle the large-scale source dataset (e.g., CT volumes) with adequate batch sizes.
- **Target Adaptation:** Because the SAM2 encoder is completely frozen and only the lightweight decoder and prompt module are updated, the $k$-shot target adaptation requires significantly less memory and can run comfortably on a single RTX 3090 or RTX 4090 GPU (24GB VRAM).

## 2. Dataset Structure

Ensure your `data/` directory follows the exact hierarchy:
```text
data/
  amos/               # External CT
    images/
    masks/
    splits.json
  brats/              # External MRI
    images/
    masks/
    splits.json
  rotator_cuff/       # Internal Rotator Cuff US/MRI
    images/
    masks/
    splits.json
```
For specific metadata about AMOS vs. BraTS vs. Internal RCT datasets, please see [DATASETS.md](DATASETS.md).

## 3. Required Split JSON Schema

All splits must be strictly patient-wise to avoid data leakage. The `splits.json` format must include keys for `train`, `val`, and `test` (and optionally `support_sets` for pre-defined few-shot indices).
To validate your splits, use the utility script:
```bash
python main.py validate-splits --split data/amos/splits.json
```

## 4. Commands to Reproduce Experiments

1. **Source Training**
   ```bash
   python main.py train --config bayesprompt/configs/external_ct_mri.yaml dataset.source_modality=CT
   ```
   *Note: The example config specifies `training.source_epochs=5` for rapid execution and smoke testing. Set `training.source_epochs=100` to run the full training protocol described in the paper.*

2. **Target Adaptation**
   ```bash
   python main.py adapt --config bayesprompt/configs/external_ct_mri.yaml dataset.source_modality=CT dataset.target_modality=MRI fewshot.k=3
   ```
3. **Cross-Modality Evaluation**
   ```bash
   python main.py crossmod --config bayesprompt/configs/external_ct_mri.yaml
   ```
4. **Calibration / ECE**
   ```bash
   python main.py eval --config bayesprompt/configs/external_ct_mri.yaml
   ```
5. **Ablation Studies**
   ```bash
   python main.py ablate --config bayesprompt/configs/internal_us_mri.yaml fewshot.k=3
   ```
6. **Qualitative Figures**
   ```bash
   python main.py visualize --config bayesprompt/configs/internal_us_mri.yaml
   ```

## 5. Random Seeds and Confidence Intervals

- **Seeds:** All experiments use 5 pre-defined random seeds for drawing the $k$-shot support set. This ensures statistical robustness against support sampling variance.
- **95% CI Computation:** Mean values and confidence intervals across the 5 seeds are computed strictly using standard error $\pm 1.96 \times \frac{\sigma}{\sqrt{n}}$, explicitly implemented in the evaluation runner.
