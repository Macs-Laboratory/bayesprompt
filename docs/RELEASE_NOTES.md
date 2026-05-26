# Release Notes & Repository Features

This document highlights the major features and reproducibility improvements implemented in the BayesPrompt codebase.

- [x] **Reproducibility Framework:** The entire workflow is now orchestrated by deterministic experiment runners enforcing seeds and standardized metric tracking.
- [x] **Dataset Separations:** The distinction between public (AMOS/BraTS) and private (Rotator Cuff Tear) datasets is clearly documented in `DATASETS.md`.
- [x] **Multi-Modality Benchmarking:** AMOS is defined as the CT/MRI abdominal cross-modality benchmark, and BraTS is recognized as an MRI benchmark. Direct cross-modality evaluations are handled via distinct source and target configuration logic.
- [x] **Baseline Groups Organization:** `BASELINES.md` groups baselines into logical categories (General, Cross-domain, SAM-Family, Proposed) and `check_baselines.py` programmatically ensures their presence in the final reports.
- [x] **Patient-Wise Split Validator:** `bayesprompt/datasets/validate_splits.py` rigorously tests split JSON files to prevent data leakage between train, validation, test, and support splits.
- [x] **Iteration Sensitivity Analysis:** `bayesprompt/experiments/run_iteration_sensitivity.py` evaluates target adaptation robustness at various step lengths (e.g., [5, 10, 20, 50] iterations).
- [x] **Efficiency Profiler:** `bayesprompt/evaluation/efficiency_report.py` programmatically counts trainable vs. frozen parameters to verify the adaptation parameter budget.
- [x] **Qualitative Visualization Suite:** `bayesprompt/visualization/qualitative_grid.py` supports distinct MRI and Ultrasound paneling, ground-truth/prediction overlays, and zoomed region-of-interest (ROI) views.
- [x] **LaTeX Table Builder:** `bayesprompt/evaluation/build_tables.py` standardizes LaTeX table population directly from raw metrics logs.
