# Dataset Organization and Usage

This document clarifies the roles of the public and internal datasets used in BayesPrompt.

## 1. AMOS Dataset (Public)
- **Modality:** CT and MRI
- **Task:** Abdominal multi-organ segmentation
- **Usage:** AMOS provides CT/MRI abdominal multi-organ data. The camera-ready paper reports public cross-modality transfer (e.g. CT $\leftrightarrow$ MRI) in compressed form.
- **Note:** BraTS and AMOS represent distinct anatomy and sequences; they are not directly interchangeable datasets. Standard evaluations utilize CT/MRI splits within AMOS or sequence transitions within BraTS.

## 2. BraTS Dataset (Public)
- **Modality:** MRI (multiple sequences: T1, T1ce, T2, FLAIR)
- **Task:** Brain tumor segmentation
- **Usage:** BraTS provides multi-sequence MRI brain tumor data for sequence-to-sequence evaluation.

## 3. Internal RCT Dataset (Private)
- **Modality:** Ultrasound (US) and MRI
- **Task:** Rotator Cuff Tear segmentation
- **Usage:** Used for **US $\leftrightarrow$ MRI** cross-modality transfer experiments.
- **Availability:** This dataset is acquired from an anonymized tertiary hospital cohort. It is **not publicly released** due to strict institutional data-use restrictions and IRB requirements. 

## 4. Expected Folder Structure
```text
data/
  amos/
    images/
    masks/
    splits.json
  brats/
    images/
    masks/
    splits.json
  rotator_cuff/
    images/
    masks/
    splits.json
```

**NOTE:** Ensure you match dataset domains and configurations correctly within your YAML files to align with the desired benchmark transfer protocol.
