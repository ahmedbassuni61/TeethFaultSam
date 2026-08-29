# Synthetic Teeth Fault Generation Pipeline

This document provides an overview of the synthetic faults generation pipeline implemented in `synthetic_faults_pipeline.ipynb`. 

The pipeline is designed to load a 3D dental mesh with its corresponding segmentation labels, visualize the segmentation, and systematically apply realistic synthetic dental faults to generate training data for the 3DTeethSAM model.

## Core Features

The pipeline ensures high-quality synthetic data through the following key constraints:

1. **Gum Protection:** All fault generation algorithms are explicitly masked to affect *only* the teeth vertices (`labels > 0`). The gum region (`label == 0`) is perfectly preserved to maintain mesh integrity.
2. **Multi-Tooth Span:** Faults are not limited to single teeth. Depending on their severity radius, defects like cavities and cracks can naturally span across multiple adjacent teeth.
3. **Jaw Orientation Control:** The algorithms auto-detect or accept parameters for jaw orientation (`upper` or `lower`), ensuring faults target the correct surfaces (e.g., crowns vs. roots).

## Pipeline Stages

1. **Data Extraction & Imports:** Loading of `.obj` meshes and `.json` label files.
2. **Utilities & Segmentation Visualization:** Interactive 3D visualization using `plotly` to verify healthy segmentation.
3. **Fault Generators:** Application of synthetic faults (`generate_cavity`, `generate_crack`, `generate_abrasion`, `generate_calculus`).
4. **Pipeline Execution:** Execution and visualization of the resulting faulty mesh and fault mask.
