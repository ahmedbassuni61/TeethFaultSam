# 🦷 3DTeethSAM: Project & Context Documentation

## 📌 Overview
This repository contains the local clone of `3dteethsam`, a deep learning framework dedicated to the segmentation and processing of 3D dental meshes via a frozen SAM2 (Hiera-L) backbone. The core objective is to accurately identify, segment, and extract 3D tooth models from broader dental scans.

## 💻 Environment & Workflow Constraints
**CRITICAL AGENT INSTRUCTION:** 
* **Documentation Rule:** This `Instructions.md` document is the source of truth and must be updated regularly to reflect the current project status and completed tasks.
* **Execution Environment:** Code is primarily being executed on **Google Colab**.
* **File Storage:** The repository and raw files reside **locally** on the user's PC.
* **Modification Rule:** DO NOT destructively modify the local repository files. Add new scripts for training/processing and use libraries like `peft` to inject modifications at runtime. 

## 🧬 Current Objective: Synthetic Fault Dataset & Custom Head Fine-tuning
The goal is to extend the base 3DTeethSAM architecture to detect and segment specific dental faults. 

**Phase 1: Synthetic Dataset Generation Pipeline**
To ensure accurate geometry, the pipeline now follows these steps (detailed documentation from `synthetic_faults_pipeline.ipynb` is available in `docs/pipeline_overview.md` and `docs/fault_generators.md`):
1. **Manual Curation (Gold Standard Subset):** Use the `Calculus_Curation_UI.ipynb` to manually inspect and flag raw meshes as "Clean" or "Has Calculus" into a `curation_ledger.csv`. This prevents algorithmic smoothing distortion by ensuring we apply synthetic tartar only on a mathematically perfect baseline.
2. **Segmentation Visualization:** First, visualize the healthy mesh segmentation to confirm proper identification of teeth vs. gum.
3. **Targeted Fault Generation (`trimesh`):** The script (`scripts/generate_synthetic_dataset.py`) filters for "Clean" meshes. Currently, it is **isolated to exclusively focus on `generate_calculus`** to prevent noisy baseline issues. Faults are masked to affect *only* the teeth (where `labels > 0`). Each fault type is implemented as a standalone function:
*   `generate_cavity(mesh, labels, jaw, severity)`: Simulates volume loss by translating crown vertices inward via a smooth Gaussian decay. Masks out the gum.
*   `generate_crack(mesh, labels, jaw, depth)`: Simulates a jagged structural fissure using localized vertex displacement (avoiding unreliable boolean engines). Masks out the gum.
*   `generate_abrasion(mesh, labels, jaw, wear_level)`: Simulates bruxism by identifying and planar-clipping the occlusal surfaces (highest/lowest Z depending on jaw orientation).
*   `generate_calculus(mesh, labels, jaw, buildup)`: Simulates tartar buildup near the cervical margin (gumline) via outward displacement, strictly limited to teeth vertices.

**Phase 2: Architectural Modification (LoRA & Custom Head)**
*   **LoRA Injection:** The SAM2 Hiera-L backbone remains frozen. Use `peft` to inject LoRA matrices into the Query ($Q$) and Value ($V$) projection layers within the attention blocks of the Hiera backbone (`model/sam2/modeling/backbones/hieradet.py`), as well as the self/cross-attention layers of the mask decoder (`model/sam2/modeling/sam/transformer.py`).
*   **Custom Segmentation Head:** Do not modify the original `Mask_Refiner.py`. Create a new, separate file named `model/Fault_Refiner.py` to handle the specific task of fault segmentation (which relies on depth/texture anomalies rather than object boundaries).
*   **Training Script:** Create `start_train_faults.py` alongside the original `start_train.py` to manage the Colab training loop.

## 🎯 Downstream Context: VR Haptic Dental Simulation
**Agent Note:** When generating code or troubleshooting for this project, keep the following end-goal in mind:

The outputs of this modified pipeline are critical inputs for a real-time haptic VR simulation for dental procedures. 
* **Variable Haptic Resistance:** A segmented "decay" region will be programmed with a lower density threshold in the physics engine, allowing a haptic drill to push through it more easily than healthy enamel. Conversely, a "calculus" region will require a specific "snapping" force to simulate scaling.
* **Mesh Integrity:** The generated synthetic faults must be topologically sound (manifold) for the collision algorithms to calculate accurate real-time force-feedback in the mechatronic loop.

## 📂 Expected Data Flow
1. **Input:** Raw 3D dental scans (healthy).
2. **Synthetic Generation:** Passes the mesh through one or more of the separated fault functions (`generate_cavity`, `generate_crack`, etc.) to inject procedural faults and generate ground-truth labels.
3. **Processing:** Extracts features via the LoRA-adapted SAM backbone and routes them through the new `Fault_Refiner` head.
4. **Output:** Segmented 3D meshes with specific labeled sub-regions (faults), ready for physics and haptic mapping.

## 🚀 Current Status & Agent Tasks
*(Update this section before prompting the agent with specific tasks)*

* [x] Repository cloned locally.
* [x] Architecture plan finalized (LoRA + separate head).
* [x] Draft the `trimesh` synthetic data script containing the separated fault functions (`generate_cavity`, `generate_crack`, `generate_abrasion`, `generate_calculus`).
* [ ] Write `model/Fault_Refiner.py`.
* [ ] Create `start_train_faults.py` for the Colab training loop.