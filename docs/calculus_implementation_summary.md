# 3DTeethSAM Calculus Extension: Implementation & Bug Fix Log

This document serves as the official technical summary of the Calculus Segmentation Head implementation, including the architectural design, environmental fixes, and advanced tooling created for Google Colab resilience.

## 1. Architectural Design & Implementation

We successfully built a completely standalone branch of the SAM2 pipeline specifically tuned for binary calculus segmentation. By isolating the calculus task, the existing 17-class tooth instance segmentation remains entirely untouched and frozen.

### 1.1 Core Network Components (`model/calculus/`)
* **`CalculusPEG`** (`calculus_prompt_generator.py`): A lightweight prompt generator using a 3-layer Transformer Decoder with 4 learnable queries (reduced from the 16 used for multi-class teeth).
* **`CalculusRefiner`** (`calculus_head.py`): A binary ResUNet that fuses high-resolution image features with SAM2's coarse masks. Optimized by halving the channel widths (`32→64→128→256`) to save VRAM and speed up inference.
* **`CalculusSegmentationSystem`** (`calculus_system.py`): The orchestrator that loads SAM2, freezes it, injects LoRA, runs the PEG, and passes the outputs through the Refiner.
* **`LoRA`** (`lora.py`): A custom injection script specifically adapted for SAM2's Hiera attention blocks (`q_proj`, `v_proj`).
* **`CalculusLoss`** (`loss_calculus.py`): A composite loss function combining Binary Focal Loss (to counteract the massive calculus class imbalance), Soft Dice Loss, and Differentiable Boundary Loss.

### 1.2 Data & Training Pipeline (`model/dataset/`, `model/trainer_calculus.py`)
* **`HDF5CalculusDataset`**: A robust PyTorch Dataset that pulls multi-view images and binary masks from HDF5 lazily to prevent memory exhaustion.
* **`CalculusTrainer`**: A full, single-stage training loop supporting Automatic Mixed Precision (AMP), gradient clipping, early stopping, and tracking of Binary Dice/IoU metrics.
* **`config_calculus.py`**: A centralized configuration file routing all hyperparameter logic.

---

## 2. Environmental & Google Colab Fixes

During the initial deployment in Google Colab (using T4 GPUs and Python 3.13), several obscure bugs were encountered and successfully bypassed.

### 2.1 PyTorch & CUDA Fixes
* **T4 GPU Deadlocks (`bfloat16`)**: Discovered that T4 GPUs silently freeze when PyTorch AMP attempts to use `bfloat16`. Fixed by permanently forcing AMP down to `float16` in the configuration.
* **Double Precision Inference Bug (`float64`)**: PyTorch3D loaded `.obj` meshes into `float64`, which crashed the SAM2 convolutions upon projection. Fixed by explicitly casting `verts.float()` and `images_tensor.float()` in `inference_calculus.py`.
* **CUDA Out Of Memory (OOM)**: SAM2-Large with the CalculusHead and `batch_size=4` at 512x512 exceeded the 15GB VRAM of the T4 GPU. Lowered `batch_size` to 2.

### 2.2 Multiprocessing & Jupyter Fixes
* **Python 3.13 Fork Safety Crashing**: Fixed the `filelock` `_audit_fork_safety` crash by forcing `num_workers = 0`, preventing PyTorch from calling `os.fork()` while the `pin_memory` thread is active.
* **HDF5 Shared Memory Deadlocks**: Moved the `h5py.File` initialization from the dataset's `__init__` into `__getitem__` to prevent multiprocessing locks from freezing the training script.
* **Jupyter Logging Silencing**: Colab notebook cells hijacked the root logger and swallowed all progress text. Fixed by explicitly attaching a `StreamHandler` to `sys.stdout` directly in the training block.
* **Colab Tqdm Spam**: Purged `tqdm` progress bars from the trainer to prevent it from flooding the Colab output with thousands of un-collapsed carriage returns.

---

## 3. Advanced Tooling & Resilience

### 3.1 Google Drive "Smart Auto-Resume"
To protect against Google Colab's random disconnects and ephemeral storage, custom training cells were created that:
1. Automatically mount Google Drive via `google.colab.drive`.
2. Route all `save_dir` checkpoints directly to Drive.
3. Feature a **Smart Auto-Resume** script that scans the Drive folder on startup, finds the latest `interrupted_checkpoint.pth` (saved automatically via a graceful `KeyboardInterrupt` catch), and resumes training seamlessly.

### 3.2 3D Inference & Diagnostic Visualization
* **`CalculusInferencePipeline`** (`inference_calculus.py`): Natively loads raw `.obj` files, unprojects them to 7 2D views, runs SAM2, re-projects the binary predictions back onto the 3D vertices using depth-weighting, and exports a final `.json` fault map.
* **Plotly Error-Analysis Viewer**: A custom interactive Colab 3D visualizer was built. It loads the raw `.obj`, reads the predicted labels alongside the ground-truth `.json`, and color-codes the mesh to visually debug the AI's accuracy directly inside the Jupyter notebook:
  * 🟢 **Green**: True Positive (Correctly detected calculus)
  * 🔴 **Red**: False Positive (Hallucinated)
  * 🟠 **Orange**: False Negative (Missed calculus)
  * ⚪ **Grey**: True Negative (Normal healthy tooth)
