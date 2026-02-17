# FMRIR Project - Active Scripts Guideline

**Generated:** February 9, 2026  
**Based on:** Recent 50 commits analysis

## Training Scripts

### `trainer-atf-3d.py`
- Main training script for the 3D flow-matching ATF prediction model.
- Handles model training, validation, checkpointing, and WandB logging with configurable hyperparameters.

## Inference Scripts

### `inference.py`
- Primary inference script for generating 3D ATF predictions from trained SFlow models.
- Supports multiple guidance scales, metric calculation (MSE, LSD), and visualization of spatial sound fields.

### `inference_1d_atf.py`
- 1D ATF inference script for microphone-specific ATF prediction.
- Generates and compares ATF predictions at specific microphone positions across multiple guidance scales.

### `inference-unet-ATF.py`
- Legacy inference script for UNet-based ATF models (pre-flow-matching).
- Maintained for backward compatibility and comparison with earlier model architectures.

## Evaluation Scripts

### `unified_evaluation.py`
- Comprehensive evaluation script comparing your model against reference methods (FSMPAE, KRR).
- Generates detailed metrics, plots, and statistical comparisons across multiple models and guidance scales.

### `clean_evaluation.py`
- Streamlined evaluation script for quick model performance assessment.
- Computes LSD and MSE metrics comparing your model with reference baselines on test sources.

### `reference_inference.py`
- Reference model inference wrapper for AUTOENCODER (FSMPAE) baseline comparison.
- Loads pre-computed reference results and facilitates direct comparison with your flow-matching model.

---

## Visualization Scripts

### `paper_figures.py`
- Generates publication-quality figures for spatial sound field visualizations and ATF comparisons.
- Creates multi-frequency slice plots, 3D visualizations, and comparison plots against reference methods.

### `standalone_atf_plotter.py`
- Standalone tool for generating ATF comparison plots using a specified model path.
- Provides quick visualization of model predictions vs ground truth and reference methods.

---

## Data Generation & Processing Scripts

### `irdata_gen_mult.py`
- Generates synthetic impulse response and ATF data using pyroomacoustics image-source method.
- Creates training/validation datasets with configurable room dimensions, RT60, and microphone/source grids.

### `process_npz_files.py`
- Batch processing script for NPZ dataset files.
- Slices ATF frequency components and reorganizes data for training pipeline compatibility.

---

## Core Utilities

### `fm_utils.py`
- Core utility module containing model architectures, samplers, and training components.
- Implements SetEncoder, UNet3D, ODE solvers, schedulers, and data processing utilities for flow matching.

### `irdata_utils.py`
- Utility functions for impulse response data processing and visualization.
- Includes MSE/LSD loss functions, ATF plotting tools, and model save/load utilities.

### `model_paths.py`
- Centralized configuration file for model checkpoint paths and multi-model evaluation.
- Manages paths to trained models for inference, evaluation, and comparison experiments.

### `modify_architecture_version.py`
- Tool for modifying architecture version metadata in saved model checkpoints.
- Enables updating model configs for compatibility with newer code versions.

---

## Recent Development Activity (Last 50 Commits)

Based on the recent commit history, the project has focused on:

- **Model architecture refinement** - UNet variants, attention mechanisms, architecture versioning
- **Training optimization** - Learning rate schedules, switching between DDPM and flow matching
- **Dataset versioning** - Source split management (r1-r4 versions), validation logic improvements
- **Microphone sampling** - Random vs fixed sampling strategies, M range configurations
- **Inference pipeline** - Multiple guidance scales, boundary exclusion, interior masking
- **Evaluation framework** - Metric calculations, multi-model comparison, statistical analysis
- **Visualization** - Paper figure generation, ATF plotting improvements

### Most Frequently Modified Files:
1. `fm_utils.py` - Core model utilities and architectures
2. `cmds.py` - Command-line interface (not in main directory)
3. `trainer-atf-3d.py` - Training pipeline
4. `inference.py` - Inference and evaluation
5. `model_paths.py` - Model configuration paths

---

## Quick Start Guide

### Training a New Model
```bash
python trainer-atf-3d.py [args]
```
Configure model architecture, dataset version, and training hyperparameters. The script handles checkpointing, validation, and WandB logging automatically.

### Running Inference
1. Set model path in `model_paths.py`
2. Run: `python inference.py`
3. Configure guidance scales and timesteps as needed

### Model Evaluation
```bash
# Comprehensive evaluation with reference comparison
python unified_evaluation.py

# Quick metric computation
python clean_evaluation.py
```

### Generating Training Data
```bash
python irdata_gen_mult.py --num_src 1024 --int_mic 0.1
```
Configure room dimensions, RT60, and source/microphone positions in the script.

### Creating Paper Figures
```bash
python paper_figures.py
```
Set model path and visualization parameters in the script.

---

## Project Structure

```
FMRIR/
├── artifacts/              # Trained model checkpoints
├── AUTOENCODER/           # Reference model implementation (FSMPAE)
├── eval/                  # Evaluation results and metrics
├── paper_figures/         # Generated publication figures
├── legacy/                # Deprecated scripts (maintained for reference)
├── tests/                 # Test scripts (MNIST DDPM sanity checks)
├── *.py                   # Main scripts (see detailed descriptions above)
├── *.sh                   # Job submission scripts for cluster
└── requirements.txt       # Python dependencies
```

---

## Key Directories

- **`artifacts/`** - Contains trained model checkpoints organized by experiment name
- **`AUTOENCODER/`** - Reference baseline implementation (FSMPAE model from prior work)
- **`eval/`** - Stores evaluation outputs and comparison results
- **`paper_figures/`** - Publication-ready figures and visualizations
- **`legacy/`** - Older scripts kept for backward compatibility
- **`tests/mnist_&_ddpm/`** - DDPM sanity check scripts using MNIST

---

## Dataset Versions

The project uses multiple dataset versions (r1-r4):
- **r1** - Original 1024 source dataset
- **r4** - Extended 8192 source dataset
- Each version has specific source splits for train/val/test

Dataset configuration is managed through `get_dataset_version_from_model_name()` in `fm_utils.py`.

---

## Notes

- This guideline was automatically generated based on code analysis and commit history
- For the most up-to-date information, refer to the codebase documentation and inline comments
- Most scripts support command-line arguments; use `--help` for details

---

**End of Document**
