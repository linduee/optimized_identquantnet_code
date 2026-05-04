# Optimized IdentQuantNet Code

This folder contains a cleaned working bundle for the code associated with:

`Optimized IdentQuantNet: A machine learning-based approach for identification and quantification of multiple drugs with interaction on electrochemical sensors in personalized medicine`

It includes:

- `optimized_identquantnet_inference.py`
  Loads pretrained weights and runs the extracted main workflow for inference / evaluation.
- `optimized_identquantnet_training.py`
  Retrains and saves model weights, including the weighted quantification models.

## Purpose

This bundle is intended to separate two common use cases:

- `optimized_identquantnet_inference.py`: reuse existing pretrained weights to validate or run the pipeline
- `optimized_identquantnet_training.py`: retrain the models and generate new weight files

## Folder Structure

```text
optimized_identquantnet_code/
├── optimized_identquantnet_inference.py
├── optimized_identquantnet_training.py
├── identification_model.py
├── optimized_identquantnet_pretrained/
├── model_save_trial/
├── README.md
├── .gitignore
└── requirements.txt
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How To Run

### 1. Run the extracted main workflow

This script uses existing weights and is meant for validation / inference:

```bash
python optimized_identquantnet_inference.py
```

By default it looks for pretrained checkpoints in:

```text
optimized_identquantnet_code/optimized_identquantnet_pretrained/
```

### 2. Retrain and save new weights

This script retrains the models and saves new checkpoints:

```bash
python optimized_identquantnet_training.py
```

By default it writes new checkpoints to:

```text
optimized_identquantnet_code/model_save_trial/
```

## Important Notes

### 1. Path handling

The scripts no longer rely on hard-coded absolute local paths for results directories.

For dataset overrides, you can optionally define:

- `OPTIMIZED_IDENTQUANTNET_DATASET_DIR`

If this variable is not set, the scripts fall back to the local `data/` directory inside this bundle.

### 2. Local helper module

The bundle keeps a local copy of `identification_model.py`, which is imported directly by both main scripts.

### 3. Pretrained vs retrained weights

- `optimized_identquantnet_inference.py` is for using previously generated weights
- `optimized_identquantnet_training.py` is for generating new weights

The inference script reads from `optimized_identquantnet_pretrained/`.

The training script writes to `model_save_trial/` by default, so newly trained results do not overwrite the pretrained bundle.

## Main Dependencies

- Python 3.8+
- numpy
- pandas
- torch
- matplotlib
- scikit-learn
- scipy
- joblib
- openpyxl


