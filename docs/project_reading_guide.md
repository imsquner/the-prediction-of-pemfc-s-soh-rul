# Project Reading Guide

## Goal

Help new contributors understand the minimum path from data to prediction results.

## Recommended Reading Order

1. `README.md`
   - Understand run commands, output directories, and retained artifacts.
2. `train.py`
   - Main orchestration for data loading, preprocessing, model build, evaluation, and reporting.
3. `model.py`
   - GRU model definition and forward logic.
4. `data_processing.py` and `data_processors.py`
   - Data cleaning, feature handling, sequence generation, and scaling.
5. `predict_visualize.py`
   - Additional plotting and prediction utilities (if needed).
6. `gui/`
   - Optional GUI workflow; not required for CLI training/evaluation path.

## Runtime Path (CLI)

1. `train.py` loads latest FC1 processed npz from `processed_results/FC1/`.
2. Data processor builds train/val/test sequences and scalers.
3. Model is constructed and checkpoint is loaded in eval-only mode.
4. Metrics, plots, and final reports are written to `train_results_paper/...`.
5. FC2 inference uses latest file under `processed_results/FC2/`.

## Output Artifacts You Should Care About

- Metrics table:
  - `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/tables/metrics_overall.csv`
- Prediction CSV:
  - `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/csv_files/predictions.csv`
- Final report:
  - `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/configs/final_report.json`
  - `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/configs/final_report.txt`
- Best checkpoint:
  - `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/models/best_model.pth`

## What Is Considered Regenerable Intermediate Data

- Build/cache artifacts (`build/`, `__pycache__/`, temporary logs)
- Historical processed snapshots older than latest FC1/FC2 pairs
- Historical training logs beyond recent troubleshooting windows
- Historical catboost reports older than latest analysis batch

## Naming Suggestions For Future Files

- Python modules: `snake_case.py`
- Reports/charts/results: `artifact_YYYYMMDD_HHMMSS.ext`
- Keep docs under `docs/` and avoid root-level scratch text files
