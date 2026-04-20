# Environment Setup

## 1. Python Version

- Recommended: Python 3.10+
- Verified in this workspace: Python 3.12 (.venv)

## 2. Create and Activate Virtual Environment (PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Quick Verification

Run evaluation-only mode to verify the project can execute end-to-end:

```powershell
$env:EVAL_ONLY="1"
& .\.venv\Scripts\python.exe .\train.py
```

Expected outputs are generated under:

- `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/csv_files/`
- `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/images/`
- `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/configs/`

## 5. Full Training

```powershell
Remove-Item Env:EVAL_ONLY -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe .\train.py
```

## 6. Common Notes

- Current model checkpoint path:
  - `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/models/best_model.pth`
- Data files are auto-selected by latest timestamp under:
  - `processed_results/FC1/`
  - `processed_results/FC2/`
