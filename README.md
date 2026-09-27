

## Repository layout
```
data/        raw CSVs (committed so results never depend on upstream changes);
             mackey_glass.csv is generated on first run
src/data.py      loading, missing values, log/diff transforms, z-scaling, windows, ADF/KPSS
src/cells.py     ElmanCell, JordanCell, MRNCell + SimpleRNN wrapper (shared output layer)
src/cv.py        expanding-window CV folds + chronological early-stopping split
src/train.py     Adam/MSE training, early stopping, weight decay, grad clipping,
                 random search over CV folds, final 10-seed runs
src/evaluate.py  RMSE/MAE/sMAPE, persistence baseline, Friedman/Nemenyi, Wilcoxon-Holm
src/figures.py   all figures (PDF)            src/tables.py  LaTeX tables for the report
tests/test_cells.py  unit tests (equations vs NumPy reference, fits a sine)
results/     cached tuning JSONs, final-run .npz files, summary.json
figures/     report figures              report/  PDF
run_all.py   one command for everything
```

## Running on Windows (WSL) or Linux
```bash
sudo apt update
sudo apt install -y python3-venv
# 3. Put the project in your Linux home (not /mnt/c: much slower), then:
cd ~/rnn_ts
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m unittest -v tests.test_cells
python run_all.py --quick

python run_all.py --force 
```
Without `--force`, cached results in `results/` are reused and only the
statistics, figures, tables and PDF are regenerated (seconds).

## Reproducibility note
Seeds are fixed (tuning: seed 0; final runs: seeds 0-9) and PyTorch runs
single-threaded. Identical numbers are expected on the same library versions;
different PyTorch/BLAS versions can change results in the last digits.
